import functools
from contextlib import ExitStack
import os
import time
import json
from argparse import ArgumentParser
from typing import Tuple, List, Optional, Sequence, Any

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from tqdm import tqdm
import imageio
import numpy as np

import tinycudann as tcnn

import tc_plenoxel
from tc_plenoxel import Grid

torch.manual_seed(0)
np.random.seed(0)


def initialize_grid(resolution, ini_rgb=0.0, ini_sigma=0.1, harmonic_degree=0, device=None,
                    dtype=torch.float32, init_indices=True) -> Grid:
    """
    :param resolution:
    :param ini_rgb: Initial value in the spherical harmonics
    :param ini_sigma: Initial value for density sigma
    :param harmonic_degree:
    :return:
        Tuple containing the indices of each voxel in the grid, and the data contained in each voxel.
        The data contains the RGB values of the spherical harmonics, and the value for density sigma.
    """
    sh_dim = (harmonic_degree + 1) ** 2
    total_data_channels = sh_dim * 3 + 1

    data = torch.full((resolution ** 3, total_data_channels), ini_rgb, dtype=dtype, device=device)
    data[:, -1].fill_(ini_sigma)

    if not init_indices:
        return Grid(grid=data, indices=None)

    indices = torch.arange(resolution ** 3, dtype=torch.long, device=device).reshape(
        (resolution, resolution, resolution))
    return Grid(grid=data, indices=indices)


def get_data(root: str, stage: str, max_frames: Optional[int] = None) -> Tuple[
    float, torch.Tensor, torch.Tensor]:
    all_c2w = []
    all_gt = []

    data_path = os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    j = json.load(open(data_json, 'r'))
    if max_frames is None:
        max_frames = len(j['frames'])

    for frame in tqdm(j['frames'][:max_frames]):
        fpath = os.path.join(data_path, os.path.basename(frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = torch.from_numpy(np.asarray(all_gt))
    all_c2w = torch.from_numpy(np.asarray(all_c2w))
    return focal, all_c2w, all_gt


def get_rays(H: int, W: int, focal, c2w) -> torch.Tensor:
    """

    :param H:
    :param W:
    :param focal:
    :param c2w:
    :return:
        Tensor of size [2, W, H, 3] where the first dimension indexes origin and direction
        of rays
    """
    i, j = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing='xy')
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return torch.stack((rays_o, rays_d), dim=0)


def get_training_set(train_gt, train_c2w, img_h, img_w, focal, resolution, device, dtype):
    # Generate training rays
    rays = torch.stack(
        [get_rays(img_h, img_w, focal, p) for p in train_c2w[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    # Merge N, H, W dimensions
    rays = rays.permute(0, 2, 3, 1, 4).reshape(-1, 2, 3)  # [N*H*W, ro+rd, 3]
    rays = rays.to(dtype=dtype)

    # Resize the training images
    lowp_imgs = multi_lowpass(train_gt, resolution)  # [N, H, W, 3]
    # Merge N, H, W dimensions
    lowp_imgs = lowp_imgs.reshape(-1, 3)  # [N*H*W, 3]
    lowp_imgs = lowp_imgs.to(dtype=dtype)

    if device != "cpu":
        rays = rays.pin_memory()
        lowp_imgs = lowp_imgs.pin_memory()

    return rays, lowp_imgs


def multi_lowpass(gt: torch.Tensor, resolution: int) -> torch.Tensor:
    """
    low-pass filter a stack of images where the first dimension indexes over the images

    :param gt:
        Tensor of size: [N, H, W, 3]
    :param resolution:
    :return:
    """
    if gt.dim() <= 3:
        print(
            f'multi_lowpass called on image with 3 or fewer dimensions; did you mean to use lowpass instead?')
    H = gt.shape[1]
    W = gt.shape[2]
    clean_gt = torch.empty_like(gt)
    to_pil = transforms.ToPILImage()
    to_pyt = transforms.ToTensor()
    for i in range(len(gt)):
        im = to_pil(gt[i].permute(2, 0, 1))  # Move channels first
        im = im.resize(size=(resolution * 2, resolution * 2))
        im = im.resize(size=(H, W))
        clean_gt[i, ...] = to_pyt(im).permute(1, 2, 0)  # Move channels last
    return clean_gt


# @torch.jit.script
def train_batch(params: Any, params_type: str, target: torch.Tensor, rays: torch.Tensor,
                resolution: int,
                radius: float,
                harmonic_degree: int,
                jitter: bool, uniform: float,
                occupancy_penalty: float,
                interpolation: str,
                sh_encoder,
                lrs):
    rays_o, rays_d = rays[:, 0], rays[:, 1]
    if params_type == "irregular_grid":
        grid_data, grid_idx = params
        intrp_w, neighbor_ids, intersections = tc_plenoxel.fetch_intersections(
           grid_idx, rays_o=rays_o, rays_d=rays_d, resolution=resolution, radius=radius,
           uniform=uniform, interpolation=interpolation)
        rgb = tc_plenoxel.ComputeIntersection.apply(grid_data, neighbor_ids, intrp_w, rays_d,
                                                    intersections, harmonic_degree)
        loss = F.mse_loss(rgb, target) + occupancy_penalty * torch.mean(torch.relu(grid_data[..., -1]))
        grads = torch.autograd.grad(loss, grid_data)
        upd_data = update_grids(grid_data, lrs, grads[0])
        del rgb, upd_data, grads, loss, target, rays
    elif params_type == "grid":
        grid_data = params
        rgb = tc_plenoxel.compute_intersection_results(
            grid_data=grid_data, rays_d=rays[0], rays_o=rays[1], radius=radius, resolution=resolution,
            uniform=uniform, harmonic_degree=harmonic_degree, sh_encoder=sh_encoder, white_bkgd=True)
        loss = F.mse_loss(rgb, target) + occupancy_penalty * torch.mean(torch.relu(grid_data[..., -1]))
        grads = torch.autograd.grad(loss, grid_data)
        upd_data = update_grids(grid_data, lrs, grads[0])
        del rgb, upd_data, grads, loss, target, rays
    elif params_type == "mr_hash_grid":
        hg = params
        rgb = tc_plenoxel.compute_with_hashgrid(
            hg=hg, rays_d=rays[0], rays_o=rays[1], radius=radius, resolution=resolution,
            uniform=uniform, harmonic_degree=harmonic_degree, sh_encoder=sh_encoder, white_bkgd=True)
        # TODO: add regularization
        loss = F.mse_loss(rgb, target)# + occupancy_penalty * torch.mean(torch.relu(grid_data[..., -1]))
        grads = torch.autograd.grad(loss, hg.parameters())
        # TODO: Update parameters
    else:
        raise ValueError(params_type)
    return None


@torch.jit.script
def update_grids(grid_data: torch.Tensor,
                 lrs: torch.Tensor,
                 grid_grad: torch.Tensor) -> torch.Tensor:
    """

    :param grid_data:
        The input grid: [n_voxels, n_channels]
    :param lrs:
        The learning rate, one per channel group (sh have 3 channels per group, density has 1)
    :param grid_grad:
        The gradient (same shape as grid_data)
    :return:
        The new grid, updated by a gradient descent step.
    """
    lrs_shape = [1] * (grid_data.dim() - 1) + [lrs.shape[0]]
    return grid_data.sub_(grid_grad * lrs.view(lrs_shape))


# def run_test_step(i, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, key, name_appendage=''):
#     print('Evaluating')
#     sh_dim = (FLAGS.harmonic_degree + 1)**2
#     tpsnr = 0.0
#     for j, (c2w, gt) in tqdm(enumerate(zip(test_c2w, test_gt))):
#         rgb, disp, _, _ = render_pose_rays(data_dict, c2w, H, W, focal, FLAGS.resolution, FLAGS.radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, key, sh_dim, FLAGS.physical_batch_size, FLAGS.interpolation)
#         mse = jnp.mean((rgb - gt)**2)
#         psnr = -10.0 * np.log(mse) / np.log(10.0)
#         tpsnr += psnr
#
#         if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
#             disp3 = jnp.concatenate((disp[...,jnp.newaxis], disp[...,jnp.newaxis], disp[...,jnp.newaxis]), axis=2)
#             vis = jnp.concatenate((gt, rgb, disp3), axis=1)
#             vis = np.asarray((vis * 255)).astype(np.uint8)
#             imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", vis)
#         del rgb, disp
#     tpsnr /= n_test_imgs
#     return tpsnr


def init_params(params_type, resolution, deg, **kwargs):
    if params_type == "irregular_grid":
        grid = initialize_grid(resolution, harmonic_degree=deg, device=kwargs['dev'],
                               dtype=torch.float32, init_indices=True,
                               ini_sigma=kwargs['ini_sigma'], ini_rgb=kwargs['ini_rgb'])
        grid_data, grid_idx = grid.grid, grid.indices
        grid_data.requires_grad_()
        return grid_data, grid_idx
    elif params_type == "grid":
        grid = initialize_grid(resolution, harmonic_degree=deg, device=kwargs['dev'],
                               dtype=torch.float32, init_indices=False,
                               ini_sigma=kwargs['ini_sigma'], ini_rgb=kwargs['ini_rgb'])
        grid_data = grid.grid
        grid_data.requires_grad_()
        return grid_data
    elif params_type == "mr_hash_grid":
        assert deg == 2
        n_levels = 14  # we need 28 channels
        base_res = 16
        per_level_scale = resolution / (n_levels * base_res)
        hg = tcnn.Encoding(3, {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,
            "n_features_per_level": 2,  # Total of 10*3 = 30 features
            "log2_hashmap_size": kwargs['log2_hashmap_size'],
            "base_resolution": base_res,  # Resolution of the coarsest level
            "per_level_scale": per_level_scale,  # How much the resolution increases at each level
            "interpolation": "Linear",
        })
        # TODO: Maybe initialize parameters
        return hg
    else:
        raise ValueError(params_type)


def trace_handler(p, exp_name, dev):
    print(p.key_averages().table(sort_by=f"self_{dev}_time_total", row_limit=10))
    print(p.key_averages().table(sort_by=f"self_{dev}_memory_usage", row_limit=10))
    # p.export_chrome_trace(f"./trace_{exp_name}_{p.step_num}.json")
    # p.export_stacks(f"./profiler_stacks_{exp_name}_{p.step_num}.txt", f"self_{dev}_time_total")
    torch.profiler.tensorboard_trace_handler(f"tb_logs/{exp_name}")(p)


def run(args):
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda:0"
    log_dir = args.log_dir + args.expname
    os.makedirs(log_dir, exist_ok=True)

    focal, train_c2w, train_gt = get_data(args.data_dir, "train", max_frames=10)
    test_focal, test_c2w, test_gt = get_data(args.data_dir, "test", max_frames=10)

    H, W = train_gt[0].shape[:2]

    if args.lr_rgb is None or args.lr_sigma is None:
        args.lr_rgb = 150 * (args.resolution ** 1.75)
        args.lr_sigma = 51.5 * (args.resolution ** 2.37)

    print(f'Initializing the grid')
    params = init_params(params_type=args.params_type,
                         resolution=args.resolution,
                         deg=args.harmonic_degree,
                         ini_sigma=args.ini_sigma,
                         ini_rgb=args.ini_rgb,
                         dev=dev,
                         log2_hashmap_size=args.log2_hashmap_size)

    tr_rays, tr_rgb = get_training_set(train_gt=train_gt, train_c2w=train_c2w, img_h=H, img_w=W,
                                       focal=focal, resolution=args.resolution,
                                       dtype=torch.float32, device=dev)
    n_samples = tr_rays.shape[0]

    occupancy_penalty = args.occupancy_penalty / (n_samples // args.batch_size)
    sh_dim = (args.harmonic_degree + 1) ** 2
    lrs = [args.lr_rgb] * (sh_dim * 3) + [args.lr_sigma]
    lrs = torch.tensor(lrs, dtype=torch.float32, device=dev)

    # Tiny-CUDA-NN modules
    sh_enc = tcnn.Encoding(3, {
        "otype": "SphericalHarmonics",
        "degree": args.harmonic_degree + 1,
    })

    profiling_handler = functools.partial(trace_handler, exp_name=args.expname,
                                          dev="cpu" if dev == "cpu" else "cuda")
    # Main iteration starts here
    for i in range(args.num_epochs):
        print("Starting epoch %d" % (i))
        # Shuffle rays over all training images
        epoch_perm = np.random.permutation(n_samples)
        tr_rays = tr_rays[epoch_perm]
        tr_rgb = tr_rgb[epoch_perm]

        with ExitStack() as stack:
            p = None
            if args.profile:
                p = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=5, warmup=5, active=15),
                    on_trace_ready=profiling_handler,
                    with_stack=False, profile_memory=True, record_shapes=True)
                stack.enter_context(p)
            for k in tqdm(range(n_samples // args.batch_size)):
                t_s = time.time()
                b_rays = tr_rays[k * args.batch_size: (k + 1) * args.batch_size].to(device=dev)
                b_rgb = tr_rgb[k * args.batch_size: (k + 1) * args.batch_size].to(device=dev)
                t_s = time.time()
                loss, grid_data = train_batch(params=params, params_type=args.params_type,
                                              target=b_rgb, rays=b_rays, resolution=args.resolution,
                                              radius=args.radius, harmonic_degree=args.harmonic_degree,
                                              jitter=args.jitter, uniform=args.uniform,
                                              occupancy_penalty=occupancy_penalty,
                                              interpolation=args.interpolation, lrs=lrs,
                                              sh_encoder=sh_enc)
                t_e = time.time()
                print(f"Iteration takes {t_e - t_s:.4f}s")
                t_s = time.time()
                if p is not None:
                    p.step()
                print("\n\n")

            # if i % args.val_interval == args.val_interval - 1 or i == args.num_epochs - 1:
            #     validation_psnr = run_test_step(i + 1, data_dict, test_c2w, test_gt, H, W, focal, FLAGS,
            #                                     render_keys)
            #     print(f'at epoch {i}, test psnr is {validation_psnr}')


def parse_args():
    flags = ArgumentParser()

    flags.add_argument(
        "--data_dir", '-d',
        type=str,
        default='./nerf/data/nerf_synthetic/',
        help="Dataset directory e.g. nerf_synthetic/"
    )
    flags.add_argument(
        "--expname",
        type=str,
        default="experiment",
        help="Experiment name."
    )
    flags.add_argument(
        "--scene",
        type=str,
        default='lego',
        help="Name of the synthetic scene."
    )
    flags.add_argument(
        "--log_dir",
        type=str,
        default='jax_logs/',
        help="Directory to save outputs."
    )
    flags.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Grid size."
    )
    flags.add_argument(
        "--ini_rgb",
        type=float,
        default=0.0,
        help="Initial harmonics value in grid."
    )
    flags.add_argument(
        "--ini_sigma",
        type=float,
        default=0.1,
        help="Initial sigma value in grid."
    )
    flags.add_argument(
        "--radius",
        type=float,
        default=1.3,
        help="Grid radius. 1.3 works well on most scenes, but ship requires 1.5"
    )
    flags.add_argument(
        "--harmonic_degree",
        type=int,
        default=2,
        help="Degree of spherical harmonics. Supports 0, 1, 2, 3, 4."
    )
    flags.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='Epochs to train for.'
    )
    flags.add_argument(
        '--render_interval',
        type=int,
        default=40,
        help='Render images during test/val step every x images.'
    )
    flags.add_argument(
        '--val_interval',
        type=int,
        default=2,
        help='Run test/val step every x epochs.'
    )
    flags.add_argument(
        '--lr_rgb',
        type=float,
        default=None,
        help='SGD step size for rgb. Default chooses automatically based on resolution.'
    )
    flags.add_argument(
        '--lr_sigma',
        type=float,
        default=None,
        help='SGD step size for sigma. Default chooses automatically based on resolution.'
    )
    flags.add_argument(
        '--physical_batch_size',
        type=int,
        default=4000,
        help='Number of rays per batch, to avoid OOM.'
    )
    flags.add_argument(
        '--logical_batch_size',
        type=int,
        default=4000,
        help='Number of rays per optimization batch. Must be a multiple of physical_batch_size.'
    )
    flags.add_argument(
        '--batch_size',
        type=int,
        default=4000,
        help='Number of rays per optimization batch. Must be a multiple of physical_batch_size.'
    )
    flags.add_argument(
        '--jitter',
        type=float,
        default=0.0,
        help='Take samples that are jittered within each voxel, where values are computed with trilinear interpolation. Parameter controls the std dev of the jitter, as a fraction of voxel_len.'
    )
    flags.add_argument(
        '--uniform',
        type=float,
        default=0.5,
        help='Initialize sample locations to be uniformly spaced at this interval (as a fraction of voxel_len), rather than at voxel intersections (default if uniform=0).'
    )
    flags.add_argument(
        '--occupancy_penalty',
        type=float,
        default=0.0,
        help='Penalty in the loss term for occupancy; encourages a sparse grid.'
    )
    flags.add_argument(
        '--reload_epoch',
        type=int,
        default=None,
        help='Epoch at which to resume training from a saved model.'
    )
    flags.add_argument(
        '--save_interval',
        type=int,
        default=1,
        help='Save the grid checkpoints after every x epochs.'
    )
    flags.add_argument(
        '--prune_epochs',
        type=int,
        nargs='+',
        default=[],
        help='List of epoch numbers when pruning should be done.'
    )
    flags.add_argument(
        '--prune_method',
        type=str,
        default='weight',
        help='Weight or sigma: prune based on contribution to training rays, or opacity.'
    )
    flags.add_argument(
        '--prune_threshold',
        type=float,
        default=0.001,
        help='Threshold for pruning voxels (either by weight or by sigma).'
    )
    flags.add_argument(
        '--split_epochs',
        type=int,
        nargs='+',
        default=[],
        help='List of epoch numbers when splitting should be done.'
    )
    flags.add_argument(
        '--interpolation',
        type=str,
        default='trilinear',
        help='Type of interpolation to use. Options are constant, trilinear, or tricubic.'
    )
    flags.add_argument(
        '--params_type',
        type=str,
        default='irregular_grid',
        help="Type of exp. Can be 'grid', 'irregular_grid', 'mr_hash_grid'"
    )
    flags.add_argument(
        '--log2_hashmap_size',
        type=int,
        default=19,
        help="Only relevant for mr_hash_grid experiment"
    )
    flags.add_argument(
        '--profile',
        action='store_true',
        help='activates pytorch profiling'
    )

    args = flags.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.scene)
    args.radius = args.radius
    return args


if __name__ == "__main__":
    _cmd_args = parse_args()
    run(_cmd_args)
