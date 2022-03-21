import functools
from contextlib import ExitStack
import os
import time
import json
from argparse import ArgumentParser
from typing import Tuple, List, Optional, Sequence

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
                    dtype=torch.float32) -> Grid:
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

    data = torch.full((resolution ** 3, total_data_channels), ini_rgb, dtype=torch.float32,
                      device=device)
    data[:, -1].fill_(ini_sigma)

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
def train_batch(grid_idx: torch.Tensor,
               grid_data: torch.Tensor,
               rays: Tuple[torch.Tensor, torch.Tensor],
               gt: torch.Tensor,
               resolution: int,
               radius: float,
               harmonic_degree: int,
               jitter: bool, uniform: float,
               occupancy_penalty: float,
               interpolation: str,
                sh_encoder,
               lrs):
    """
    Compute the rendered rays, and the loss

    :param grid:
        Grid object. Contains grid indices, and the data (RGB+density) inside the grid.
    :param rays:
        Tuple containing ray origins and directions. Each ray is a 3D vector.
    :param gt:
        Stack of images for which to compute the rays. Has size [num_pixels, 3]
    :param resolution:
        The resolution of the images
    :param radius:
    :param harmonic_degree:
    :param jitter:
        Unused, delete
    :param uniform:
        Unused, delete
    :param occupancy_penalty:
        Penalty on the density.
    :param interpolation:
        Type of interpolation, should always be 'trilinear'
    :return:
    """
    print("loss start. %.2fGB" % (torch.cuda.memory_allocated() / 2**30))
    if not grid_data.requires_grad:
        grid_data.requires_grad_()
    #with torch.autograd.no_grad():
    #    t_s = time.time()
    #    intrp_w, neighbor_ids, intersections = tc_plenoxel.fetch_intersections(
    #        grid_idx, rays_o=rays[0], rays_d=rays[1], resolution=resolution, radius=radius,
    #        uniform=uniform, interpolation=interpolation)
    #    t_inters = time.time() - t_s
    #print("intersections done. %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

    #t_s = time.time()
    # rgb, disp, acc, weights = tc_plenoxel.compute_intersection_results(
    #     interp_weights=intrp_w, neighbor_data=neighbor_data, rays_d=rays[1],
    #     intersections=intersections, harmonic_degree=harmonic_degree)
    # rgb = tc_plenoxel.ComputeIntersection.apply(grid_data, neighbor_ids, intrp_w, rays[1], intersections, harmonic_degree)
    rgb = tc_plenoxel.compute_intersection_results(
        grid_data=grid_data, rays_d=rays[0], rays_o=rays[1], radius=radius, resolution=resolution,
        uniform=uniform, harmonic_degree=harmonic_degree, sh_encoder=sh_encoder, white_bkgd=True)
    # t_res = time.time() - t_s
    t_s = time.time()
    loss = F.mse_loss(rgb, gt) + occupancy_penalty * torch.mean(torch.relu(grid_data[..., -1]))
    t_loss = time.time() - t_s

    with torch.autograd.no_grad():
        t_s = time.time()
        grads = torch.autograd.grad(loss, grid_data)
        print("after grad %.2fGB" % (torch.cuda.memory_allocated() / 2**30))
        t_g = time.time() - t_s
        t_s = time.time()
        upd_data = update_grids(grid_data, lrs, grads[0])
        print("after update %.2fGB" % (torch.cuda.memory_allocated() / 2**30))
        t_u = time.time() - t_s
    #print(f"Intersections {t_inters*1000:.2f}ms    diff {t_res*1000:.2f}ms    loss {t_loss*1000:.2f}ms    grad {t_g*1000:.2f}ms   update {t_u*1000:.2f}ms")
    del rgb, upd_data, grads, loss, gt
    print("at end %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

    return None, grid_data


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
    grid = initialize_grid(resolution=args.resolution,
                           ini_rgb=args.ini_rgb,
                           ini_sigma=args.ini_sigma,
                           harmonic_degree=args.harmonic_degree,
                           device=dev,
                           dtype=torch.float32)

    print(f'precomputing all the training rays')
    # Precompute all the training rays
    rays = torch.stack([get_rays(H, W, focal, p) for p in train_c2w[:, :3, :4]],
                       0)  # [N, ro+rd, H, W, 3]
    lowp_imgs = multi_lowpass(train_gt, args.resolution)  # [N, H, W, 3]
    rays_rgb = torch.cat((rays, lowp_imgs.unsqueeze(1)), dim=1)  # [N, ro+rd+rgb, H, W,   3]
    rays_rgb = torch.permute(rays_rgb, (0, 2, 3, 1, 4))  # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = torch.reshape(rays_rgb, (-1, 3, 3))  # [N*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.to(dtype=torch.float32)
    if dev != "cpu":
        rays_rgb = rays_rgb.pin_memory()

    start_epoch = 0
    sh_dim = (args.harmonic_degree + 1) ** 2

    occupancy_penalty = args.occupancy_penalty / (len(rays_rgb) // args.batch_size)
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
    for i in range(start_epoch, args.num_epochs):
        # Shuffle rays over all training images
        rays_rgb = rays_rgb[np.random.permutation(rays_rgb.shape[0])]

        print("Starting epoch %d" % (i))

        with ExitStack() as stack:
            p = None
            if False:
                p = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=5, warmup=5, active=15),
                    on_trace_ready=profiling_handler,
                    with_stack=False, profile_memory=True, record_shapes=True)
                stack.enter_context(p)
            for k in tqdm(range(len(rays_rgb) // args.batch_size)):
                t_s = time.time()
                batch = rays_rgb[k * args.batch_size: (k + 1) * args.batch_size]
                batch = batch.to(device=dev)
                batch_rays, target_s = (batch[:, 0, :], batch[:, 1, :]), batch[:, 2, :]
                t_e = time.time()
                print(f"Data loading takes {t_e - t_s:.4f}s")
                t_s = time.time()
                loss, grid_data = train_batch(grid_idx=grid.indices, grid_data=grid.grid, rays=batch_rays,
                                     gt=target_s, resolution=args.resolution,
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

    args = flags.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.scene)
    args.radius = args.radius
    return args


if __name__ == "__main__":
    _cmd_args = parse_args()
    run(_cmd_args)
