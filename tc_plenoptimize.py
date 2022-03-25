import argparse
import functools
import math
import os
import time
from argparse import ArgumentParser
from contextlib import ExitStack
from typing import Any, Union, Optional

import imageio
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import tc_plenoxel
from synthetic_nerf_dataset import SyntheticNerfDataset


def parse_config():
    # Build experiment configuration
    parser = argparse.ArgumentParser("Train + evaluate kernel model")
    parser.add_argument("--config", default=None)
    parser.add_argument('--config-updates', default=[], nargs='*')
    args = parser.parse_args()
    cfg = config.get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.config_updates)
    cfg.freeze()
    print(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    return cfg


def init_plenoxel_lrs(cfg, h_degree, dev):
    resolution = cfg.data.resolution
    sh_dim = (h_degree + 1) ** 2
    # Initialize learning-rate
    lr_rgb, lr_sigma = cfg.optim.lr_rgb, cfg.optim.lr_sigma
    if lr_rgb is None or lr_sigma is None:
        lr_rgb = 150 * (resolution ** 1.75)
        lr_sigma = 51.5 * (resolution ** 2.37)
    lrs = [lr_rgb] * (sh_dim * 3) + [lr_sigma]
    print("Learning rates: ", lrs)
    return torch.tensor(lrs, dtype=torch.float32, device=dev)


def init_datasets(cfg, dev):
    resolution = cfg.data.resolution
    tr_dset = SyntheticNerfDataset(cfg.data.datadir, split='train', downsample=cfg.data.downsample,
                                   resolution=resolution, max_frames=cfg.data.max_tr_frames)
    tr_loader = DataLoader(tr_dset, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=2,
                           prefetch_factor=3,
                           pin_memory=dev.startswith("cuda"))
    ts_dset = SyntheticNerfDataset(cfg.data.datadir, split='test', downsample=cfg.data.downsample,
                                   resolution=resolution, max_frames=cfg.data.max_ts_frames)
    return tr_dset, tr_loader, ts_dset


def init_profiler(cfg, dev) -> torch.profiler.profile:
    profiling_handler = functools.partial(trace_handler, exp_name=cfg.expname,
                                          dev="cpu" if dev == "cpu" else "cuda")
    p = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=15),
        on_trace_ready=profiling_handler,
        with_stack=False, profile_memory=True, record_shapes=True)
    return p


def init_sh_encoder(cfg, h_degree):
    if cfg.sh.sh_encoder == "tcnn":
        return tcnn.Encoding(3, {
            "otype": "SphericalHarmonics",
            "degree": h_degree + 1,
        })
    elif cfg.sh.sh_encoder == "plenoxels":
        return tc_plenoxel.plenoxel_sh_encoder(h_degree)
    else:
        raise ValueError(cfg.sh.sh_encoder)


def train_hierarchical_grid(cfg):
    h_degree = cfg.sh.degree
    resolution = cfg.data.resolution
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    tr_dset, tr_loader, ts_dset = init_datasets(cfg, dev)
    lrs = init_plenoxel_lrs(cfg, h_degree, dev)
    sh_enc = init_sh_encoder(cfg, h_degree)

    n_levels = 14  # we need 28 channels
    base_res = 16
    per_level_scale = resolution / (n_levels * base_res)
    hg = tcnn.Encoding(3, {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": n_levels,
        "n_features_per_level": 2,  # Total of 10*3 = 30 features
        "log2_hashmap_size": cfg.hash_grid.log2_hashmap_size,
        "base_resolution": base_res,  # Resolution of the coarsest level
        "per_level_scale": per_level_scale,  # How much the resolution increases at each level
        "interpolation": "Linear",
    })
    for param in hg.parameters():  # Same init as in paper.
        torch.nn.init.uniform_(param, -1e-4, 1e-4)

    # Initialize model
    model = tc_plenoxel.HashGrid(
        resolution=torch.tensor([resolution, resolution, resolution], dtype=torch.int32),
        aabb=tr_dset.scene_bbox,
        deg=h_degree,
        ini_sigma=cfg.grid.ini_sigma,
        ini_rgb=cfg.grid.ini_rgb,
        sh_encoder=sh_enc,
        white_bkgd=tr_dset.white_bg,
        uniform_rays=0.5,
        count_intersections=cfg.irreg_grid.count_intersections,
        harmonic_degree=h_degree,
        hg_encoder=hg,
    ).to(dev)
    #optim = torch.optim.Adam(params=model.parameters(), lr=0.1)

    # Main iteration starts here
    for epoch in range(cfg.optim.num_epochs):
        psnrs, mses = [], []
        with ExitStack() as stack:
            p = None
            if cfg.optim.profile:
                p = init_profiler(cfg, dev)
                stack.enter_context(p)
            model = model.train()
            for i, batch in tqdm(enumerate(tr_loader), desc=f"Epoch {epoch}"):
                # optim.zero_grad()
                rays, imgs = batch
                rays = rays.to(device=dev)
                rays_o = rays[:, 0].contiguous()
                rays_d = rays[:, 1].contiguous()
                imgs = imgs.to(device=dev)
                preds, _, _ = model(rays_o=rays_o, rays_d=rays_d)
                loss = F.mse_loss(preds, imgs)
                with torch.autograd.no_grad():
                    # Using standard optimizer
                    # total_loss.backward()
                    # optim.step()
                    # Our own optimization procedure
                    grad = torch.autograd.grad(loss, model.grid_data)[0]  # [batch, n_ch]
                    grad.mul_(lrs.unsqueeze(0))
                    model.grid_data.sub_(grad)

                # Reporting
                loss = loss.detach().item()
                psnrs.append(-10.0 * math.log(loss) / math.log(10.0))
                mses.append(loss)
                if i % cfg.optim.progress_refresh_rate == 0:
                    print(f"Epoch {epoch} - iteration {i}: "
                          f"MSE {np.mean(mses):.4f} PSNR {np.mean(psnrs):.4f}")
                    psnrs, mses = [], []

                if (i + 1) % cfg.optim.eval_refresh_rate == 0:
                    ts_psnr = run_test_step(
                        ts_dset, model, render_every=cfg.optim.render_refresh_rate,
                        log_dir=cfg.logdir, epoch=epoch, batch_size=cfg.optim.batch_size,
                        device=dev)
                    print(f"Epoch {epoch} - iteration {i}: Test PSNR: {ts_psnr:.4f}")

                # Profiling
                if p is not None:
                    p.step()


def train_irregular_grid(cfg):
    h_degree = cfg.sh.degree
    resolution = cfg.data.resolution
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    tr_dset, tr_loader, ts_dset = init_datasets(cfg, dev)
    lrs = init_plenoxel_lrs(cfg, h_degree, dev)
    sh_enc = init_sh_encoder(cfg, h_degree)

    occupancy_penalty = cfg.optim.occupancy_penalty / (len(tr_dset) // cfg.optim.batch_size)

    # Initialize model
    model = tc_plenoxel.IrregularGrid(
        resolution=torch.tensor([resolution, resolution, resolution], dtype=torch.int32),
        aabb=tr_dset.scene_bbox,
        deg=h_degree,
        ini_sigma=cfg.grid.ini_sigma,
        ini_rgb=cfg.grid.ini_rgb,
        sh_encoder=sh_enc,
        white_bkgd=tr_dset.white_bg,
        uniform_rays=0.5,
        prune_threshold=cfg.irreg_grid.prune_threshold,
        count_intersections=cfg.irreg_grid.count_intersections,
        near_far=tuple(tr_dset.near_far),
    ).to(dev)
    #optim = torch.optim.Adam(params=model.parameters(), lr=0.1)

    # Main iteration starts here
    for epoch in range(cfg.optim.num_epochs):
        psnrs, mses = [], []
        with ExitStack() as stack:
            p = None
            if cfg.optim.profile:
                p = init_profiler(cfg, dev)
                stack.enter_context(p)
            model = model.train()
            for i, batch in tqdm(enumerate(tr_loader), desc=f"Epoch {epoch}"):
                # optim.zero_grad()
                rays, imgs = batch
                rays = rays.to(device=dev)
                rays_o = rays[:, 0].contiguous()
                rays_d = rays[:, 1].contiguous()
                imgs = imgs.to(device=dev)
                preds, _, _ = model(rays_o=rays_o, rays_d=rays_d)
                loss = F.mse_loss(preds, imgs)
                if occupancy_penalty > 0:
                    total_loss = loss + occupancy_penalty * model.approx_density_tv_reg()
                else:
                    total_loss = loss
                with torch.autograd.no_grad():
                    # Using standard optimizer
                    # total_loss.backward()
                    # optim.step()
                    # Our own optimization procedure
                    grad = torch.autograd.grad(total_loss, model.grid_data)[0]  # [batch, n_ch]
                    grad.mul_(lrs.unsqueeze(0))
                    model.grid_data.sub_(grad)

                # Reporting
                loss = loss.detach().item()
                psnrs.append(-10.0 * math.log(loss) / math.log(10.0))
                mses.append(loss)
                if i % cfg.optim.progress_refresh_rate == 0:
                    print(f"Epoch {epoch} - iteration {i}: "
                          f"MSE {np.mean(mses):.4f} PSNR {np.mean(psnrs):.4f}")
                    psnrs, mses = [], []

                if (i + 1) % cfg.optim.eval_refresh_rate == 0:
                    ts_psnr = run_test_step(
                        ts_dset, model, render_every=cfg.optim.render_refresh_rate,
                        log_dir=cfg.logdir, epoch=epoch, batch_size=cfg.optim.batch_size,
                        device=dev)
                    print(f"Epoch {epoch} - iteration {i}: Test PSNR: {ts_psnr:.4f}")

                # Profiling
                if p is not None:
                    p.step()


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


def run_test_step(test_dset: SyntheticNerfDataset,
                  model,
                  render_every: int,
                  log_dir: str,
                  epoch: int,
                  batch_size: int,
                  device: Union[torch.device, str],
                  **model_kwargs) -> float:
    model.eval()
    with torch.autograd.no_grad():
        total_psnr = 0.0
        for i, test_el in tqdm(enumerate(test_dset), desc="Evaluating on test data"):
            # These are rays/rgb for a full single image
            rays = test_el['rays']
            rgb = test_el['rgb'].reshape(test_dset.img_h, test_dset.img_w, 3)
            # We need to do some manual batching
            rgb_map, depth = [], []
            for b in range(math.ceil(rays.shape[0] / batch_size)):
                rays_o = rays[b * batch_size: (b + 1) * batch_size, 0].contiguous().to(
                    device=device)
                rays_d = rays[b * batch_size: (b + 1) * batch_size, 1].contiguous().to(
                    device=device)
                rgb_map_b, _, depth_b = model(rays_o=rays_o, rays_d=rays_d, **model_kwargs)
                rgb_map.append(rgb_map_b.cpu())
                depth.append(depth_b.cpu())
            rgb_map = torch.stack(rgb_map, 0).reshape(test_dset.img_h, test_dset.img_w, 3)
            depth = torch.stack(depth).reshape(test_dset.img_h, test_dset.img_w)

            # Compute loss metrics
            mse = torch.mean((rgb_map - rgb) ** 2)
            psnr = -10.0 * torch.log(mse) / math.log(10)
            total_psnr += psnr

            # Render and save result to file
            if (i + 1) % render_every == 0:
                depth = depth.unsqueeze(-1).repeat(1, 1, 3)
                out_img = torch.cat((rgb, rgb_map, depth), dim=1)
                out_img = (out_img * 255).to(dtype=torch.uint8).numpy()
                imageio.imwrite(f"{log_dir}/{epoch:04}_{i:04}.png", out_img)
    return total_psnr / i


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
    if False:
        sh_enc = tcnn.Encoding(3, {
            "otype": "SphericalHarmonics",
            "degree": args.harmonic_degree + 1,
        })
    else:
        sh_enc = tc_plenoxel.plenoxel_sh_encoder(args.harmonic_degree)

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
                train_batch(params=params, params_type=args.params_type,
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
    # _cmd_args = parse_args()
    # run(_cmd_args)
    _cfg = parse_config()
    if _cfg.model_type == "irregular_grid":
        train_irregular_grid(_cfg)
    elif _cfg.model_type == "hash_grid":
        train_hierarchical_grid(_cfg)
    else:
        raise ValueError(f"Model type {_cfg.model_type} unknown.")

