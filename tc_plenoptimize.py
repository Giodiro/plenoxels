import argparse
import functools
import math
import os
from contextlib import ExitStack
from typing import Union

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
        lr_rgb = 150 * (resolution ** 1.75) * (cfg.optim.batch_size / 4000)
        lr_sigma = 51.5 * (resolution ** 2.37) * (cfg.optim.batch_size / 4000)
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
        torch.nn.init.constant_(param, 0.001)

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
    optim = torch.optim.Adam(params=model.parameters(), lr=cfg.optim.adam.lr, eps=1e-15, betas=(0.9, 0.99))

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
                optim.zero_grad()
                rays, imgs = batch
                rays = rays.to(device=dev)
                rays_o = rays[:, 0].contiguous()
                rays_d = rays[:, 1].contiguous()
                imgs = imgs.to(device=dev)
                preds, _, _ = model(rays_o=rays_o, rays_d=rays_d)
                loss = F.mse_loss(preds, imgs)
                with torch.autograd.no_grad():
                    # Using standard optimizer
                    loss.backward()
                    optim.step()
                    # Our own optimization procedure
                    # grad = torch.autograd.grad(loss, model.grid_data)[0]  # [batch, n_ch]
                    # grad.mul_(lrs.unsqueeze(0))
                    # model.grid_data.sub_(grad)

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
                        log_dir=cfg.logdir, iteration=epoch * len(tr_loader) + i + 1,
                        batch_size=cfg.optim.batch_size, device=dev, exp_name=cfg.expname)
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
    # optim = torch.optim.Adam(params=model.parameters(), lr=0.1)

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
                        log_dir=cfg.logdir, iteration=epoch * len(tr_loader) + i + 1,
                        batch_size=cfg.optim.batch_size, device=dev, exp_name=cfg.expname)
                    print(f"Epoch {epoch} - iteration {i}: Test PSNR: {ts_psnr:.4f}")

                # Profiling
                if p is not None:
                    p.step()


def train_grid(cfg):
    h_degree = cfg.sh.degree
    resolution = cfg.data.resolution
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    tr_dset, tr_loader, ts_dset = init_datasets(cfg, dev)
    lrs = init_plenoxel_lrs(cfg, h_degree, dev)
    sh_enc = init_sh_encoder(cfg, h_degree)

    occupancy_penalty = cfg.optim.occupancy_penalty / (len(tr_dset) // cfg.optim.batch_size)

    # Initialize model
    model = tc_plenoxel.RegularGrid(
        resolution=torch.tensor([resolution, resolution, resolution], dtype=torch.int32),
        aabb=tr_dset.scene_bbox,
        deg=h_degree,
        ini_sigma=cfg.grid.ini_sigma,
        ini_rgb=cfg.grid.ini_rgb,
        sh_encoder=sh_enc,
        white_bkgd=tr_dset.white_bg,
        uniform_rays=0.5,
        count_intersections=cfg.irreg_grid.count_intersections,
        near_far=tuple(tr_dset.near_far),
        abs_light_thresh=cfg.grid.abs_light_thresh,
        occupancy_thresh=cfg.grid.occupancy_thresh
    ).to(dev)
    def init_optim():
        return torch.optim.SGD(params=[
            {'params': (model.rgb_data, ), 'lr': lrs[0]},
            {'params': (model.sigma_data, ), 'lr': lrs[-1]}
        ])
    optim = init_optim()

    # Initialize list of resolutions
    reso_multiplier = 1.4

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
                g_iter = epoch * len(tr_loader) + i + 1
                optim.zero_grad()
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
                    total_loss.backward()
                    optim.step()

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
                        log_dir=cfg.logdir, iteration=g_iter,
                        batch_size=cfg.optim.batch_size * 4, device=dev, exp_name=cfg.expname)
                    print(f"Epoch {epoch} - iteration {i}: Test PSNR: {ts_psnr:.4f}")

                if g_iter in cfg.grid.update_occ_iters:
                    model.update_occupancy()
                if g_iter in cfg.grid.shrink_iters:
                    model.shrink()
                if g_iter in cfg.grid.upsample_iters:
                    model.upscale(new_resolution=(model.resolution * reso_multiplier).long())
                if model.params_changed:
                    optim.param_groups[0]['params'] = (model.rgb_data, )
                    optim.param_groups[1]['params'] = (model.sigma_data, )
                    model.params_changed = False

                # Profiling
                if p is not None:
                    p.step()


def run_test_step(test_dset: SyntheticNerfDataset,
                  model,
                  render_every: int,
                  log_dir: str,
                  iteration: int,
                  batch_size: int,
                  device: Union[torch.device, str],
                  exp_name: str,
                  **model_kwargs) -> float:
    model.eval()
    with torch.autograd.no_grad():
        total_psnr = 0.0
        for i, test_el in tqdm(enumerate(test_dset), desc="Evaluating on test data"):
            # These are rays/rgb for a full single image
            rays, rgb = test_el
            rgb = rgb.reshape(test_dset.img_h, test_dset.img_w, 3)
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
                out_dir = os.path.join(log_dir, exp_name)
                os.makedirs(out_dir, exist_ok=True)
                imageio.imwrite(os.path.join(out_dir, f"{iteration:05}_test_img_{i:04}.png"), out_img)
    return total_psnr / i


def trace_handler(p, exp_name, dev):
    print(p.key_averages().table(sort_by=f"self_{dev}_time_total", row_limit=10))
    print(p.key_averages().table(sort_by=f"self_{dev}_memory_usage", row_limit=10))
    # p.export_chrome_trace(f"./trace_{exp_name}_{p.step_num}.json")
    # p.export_stacks(f"./profiler_stacks_{exp_name}_{p.step_num}.txt", f"self_{dev}_time_total")
    torch.profiler.tensorboard_trace_handler(f"tb_logs/{exp_name}")(p)


if __name__ == "__main__":
    # _cmd_args = parse_args()
    # run(_cmd_args)
    _cfg = parse_config()
    if _cfg.model_type == "regular_grid":
        train_grid(_cfg)
    elif _cfg.model_type == "irregular_grid":
        train_irregular_grid(_cfg)
    elif _cfg.model_type == "hash_grid":
        train_hierarchical_grid(_cfg)
    else:
        raise ValueError(f"Model type {_cfg.model_type} unknown.")
