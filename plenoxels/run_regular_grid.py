import time
from typing import List

import torch
import torch.nn.functional as F

import config
from plenoxels.utils import EMA
from regular_grid import RegularGrid, ShDictRender
from tc_plenoptimize import init_datasets, init_sh_encoder


def init_renderers(cfg, dsets, num_atoms, resolution, fine_resolution, interpolate):
    reso = torch.tensor([resolution] * 3, dtype=torch.int32)
    grids = []
    # Scene-specific regular-grid modules
    for dset in dsets:
        grids.append(RegularGrid(
            resolution=reso, aabb=dset.scene_bbox, data_dim=num_atoms,
            near_far=dset.near_far, interpolate=interpolate))

    # scene-independent dictionary
    sh_encoder = init_sh_encoder(cfg, cfg.sh.degree)
    render = ShDictRender(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder, grids=grids,
        fine_reso=fine_resolution,
        init_sigma=0.1, init_rgb=0.01, white_bkgd=True,
        abs_light_thresh=1e-4, occupancy_thresh=1e-4)
    print(f"Initialized renderer {render}")
    return render


def initialize(cfg,
               data_dirs: List[str],
               num_atoms: int,
               coarse_reso: int,
               fine_reso: int,
               coarse_interpolate: bool):
    # Initialize datasets
    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for dd in data_dirs:
        cfg.data.datadir = dd
        tr_dset, tr_loader, ts_dset = init_datasets(cfg, "cuda")
        tr_dsets.append(tr_dset)
        tr_loaders.append(tr_loader)
        ts_dsets.append(ts_dset)
    # Initialize model
    renderer = init_renderers(cfg, tr_dsets, num_atoms, coarse_reso, fine_reso,
                              coarse_interpolate)
    renderer.cuda()

    # Initialize optimizer
    optims = []
    for i in range(len(renderer.grids)):
        optims.append(torch.optim.Adam(
            (renderer.atoms, renderer.grids[i].data), lr=1e-2
        ))

    return {
        "train_datasets": tr_dsets,
        "train_loaders": tr_loaders,
        "test_datasets": ts_dsets,
        "model": renderer,
        "optimizers": optims,
    }


def train_epoch(renderer, data_loaders, optims, max_steps, l1_loss_coef):
    batches_per_dset = 10
    max_tot_steps = max_steps * len(data_loaders)
    eta_min = 1e-5
    ema_weight = 0.5
    dev = "cuda"
    lr_scheds = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max_tot_steps // len(optims), eta_min=eta_min)
        for optim in optims
    ]

    iters = [iter(dl) for dl in data_loaders]
    tot_steps = 0
    time_s = time.time()
    losses = [{"mse": EMA(ema_weight), "l1": EMA(ema_weight)} for _ in range(len(data_loaders))]
    while len([it for it in iters if it is not None]) > 0:
        try:
            for dset_id, dset in enumerate(iters):
                if dset is None:
                    continue
                optim = optims[dset_id]
                lr_sched = lr_scheds[dset_id]
                for i in range(batches_per_dset):
                    if tot_steps >= max_tot_steps:
                        raise StopIteration("Maximum steps reached")
                    rays_o, rays_d, imgs = next(dset)
                    imgs = imgs.to(device=dev)
                    rays_o = rays_o.to(device=dev)
                    rays_d = rays_d.to(device=dev)

                    optim.zero_grad()
                    rgb_preds, alpha, depth = renderer(rays_o, rays_d, grid_id=dset_id)
                    rec_loss = F.mse_loss(rgb_preds, imgs)
                    l1_loss = l1_loss_coef * torch.abs(renderer.grids[dset_id].data).mean()
                    loss = rec_loss + l1_loss
                    loss.backward()
                    optim.step()
                    lr_sched.step()

                    losses[dset_id]["mse"].update(rec_loss.item())
                    losses[dset_id]["l1"].update(l1_loss.item())
                    tot_steps += 1

                    if tot_steps % 100 == 0:
                        print(f"{tot_steps:6d}  ", end="")
                        for did, loss in enumerate(losses):
                            print(f"D{did}: ", end="")
                            for lname, lval in loss.items():
                                print(f"{lname}={lval.value:.4f} ", end="")
                            print("  ", end="")
                        print()
                    if tot_steps % 1000 == 0:
                        print("Time for 1000 steps: %.2fs" % (time.time() - time_s))
                        time_s = time.time()
        except StopIteration:
            print(f"Dataset {dset_id} finished")
            iters[dset_id] = None


if __name__ == "__main__":
    cfg_ = config.get_cfg_defaults()
    cfg_.data.resolution = 256
    cfg_.data.max_tr_frames = None
    cfg_.data.max_ts_frames = 10
    cfg_.data.downsample = 1
    cfg_.optim.batch_size = 2000
    cfg_.sh.degree = 2
    cfg_.sh.sh_encoder = "plenoxels"
    data_dirs_ = [
        "/data/DATASETS/SyntheticNerf/lego",
        "/data/DATASETS/SyntheticNerf/ship",
        "/data/DATASETS/SyntheticNerf/chair",
        "/data/DATASETS/SyntheticNerf/drums/",
        "/data/DATASETS/SyntheticNerf/ficus/",
    ]
    num_atoms_ = 128
    coarse_reso_ = 64
    fine_reso_ = 7
    coarse_interpolate_ = True
    max_steps_ = 20_000
    l1_loss_coef_ = 0.1

    init_data = initialize(cfg_, data_dirs_, num_atoms=num_atoms_, coarse_reso=coarse_reso_,
                           fine_reso=fine_reso_, coarse_interpolate=coarse_interpolate_)
    train_epoch(init_data["model"], init_data["train_loaders"], init_data["optimizers"],
                max_steps_, l1_loss_coef_)
