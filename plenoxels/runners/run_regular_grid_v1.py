import os
import time

import torch
import torch.utils.data
import torch.nn.functional as F

from plenoxels import multiscene_config
from plenoxels.ema import EMA
from plenoxels.regular_grid import RegularGrid, ShDictRender
from plenoxels.tc_harmonics import plenoxel_sh_encoder

from plenoxels.runners.utils import *


def train_epoch(renderer, tr_loaders, ts_dsets, optim, max_steps, log_dir, exp_name):
    batches_per_dset = 10
    max_tot_steps = max_steps * len(tr_loaders)
    eta_min = 1e-5
    ema_weight = 0.3
    l1_loss_coef = 0.1
    dev = "cuda"

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps // batches_per_dset, eta_min=eta_min)

    tr_iterators = [iter(dl) for dl in tr_loaders]
    tot_steps = 0
    time_s = time.time()
    losses = [{"mse": EMA(ema_weight), "l1": EMA(ema_weight)} for _ in range(len(tr_loaders))]
    renderer.cuda()
    while len([it for it in tr_iterators if it is not None]) > 0:
        for dset_id, dset in enumerate(tr_iterators):
            if dset is None:
                continue
            try:
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
                        for ts_dset_id, ts_dset in enumerate(ts_dsets):
                            plot_ts(ts_dset, ts_dset_id, renderer, log_dir, exp_name, tot_steps, batch_size=3000)
                        print("Plot test images in %.2fs" % (time.time() - time_s))
                        time_s = time.time()
            except StopIteration:
                print(f"Dataset {dset_id} finished")
                tr_iterators[dset_id] = None
        lr_sched.step()


def init_model(cfg, tr_dsets, interpolate):
    reso = torch.tensor([cfg.model.coarse_reso] * 3, dtype=torch.int32)
    grids = []
    for dset in tr_dsets:
        grids.append(RegularGrid(
            resolution=reso, aabb=dset.scene_bbox, data_dim=cfg.model.num_atoms,
            near_far=dset.near_far, interpolate=interpolate))
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    render = ShDictRender(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder, grids=grids,
        fine_reso=cfg.model.fine_reso,
        init_sigma=0.1, init_rgb=0.01, white_bkgd=True,
        abs_light_thresh=1e-4, occupancy_thresh=1e-4)
    return render


def init_optim(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)


if __name__ == "__main__":
    cfg_ = multiscene_config.parse_config()
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f'gpu is {gpu}')

    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(cfg_)
    model_ = init_model(cfg_, tr_dsets=tr_dsets_, interpolate=True)
    optim_ = init_optim(cfg_, model_)

    train_epoch(renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                max_steps=cfg_.optim.max_steps, log_dir=cfg_.logdir, exp_name=cfg_.expname)
