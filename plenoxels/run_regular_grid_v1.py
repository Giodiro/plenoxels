import math
import os
import time

import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from ema import EMA
import config
from regular_grid import RegularGrid, ShDictRender
from synthetic_nerf_dataset import SyntheticNerfDataset
from tc_harmonics import plenoxel_sh_encoder


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def plot_ts(ts_dset, dset_id, renderer, log_dir, exp_name, iteration, batch_size=10_000):
    with torch.autograd.no_grad():
        for ts_el in ts_dset:
            if len(ts_el) == 3:
                rays_o, rays_d, rgb = ts_el
                img_h, img_w = ts_dset.img_h, ts_dset.img_w
            else:
                rays, rgb, _, _ = ts_el
                img_h, img_w = ts_dset.low_resolution, ts_dset.low_resolution
            preds = []
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to("cuda")
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to("cuda")
                preds.append(renderer(rays_o_b, rays_d_b, dset_id)[0].cpu())
            pred = torch.cat(preds, 0).view(img_h, img_w, 3)
            rgb = rgb.view(img_h, img_w, 3)
            mse = torch.mean((pred - rgb) ** 2)
            psnr = -10.0 * torch.log(mse) / math.log(10)
            break
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(pred)
    ax[1].imshow(rgb)
    ax[0].set_title(f"PSNR={psnr:.2f}")
    os.makedirs(f"{log_dir}/{exp_name}", exist_ok=True)
    fig.savefig(f"{log_dir}/{exp_name}/dset{dset_id}_iter{iteration}.png")
    fig.close()


def train_epoch(renderer, tr_loaders, ts_dsets, optim, max_steps, log_dir, exp_name):
    batches_per_dset = 10
    max_tot_steps = max_steps * len(tr_loaders)
    eta_min = 1e-5
    ema_weight = 0.3
    l1_loss_coef = 0.1
    dev = "cuda"

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps // batches_per_dset, eta_min=eta_min)

    iters = [iter(dl) for dl in tr_loaders]
    tot_steps = 0
    time_s = time.time()
    losses = [{"mse": EMA(ema_weight), "l1": EMA(ema_weight)} for _ in range(len(tr_loaders))]
    while len([it for it in iters if it is not None]) > 0:
        try:
            for dset_id, dset in enumerate(iters):
                if dset is None:
                    continue
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
            lr_sched.step()
        except StopIteration:
            print(f"Dataset {dset_id} finished")
            iters[dset_id] = None


def init_data(cfg):
    resolution = cfg.data.resolution
    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for data_dir in cfg.data.datadir:
        tr_dsets.append(SyntheticNerfDataset(
            data_dir, split='train', downsample=cfg.data.downsample, resolution=resolution,
            max_frames=cfg.data.max_tr_frames))
        tr_loaders.append(torch.utils.data.DataLoader(
            tr_dsets[-1], batch_size=cfg.optim.batch_size, shuffle=True, num_workers=3,
            prefetch_factor=4, pin_memory=True))
        ts_dsets.append(SyntheticNerfDataset(
            data_dir, split='test', downsample=cfg.data.downsample, resolution=resolution,
            max_frames=cfg.data.max_ts_frames))
    return tr_dsets, tr_loaders, ts_dsets


def init_model(cfg, tr_dsets, num_atoms, coarse_reso, fine_reso, interpolate):
    reso = torch.tensor([coarse_reso] * 3, dtype=torch.int32)
    grids = []
    for dset in tr_dsets:
        grids.append(RegularGrid(
            resolution=reso, aabb=dset.scene_bbox, data_dim=num_atoms,
            near_far=dset.near_far, interpolate=interpolate))
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    render = ShDictRender(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder, grids=grids,
        fine_reso=fine_reso,
        init_sigma=0.1, init_rgb=0.01, white_bkgd=True,
        abs_light_thresh=1e-4, occupancy_thresh=1e-4)
    return render


def init_optim(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


if __name__ == "__main__":
    cfg_ = config.parse_config()
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f'gpu is {gpu}')
    num_atoms_ = 128
    coarse_reso_ = 64
    fine_reso_ = 3
    lr_ = 1e-2

    max_steps_ = 2000
    log_dir_ = "./logs"
    exp_name_ = "e2"

    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(cfg_)
    model_ = init_model(cfg_, tr_dsets=tr_dsets_, num_atoms=num_atoms_, coarse_reso=coarse_reso_,
                        fine_reso=fine_reso_, interpolate=True)
    optim_ = init_optim(model_, lr_)

    train_epoch(renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                max_steps=max_steps_, log_dir=log_dir_, exp_name=exp_name_)
