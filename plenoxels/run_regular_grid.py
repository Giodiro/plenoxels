import os
import time
import math
from typing import List
import imageio

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import config
from griddict import ShDictRender
from synthetic_nerf_dataset import SyntheticNerfDataset
from torch.utils.data import DataLoader
import tc_plenoxel
import numpy as np

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

class EMA():
    def __init__(self, weighting=0.9):
        self.weighting = weighting
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.weighting * val + (1 - self.weighting) * self.val

    @property
    def value(self):
        return self.val


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
    vis = torch.cat((pred, rgb), dim=1)
    vis = np.asarray((vis * 255)).astype(np.uint8)
    os.makedirs(f"{log_dir}/{exp_name}", exist_ok=True)
    imageio.imwrite(f"{log_dir}/{exp_name}/dset{dset_id}_iter{iteration}.png", vis)
    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(pred)
    # ax[1].imshow(rgb)
    # ax[0].set_title(f"PSNR={psnr:.2f}")
    # fig.savefig(f"{log_dir}/{exp_name}/dset{dset_id}_iter{iteration}.png")
    # return fig


def init_datasets(cfg, dev):
    resolution = cfg.data.resolution
    tr_dset = SyntheticNerfDataset(cfg.data.datadir, split='train', downsample=cfg.data.downsample,
                                   resolution=resolution, max_frames=cfg.data.max_tr_frames)
    tr_loader = DataLoader(tr_dset, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=3,
                           prefetch_factor=10,
                           pin_memory=dev.startswith("cuda"))
    ts_dset = SyntheticNerfDataset(cfg.data.datadir, split='test', downsample=cfg.data.downsample,
                                   resolution=resolution, max_frames=cfg.data.max_ts_frames)
    return tr_dset, tr_loader, ts_dset


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

def init_renderers(cfg, dsets, num_atoms, resolution, fine_resolution, efficient_dict):
    sh_encoder = init_sh_encoder(cfg, cfg.sh.degree)
    render = ShDictRender(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder, 
        radius=1.3, num_atoms=num_atoms, num_scenes=len(dsets),
        fine_reso=fine_resolution, coarse_reso=resolution, 
        init_sigma=0.1, init_rgb=0.01, efficient_dict=efficient_dict)
    print(f"Initialized renderer {render}")
    return render


def initialize(cfg,
               data_dirs: List[str],
               num_atoms: int,
               coarse_reso: int,
               fine_reso: int,
               efficient_dict: bool):
    # Initialize datasets
    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for dd in data_dirs:
        cfg.data.datadir = dd
        tr_dset, tr_loader, ts_dset = init_datasets(cfg, "cuda")
        tr_dsets.append(tr_dset)
        tr_loaders.append(tr_loader)
        ts_dsets.append(ts_dset)
    # Initialize model
    renderer = init_renderers(cfg, tr_dsets, num_atoms, coarse_reso, fine_reso, efficient_dict=efficient_dict)
    renderer.cuda()

    # Initialize optimizer
    optim = torch.optim.Adam(renderer.parameters(), lr=1e-2)

    return {
        "train_datasets": tr_dsets,
        "train_loaders": tr_loaders,
        "test_datasets": ts_dsets,
        "model": renderer,
        "optimizer": optim,
    }


def train_epoch(renderer, data_loaders, ts_dsets, optim, max_steps, l1_loss_coef, exp_name, batch_size):
    batches_per_dset = 10
    max_tot_steps = max_steps * len(data_loaders)
    eta_min = 1e-5
    ema_weight = 0.5
    dev = "cuda"
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max_tot_steps // len(data_loaders), eta_min=eta_min)

    iters = [iter(dl) for dl in data_loaders]
    tot_steps = 0
    time_s = time.time()
    losses = [{"mse": EMA(ema_weight), "l1": EMA(ema_weight)} for _ in range(len(data_loaders))]
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
                        for j in range(len(ts_dsets)):
                            plot_ts(ts_dsets[j], j, renderer, "logs", exp_name, tot_steps, batch_size=batch_size)
        except StopIteration:
            print(f"Dataset {dset_id} finished")
            iters[dset_id] = None


if __name__ == "__main__":
    cfg_ = config.get_cfg_defaults()
    cfg_.data.resolution = 256
    cfg_.data.max_tr_frames = None
    cfg_.data.max_ts_frames = 10
    cfg_.data.downsample = 1
    cfg_.optim.batch_size = 1000
    cfg_.sh.degree = 0
    cfg_.sh.sh_encoder = "plenoxels"
    data_dirs_ = [
        "/data/datasets/nerf/data/nerf_synthetic/lego",
        "/data/datasets/nerf/data/nerf_synthetic/chair",
        "/data/datasets/nerf/data/nerf_synthetic/drums/",
        "/data/datasets/nerf/data/nerf_synthetic/ficus/",
    ]
    efficient_dict_ = True
    num_atoms_ = 32
    coarse_reso_ = 32
    fine_reso_ = 4
    max_steps_ = 2_000
    l1_loss_coef_ = 0.1
    exp_name_ = "e1"

    init_data = initialize(cfg_, data_dirs_, num_atoms=num_atoms_, coarse_reso=coarse_reso_,
                           fine_reso=fine_reso_, efficient_dict=efficient_dict_)
    train_epoch(init_data["model"], init_data["train_loaders"], optim=init_data["optimizer"], ts_dsets=init_data['test_datasets'],
                max_steps=max_steps_, l1_loss_coef=l1_loss_coef_, exp_name=exp_name_, batch_size=cfg_.optim.batch_size)
