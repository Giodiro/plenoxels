import os
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt

from plenoxels.synthetic_nerf_dataset import SyntheticNerfDataset

__all__ = (
    "get_freer_gpu",
    "init_data",
    "plot_ts",
    "plot_ts_imageio",
)


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def init_data(cfg):
    resolution = cfg.data.resolution
    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for data_dir in cfg.data.datadirs:
        tr_dsets.append(SyntheticNerfDataset(
            data_dir, split='train', downsample=cfg.data.downsample, resolution=resolution,
            max_frames=cfg.data.max_tr_frames))
        tr_loaders.append(torch.utils.data.DataLoader(
            tr_dsets[-1], batch_size=cfg.optim.batch_size, shuffle=True, num_workers=3,
            prefetch_factor=4, pin_memory=True))
        ts_dsets.append(SyntheticNerfDataset(
            data_dir, split='test', downsample=1, resolution=800,
            max_frames=cfg.data.max_ts_frames))
    return tr_dsets, tr_loaders, ts_dsets


def render_ts_img(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  model,
                  batch_size: int,
                  img_h: int, img_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rays_o, rays_d, rgb = data
    preds = []
    for b in range(math.ceil(rays_o.shape[0] / batch_size)):
        rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to("cuda")
        rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to("cuda")
        preds.append(model(rays_o_b, rays_d_b).cpu())
    pred = torch.cat(preds, 0).view(img_h, img_w, 3)
    rgb = rgb.view(img_h, img_w, 3)
    return pred, rgb


def plot_ts(ts_dset, dset_id, renderer, log_dir, iteration, batch_size=10_000):
    psnr_list = []
    with torch.autograd.no_grad():
        for ts_el in ts_dset:
            pred, rgb = render_ts_img(
                ts_el, model=lambda ro, rd: renderer(ro, rd, dset_id)[0],
                batch_size=batch_size, img_h=ts_dset.img_h, img_w=ts_dset.img_w)
            mse = torch.mean((pred - rgb) ** 2)
            psnr = -10.0 * torch.log(mse) / math.log(10)
            psnr_list.append(psnr)
            break
    psnr = np.mean(psnr_list)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(pred)
    ax[1].imshow(rgb)
    ax[0].set_title(f"PSNR={psnr:.2f}")
    fig.savefig(os.path.join(log_dir, f"dset{dset_id}_iter{iteration}.png"))
    plt.close(fig)


def plot_ts_imageio(ts_dset, dset_id, renderer, log_dir, iteration: Union[int, str], batch_size=10_000, image_id=0, verbose=True) -> float:
    import imageio
    with torch.autograd.no_grad():
        ts_el = ts_dset[image_id]
        pred, rgb = render_ts_img(
            ts_el, model=lambda ro, rd: renderer(ro, rd, dset_id)[0],
            batch_size=batch_size, img_h=ts_dset.img_h, img_w=ts_dset.img_w)
        mse = torch.mean((pred - rgb) ** 2)
        psnr = -10.0 * torch.log(mse) / math.log(10)
    if verbose:
        print(f"D{dset_id} Test PSNR={psnr:.2f}")
    vis = torch.cat((torch.clamp(pred, 0, 1), rgb), dim=1)
    vis = (vis * 255).numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(log_dir, f"dset{dset_id}_iter{iteration}_img{image_id}.png"), vis)
    return psnr.item()
