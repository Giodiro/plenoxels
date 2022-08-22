import os
import math
from typing import Tuple, Union, Optional

import imageio
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm

from plenoxels.nerf_rendering import sigma2alpha, shrgb2rgb
from plenoxels.synthetic_nerf_dataset import SyntheticNerfDataset, get_rays

__all__ = (
    "get_freer_gpu",
    "init_data",
    "init_data_single_dset",
    "plot_ts",
    "test_model",
    "render_patches",
    "user_ask_options",
)

from plenoxels.tc_harmonics import plenoxel_sh_encoder


def get_freer_gpu():
    try:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)
    except:  # On some Giacomo GPUs this fails due to newer drivers. But I have 1 GPU anyways
        return 0


def init_data_single_dset(cfg):
    resolution = cfg.data.resolution
    downsample = cfg.data.downsample
    if cfg.data.resolution is None:
        # None is code for automatic resolution adjustment to match the model resolution
        # also adjusting the downsampling.
        resolution = cfg.model.resolution
        downsample = max(1.0, 800 / (resolution * 2))
    if downsample is None:
        downsample = 1.0
    print(f"Loading dataset with resolution={resolution}, downsample={downsample}")
    train = SyntheticNerfDataset(cfg.data.datadir, split='train', downsample=downsample,
                                 resolution=resolution, max_frames=cfg.data.max_tr_frames)
    train_load = torch.utils.data.DataLoader(train, batch_size=cfg.optim.batch_size, shuffle=True,
                                             num_workers=4, prefetch_factor=2, pin_memory=True)
    test = SyntheticNerfDataset(cfg.data.datadir, split='test', downsample=1, resolution=800,
                                max_frames=cfg.data.max_ts_frames)
    return train, train_load, test


def init_data(cfg):
    resolution = [cfg.data.resolution]
    downsample = [cfg.data.downsample]
    if cfg.data.resolution is None:
        # None is code for automatic resolution adjustment to match the fine dictionaries
        # also adjust the downsampling so that the number of rays isn't so huge for small dicts
        resolution = []
        downsample = []
        for fine in cfg.model.fine_reso:
            resolution.append(cfg.model.coarse_reso * fine)
            downsample.append(800.0 / (cfg.model.coarse_reso * fine * 2.0))
    # Training datasets are lists of lists, where each inner list is different resolutions for the same scene
    # Test datasets are a single list over the different scenes, all at full resolution
    print(f"Loading datasets with reso={resolution}, downsample={downsample}")
    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for data_dir in cfg.data.datadirs:
        train, train_load = [], []
        for reso, down in zip(resolution, downsample):
            train.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=down, resolution=reso,
                max_frames=cfg.data.max_tr_frames))
            train_load.append(torch.utils.data.DataLoader(
                train[-1], batch_size=cfg.optim.batch_size, shuffle=True, num_workers=3,
                prefetch_factor=4, pin_memory=True))
        tr_dsets.append(train)
        tr_loaders.append(train_load)
        ts_dsets.append(SyntheticNerfDataset(
            data_dir, split='test', downsample=1, resolution=800,
            max_frames=cfg.data.max_ts_frames))
    return tr_dsets, tr_loaders, ts_dsets


def save_image(img_or_fig, log_dir, img_name, iteration, summary_writer):
    if log_dir is not None:
        out_path = os.path.join(log_dir, f"{img_name}_iter{iteration}.png")
        if isinstance(img_or_fig, plt.Figure):
            img_or_fig.savefig(out_path)
            plt.close(img_or_fig)
        else:
            # imageio wants channels last
            imageio.imwrite(out_path, img_or_fig)
    if summary_writer is not None:
        if isinstance(img_or_fig, plt.Figure):
            summary_writer.add_figure(img_name, img_or_fig, global_step=iteration)
            plt.close(img_or_fig)
        else:
            summary_writer.add_image(img_name, img_or_fig, global_step=iteration, dataformats="HWC")


def render_ts_img(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  model,
                  batch_size: int,
                  img_h: int, img_w: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    rays_o, rays_d, rgb = data
    preds, depths = [], []
    for b in range(math.ceil(rays_o.shape[0] / batch_size)):
        rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to("cuda")
        rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to("cuda")
        output = model(rays_o_b, rays_d_b)
        if isinstance(output, torch.Tensor):
            preds.append(output.cpu())
        elif isinstance(output, tuple):
            preds.append(output[0].cpu())
            if len(output) > 1 and output[1] is not None:
                depths.append(output[1].cpu())
    pred = torch.cat(preds, 0).view(img_h, img_w, 3)
    depth = None
    if len(depths) > 0:
        depth = torch.cat(depths, 0).view(img_h, img_w)
    rgb = rgb.view(img_h, img_w, 3)
    return pred, rgb, depth


def plot_ts(ts_dset, dset_id, renderer, log_dir, iteration, batch_size=10_000, image_id=0,
            verbose=True, summary_writer=None, render_fn=None, plot_type="imageio"):
    if render_fn is None:
        def render_fn(ro, rd):
            return renderer(ro, rd, dset_id)[0]
    with torch.autograd.no_grad():
        ts_el = ts_dset[image_id]
        pred, rgb, opt_depth = render_ts_img(
            ts_el, model=render_fn, batch_size=batch_size, img_h=ts_dset.img_h, img_w=ts_dset.img_w)
        mse = torch.mean((pred - rgb) ** 2)
        psnr = -10.0 * torch.log(mse) / math.log(10)
    if verbose:
        print(f"D{dset_id} Test PSNR={psnr:.2f}")
    if log_dir is not None:
        if plot_type == "imageio":
            fig = torch.cat((torch.clamp(pred, 0, 1), rgb), dim=1)
            fig = (fig * 255).numpy().astype(np.uint8)
        elif plot_type == "matplotlib":
            fig, ax = plt.subplots(ncols=2 if opt_depth is None else 3, figsize=(15, 8), dpi=90)
            ax[0].imshow(torch.clamp(pred, 0, 1))
            ax[0].axis('off')
            err = ((pred - rgb)**2).sum(-1)
            err = (err - err.min()) / (err.max() - err.min())  # normalize in 0,1
            ax[1].imshow(err, cmap="plasma")
            ax[1].axis('off')
            if opt_depth is not None:
                ax[2].imshow((opt_depth - opt_depth.min()) / (opt_depth.max() - opt_depth.min()), cmap="plasma")
                ax[2].axis('off')
            ax[0].set_title(f"PSNR={psnr:.2f}")
        else:
            raise ValueError(f"plot-type {plot_type} invalid. "
                             f"Accepted values are 'imageio' and 'matplotlib'.")
        save_image(fig, log_dir, f"dset-{dset_id}-ts-{image_id}", iteration, summary_writer)
    return psnr.item()


def test_model(renderer, ts_dset, log_dir, batch_size, render_fn, plot_type="imageio", num_test_imgs=None):
    renderer.cuda()
    renderer.eval()
    if num_test_imgs is None:
        num_test_imgs = len(ts_dset)
    psnrs = []
    plot_every = max(1, num_test_imgs // 5)
    for image_id in tqdm(range(num_test_imgs), desc=f"test-dataset evaluation"):
        c_log_dir = log_dir if image_id % plot_every == plot_every - 1 else None
        psnr = plot_ts(ts_dset, 0, renderer, c_log_dir, iteration="test",
                       batch_size=batch_size, image_id=image_id, verbose=False,
                       plot_type=plot_type, render_fn=render_fn)
        psnrs.append(psnr)
    print(f"Average PSNR (over {num_test_imgs} poses): {np.mean(psnrs):.2f}")


def render_patches(renderer, patch_level, log_dir, iteration, summary_writer=None):
    def get_intersections(rays_o: torch.Tensor, rays_d: torch.Tensor, radius: float, step_size: float, n_samples: int):
        dev, dt = rays_o.device, rays_o.dtype
        offsets_pos = (radius - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (-radius - rays_o) / rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        steps = torch.arange(n_samples, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
        steps = steps.repeat(rays_d.shape[0], 1)  # [batch, n_intrs]
        intersections = start + steps * step_size  # [batch, n_intrs]
        return intersections

    def render(sh_encoder, n_samples, angle, orig, img_size, patch):
        voxel_len = math.sqrt(3) / n_samples
        rot = scipy.spatial.transform.Rotation.from_euler('ZYX', angle, degrees=True)
        orig = rot.apply(orig)  # This is random
        rot_mat = rot.as_matrix()
        rot_mat = np.concatenate((rot_mat, orig.reshape(3, 1)), axis=1)
        rays = get_rays(img_size, img_size, focal=img_size * 2, c2w=torch.from_numpy(rot_mat))
        rays_o = rays[0].view(-1, 3).float()
        rays_d = rays[1].view(-1, 3).float()
        intersections = get_intersections(
            rays_o=rays_o, rays_d=rays_d, step_size=voxel_len,
            n_samples=n_samples, radius=1.0)
        intersections_trunc = intersections[:, :-1]
        intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)
        intrs_pts = intrs_pts.flip(-1).to(device=patch.device)
        intersections = intersections.to(device=patch.device)
        rays_d = rays_d.to(device=patch.device)
        data_interp = F.grid_sample(
            patch.unsqueeze(0), intrs_pts.view(1, -1, 1, 1, 3), mode='bilinear',
            align_corners=False, padding_mode='zeros')  # [1, ch, n, 1, 1]
        data_interp = data_interp.squeeze().transpose(0, 1)
        sigma = data_interp[:, -1].view(intersections_trunc.shape)
        cdata = data_interp[:, :-1].view(*intersections_trunc.shape, -1)

        sigma = F.relu(sigma)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)
        sh_mult = sh_encoder(rays_d)  # batch, ch/3
        sh_mult = sh_mult.unsqueeze(1).unsqueeze(1).expand(-1, cdata.shape[1], -1, -1)
        cdata = cdata.view(cdata.shape[0], cdata.shape[1], 3, sh_mult.shape[-1])

        rgb = torch.sum(sh_mult * cdata, dim=-1)
        rgb = shrgb2rgb(rgb, abs_light, True)
        return rgb.cpu()

    with torch.autograd.no_grad():
        atoms = renderer.atoms
        if isinstance(atoms, nn.ParameterList):
            atoms = atoms[patch_level].detach().float()
        sh_encoder = renderer.sh_encoder
        if atoms.dim() == 3:
            reso = int(np.round(atoms.shape[0] ** (1/3)))
            atoms = atoms.view(reso, reso, reso, atoms.shape[1], atoms.shape[2])
        atoms = atoms.permute(3, 4, 0, 1, 2)  # n_atoms, data_dim, reso, reso, reso

        n_atoms_perside = min(8, int(math.sqrt(atoms.shape[0])))
        fig, ax = plt.subplots(nrows=n_atoms_perside, ncols=n_atoms_perside)
        ax = ax.flatten()
        origin = np.array([0, 0, 4], dtype=np.float32)

        for i in range(len(ax)):
            rgb = render(sh_encoder, n_samples=40, angle=(0, 0, 90), orig=origin, img_size=60, patch=atoms[i])
            ax[i].imshow(rgb.view(int(math.sqrt(rgb.shape[0])), int(math.sqrt(rgb.shape[0])), 3))
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        fig.tight_layout(pad=0.4, h_pad=0.0, w_pad=0.0)
        save_image(fig, log_dir, f"patches-{patch_level}", iteration, summary_writer)


def user_ask_options(prompt: str, opt1: str, opt2: str) -> str:
    prompt_wopt = f"{prompt} ({opt1}, {opt2})"
    while True:
        out = input(prompt_wopt)
        if out == opt1:
            return opt1
        elif out == opt2:
            return opt2
        else:
            print(f"Invalid option {out}. Please type '{opt1}' or '{opt2}'")
