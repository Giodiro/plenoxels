import math
import os
from contextlib import ExitStack
from typing import Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import svox
import svox_renderer
from synthetic_nerf_dataset import SyntheticNerfDataset
from tc_plenoptimize import init_datasets, init_profiler, parse_config
import plenoxels.c_ext as _C


def run_test_step(test_dset: SyntheticNerfDataset,
                  model,
                  renderer,
                  render_every: int,
                  log_dir: str,
                  iteration: int,
                  batch_size: int,
                  device: Union[torch.device, str],
                  exp_name: str) -> float:
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

                rgb_map_b = renderer.forward(rays=svox_renderer.Rays(origins=rays_o, dirs=rays_d, viewdirs=rays_d))
                rgb_map.append(rgb_map_b.cpu())
                # depth.append(depth_b.cpu())
            rgb_map = torch.stack(rgb_map, 0).reshape(test_dset.img_h, test_dset.img_w, 3)
            # depth = torch.stack(depth).reshape(test_dset.img_h, test_dset.img_w)

            # Compute loss metrics
            mse = torch.mean((rgb_map - rgb) ** 2)
            psnr = -10.0 * torch.log(mse) / math.log(10)
            total_psnr += psnr

            # Render and save result to file
            if (i + 1) % render_every == 0:
                # depth = depth.unsqueeze(-1).repeat(1, 1, 3)
                # out_img = torch.cat((rgb, rgb_map, depth), dim=1)
                out_img = torch.cat((rgb, rgb_map), dim=1)
                out_img = (out_img * 255).to(dtype=torch.uint8).numpy()
                out_dir = os.path.join(log_dir, exp_name)
                os.makedirs(out_dir, exist_ok=True)
                imageio.imwrite(os.path.join(out_dir, f"{iteration:05}_test_img_{i:04}.png"), out_img)
    return total_psnr / i


def init_tree(tree_type, levels, sh_degree, scene_bbox, dtype, device):
    if tree_type.lower() == "n3tree":
        model = svox.N3Tree(
            N=2,
            data_dim=3 * (sh_degree + 1) ** 2 + 1,
            depth_limit=8,
            init_reserve=64**3,
            init_refine=levels,
            geom_resize_fact=1.,
            radius=(scene_bbox[1] - scene_bbox[0]) / 2,
            center=(scene_bbox[1] + scene_bbox[0]) / 2,
            device="cpu",
            dtype=dtype,
            data_format="SH%d" % ((sh_degree + 1) ** 2),
        ).to(device)
        renderer = svox_renderer.VolumeRenderer(
            tree=model,
            step_size=1e-5,
            background_brightness=1.0,
        )
    elif tree_type.lower() == "octree":
        model = _C.Octreef2d2(
            levels, True, device,
            (scene_bbox[1] - scene_bbox[0]) / 2, (scene_bbox[1] + scene_bbox[0]) / 2
        )
        renderer = svox_renderer.SimpleVolumeRenderer(
            tree=model, background_brightness=1.0
        )
        for i in range(levels):
            model.refine(None)
    else:
        raise RuntimeError(f"Tree type {tree_type} not recognized.")
    return model, renderer


def init_optimizer(tree_type, tree, optim_type, lr):
    if tree_type.lower() in {"n3tree", "octree"}:
        params = (tree.data, )
    else:
        raise RuntimeError(f"Tree type {tree_type} not recognized.")

    if optim_type.lower() == "sgd":
        return torch.optim.SGD(params=params, lr=lr)
    else:
        raise RuntimeError(f"Optimizer type {optim_type} not recognized.")


def train_interp_tree(cfg):
    sh_degree = cfg.sh.degree
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tr_dset, tr_loader, ts_dset = init_datasets(cfg, dev)

    # Initialize model, optimizer
    model, renderer = init_tree("octree", levels=2, sh_degree=sh_degree,
                                scene_bbox=tr_dset.scene_bbox, dtype=torch.float32,
                                device=torch.device('cuda:0'))
    optim = init_optimizer("octree", tree=model, optim_type=cfg.optim.optimizer, lr=cfg.optim.lr)
    # Initialize data
    with torch.autograd.no_grad():
        model.data[..., :-1].fill_(0.01)  # RGB
        model.data[..., -1].fill_(0.1)    # Density

    # Main iteration starts here
    for epoch in range(cfg.optim.num_epochs):
        psnrs, mses = [], []
        with ExitStack() as stack:
            p = None
            if cfg.optim.profile:
                p = init_profiler(cfg, dev)
                stack.enter_context(p)
            for i, batch in tqdm(enumerate(tr_loader), desc=f"Epoch {epoch}"):
                model.train()
                optim.zero_grad()
                rays, imgs = batch
                rays_o = rays[:, 0].contiguous().to(device=dev)
                rays_d = rays[:, 1].contiguous().to(device=dev)
                imgs = imgs.to(device=dev)

                print(model.is_child_leaf)
                preds = renderer.forward(rays=svox_renderer.Rays(origins=rays_o, dirs=rays_d, viewdirs=rays_d))
                print("preds", preds[:5])
                loss = F.mse_loss(preds, imgs)
                print("loss", loss)
                loss.backward()
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
                        ts_dset, model, renderer=renderer, render_every=cfg.optim.render_refresh_rate,
                        log_dir=cfg.logdir, iteration=epoch * len(tr_loader) + i + 1,
                        batch_size=cfg.optim.batch_size, device=dev, exp_name=cfg.expname)
                    print(f"Epoch {epoch} - iteration {i}: Test PSNR: {ts_psnr:.4f}")

                # Profiling
                if p is not None:
                    p.step()


if __name__ == "__main__":
    _cfg = parse_config()
    train_interp_tree(_cfg)
