import math
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import svox
import svox_renderer
from svox_helpers import DataFormat
from tc_plenoptimize import init_datasets, init_profiler, parse_config


def train_interp_tree(cfg):
    h_degree = cfg.sh.degree
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tr_dset, tr_loader, ts_dset = init_datasets(cfg, dev)

    # Initialize model
    model = svox.N3Tree(
        N=2,
        data_dim=3 * (h_degree + 1) ** 2 + 1,
        depth_limit=8,
        init_reserve=64**3,
        init_refine=0,
        geom_resize_fact=1.,
        radius=(tr_dset.scene_bbox[1] - tr_dset.scene_bbox[0]) / 2,
        center=(tr_dset.scene_bbox[1] + tr_dset.scene_bbox[0]) / 2,
        device="cpu",
        dtype=torch.float32,
        data_format="SH%d" % ((h_degree + 1) ** 2),
    ).to(dev)
    with torch.autograd.no_grad():
        model.data[..., :-1].fill_(0.01)  # RGB
        model.data[..., -1].fill_(0.1)    # Density
    model.refine(3)
    renderer = svox_renderer.VolumeRenderer(
        tree=model,
        step_size=1e-5,
        background_brightness=1.0,
    )

    def init_optim(model_):
        if cfg.optim.optimizer.lower() == "sgd":
            lr_rgb = 150 * ((model_.data.shape[0] ** (1/3)) ** 1.75) * (cfg.optim.batch_size / 4000)
            return torch.optim.SGD(params=[
                {'params': (model_.data, ), 'lr': 1e3},
            ])
    optim = init_optim(model)

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
                rays_o = rays[:, 0].contiguous().to(device=dev)
                rays_d = rays[:, 1].contiguous().to(device=dev)
                imgs = imgs.to(device=dev)

                preds = renderer.forward(rays=svox_renderer.Rays(origins=rays_o, dirs=rays_d, viewdirs=rays_d))
                loss = F.mse_loss(preds, imgs)
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

                # if (i + 1) % cfg.optim.eval_refresh_rate == 0:
                #     ts_psnr = run_test_step(
                #         ts_dset, model, render_every=cfg.optim.render_refresh_rate,
                #         log_dir=cfg.logdir, iteration=epoch * len(tr_loader) + i + 1,
                #         batch_size=cfg.optim.batch_size, device=dev, exp_name=cfg.expname)
                #     print(f"Epoch {epoch} - iteration {i}: Test PSNR: {ts_psnr:.4f}")

                # Profiling
                if p is not None:
                    p.step()


if __name__ == "__main__":
    _cfg = parse_config()
    train_interp_tree(_cfg)
