import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.configs import parse_config, singlescene_config
from plenoxels.ema import EMA
from plenoxels.models.grid_plenoxel import RegularGrid
from plenoxels.tc_harmonics import plenoxel_sh_encoder

from plenoxels.runners.utils import *

TB_WRITER = None


def train_epoch(renderer, tr_loader, ts_dset, optim, lr_sched, max_epochs, log_dir, batch_size, start_epoch=0, start_step=0):
    ema_weight = 0.3

    renderer.cuda()
    tot_step = start_step
    TB_WRITER.add_scalar("lr", optim.param_groups[0]["lr"], tot_step)
    for e in range(start_epoch, max_epochs):
        losses = defaultdict(lambda: EMA(ema_weight))
        renderer.train()
        pb = tqdm(total=len(tr_loader), desc=f"Epoch {e + 1}")
        for batch in tr_loader:
            rays_o, rays_d, imgs = batch
            imgs = imgs.cuda()
            rays_o = rays_o.cuda()
            rays_d = rays_d.cuda()
            optim.zero_grad()

            rgb_preds = renderer(rays_o, rays_d)
            loss = F.mse_loss(rgb_preds, imgs)
            loss.backward()
            optim.step()

            loss_val = loss.item()
            losses["mse"].update(loss_val)
            TB_WRITER.add_scalar(f"mse", loss_val, tot_step)
            pb.set_postfix_str(f"mse={loss_val:.4f}", refresh=False)
            pb.update(1)
            tot_step += 1
            if lr_sched is not None:
                lr_sched.step()
                TB_WRITER.add_scalar("lr", lr_sched.get_last_lr()[0], tot_step)  # one lr per parameter-group
        pb.close()
        # Save and evaluate model
        time_s = time.time()
        psnr = plot_ts_imageio(
            ts_dset, 0, renderer, log_dir,
            iteration=tot_step, batch_size=batch_size, image_id=0, verbose=True,
            summary_writer=TB_WRITER, render_fn=lambda ro, rd: renderer(ro, rd))
        TB_WRITER.add_scalar(f"TestPSNR", psnr, tot_step)
        torch.save({
            'epoch': e,
            'tot_step': tot_step,
            'scheduler': lr_sched.state_dict() if lr_sched is not None else None,
            'optimizer': optim.state_dict(),
            'model': renderer.state_dict(),
        }, os.path.join(log_dir, "model.pt"))
        print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")


def test_model(renderer, ts_dset, log_dir, batch_size, num_test_imgs=1):
    renderer.cuda()
    renderer.eval()
    psnrs = []
    for image_id in range(num_test_imgs):
        psnr = plot_ts_imageio(ts_dset, 0, renderer, log_dir, iteration="test",
                               batch_size=batch_size, image_id=image_id, verbose=False,
                               render_fn=lambda ro, rd: renderer(ro, rd))
        psnrs.append(psnr)
    print(f"Average PSNR over {num_test_imgs} poses: {np.mean(psnrs):.2f}")


def init_model(cfg, tr_dset, checkpoint_data=None):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    renderer = RegularGrid(resolution=cfg.model.resolution, radius=tr_dset.radius,
                           sh_deg=cfg.sh.degree, sh_encoder=sh_encoder)
    if checkpoint_data is not None and checkpoint_data.get('model', None) is not None:
        renderer.load_state_dict(checkpoint_data['model'])
        print("=> Loaded model state from checkpoint")
    return renderer


# noinspection PyUnresolvedReferences,PyProtectedMember
def init_lr_scheduler(cfg, optim, num_batches_per_dset: int, checkpoint_data=None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    eta_min = 1e-4
    lr_sched = None
    if cfg_.optim.cosine:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.optim.num_epochs * num_batches_per_dset, eta_min=eta_min)
        if checkpoint_data is not None and checkpoint_data.get("scheduler", None) is not None:
            lr_sched.load_state_dict(checkpoint_data['scheduler'])
            print("=> Loaded scheduler state from checkpoint")
    return lr_sched


def init_optim(cfg, model, checkpoint_data=None):
    optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    if checkpoint_data is not None and checkpoint_data.get("optimizer", None) is not None:
        optim.load_state_dict(checkpoint_data['optimizer'])
        print("=> Loaded optimizer state from checkpoint")
    return optim


if __name__ == "__main__":
    train_cfg, reload_cfg = parse_config(singlescene_config.get_cfg_defaults())
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f'gpu is {gpu}')
    print(f'reload config is {reload_cfg}')
    print(f'train config is {train_cfg}')
    # Make config immutable
    if train_cfg is not None:
        train_cfg.freeze()
    if reload_cfg is not None:
        reload_cfg.freeze()
    # Set up the training
    cfg_ = train_cfg if train_cfg is not None else reload_cfg
    train_log_dir = os.path.join(cfg_.logdir, cfg_.expname)
    os.makedirs(train_log_dir, exist_ok=True)
    TB_WRITER = SummaryWriter(log_dir=train_log_dir)

    tr_dset_, tr_loader_, ts_dset_ = init_data_single_dset(cfg_)
    checkpoint_data = {
        'epoch': -1,
        'tot_step': -1,
    }
    if reload_cfg is not None:
        assert train_cfg is None, "run_regula_grid.py does not support setting train_cfg and reload_cfg"
        chosen_opt = user_ask_options(
            "Restart model training from checkpoint or evaluate model?", "train", "test")
        checkpoint_data = torch.load(os.path.join(train_log_dir, "model.pt"), map_location='cpu')
        if chosen_opt == "test":
            print("Running tests only.")
            model_ = init_model(cfg_, tr_dset=tr_dset_, checkpoint_data=checkpoint_data)
            test_model(renderer=model_, ts_dset=ts_dset_, log_dir=cfg_,
                       batch_size=reload_cfg.optim.batch_size, num_test_imgs=10)
            sys.exit(0)
        else:
            print(f"Resuming training from epoch {checkpoint_data['epoch'] + 1}")
    else:
        # normal training
        with open(os.path.join(train_log_dir, "config.yaml"), "w") as fh:
            fh.write(cfg_.dump())

    model_ = init_model(cfg_, tr_dset=tr_dset_, checkpoint_data=checkpoint_data)
    optim_ = init_optim(cfg_, model_, checkpoint_data=checkpoint_data)
    sched_ = init_lr_scheduler(
        cfg_, optim_, num_batches_per_dset=cfg_.optim.num_epochs * len(tr_loader_),
        checkpoint_data=checkpoint_data)
    train_epoch(
        renderer=model_, tr_loader=tr_loader_, ts_dset=ts_dset_, optim=optim_,
        max_epochs=cfg_.optim.num_epochs, log_dir=train_log_dir,
        batch_size=cfg_.optim.batch_size, lr_sched=sched_,
        start_epoch=checkpoint_data['epoch'] + 1,
        start_step=checkpoint_data['tot_step'] + 1)
