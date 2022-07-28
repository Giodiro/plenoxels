import math
import os
import time
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.configs import vbr_config
from plenoxels.ema import EMA
from plenoxels.models.variable_bitrate_nerf import VBrNerf
from plenoxels.runners.utils import *

TB_WRITER = None


def default_render_fn(renderer, level=None):
    def render_fn(ro, rd):
        return renderer(ro, rd, level=level or len(renderer.reso_list) - 1)[0]
    return render_fn


def train_epoch(renderer, tr_loader, ts_dset, optim, lr_sched, max_epochs, log_dir, batch_size,
                start_epoch=0, start_step=0):
    ema_weight = 0.3

    renderer.cuda()
    tot_step = start_step
    TB_WRITER.add_scalar("lr", optim.param_groups[0]["lr"], tot_step)
    for e in range(start_epoch, max_epochs):
        losses = defaultdict(lambda: EMA(ema_weight))
        renderer.train()
        pb = tqdm(total=len(tr_loader), desc=f"Epoch {e + 1}")
        for i, batch in enumerate(tr_loader):
            rays_o, rays_d, imgs = batch
            imgs = imgs.cuda()
            rays_o = rays_o.cuda()
            rays_d = rays_d.cuda()
            optim.zero_grad()

            rgb_preds, _ = renderer(rays_o, rays_d, None)
            loss = F.mse_loss(rgb_preds, imgs)
            loss.backward()
            optim.step()

            loss_val = loss.item()
            losses["mse"].update(loss_val)
            TB_WRITER.add_scalar(f"mse", loss_val, tot_step)
            pb.set_postfix_str(f"mse={loss_val:.4f} - psnr={-10 * math.log10(loss_val):.4f}", refresh=False)
            pb.update(1)
            tot_step += 1
            if lr_sched is not None:
                lr_sched.step()
                TB_WRITER.add_scalar("lr", lr_sched.get_last_lr()[0], tot_step)  # one lr per parameter-group
        pb.close()
        # Save and evaluate model
        time_s = time.time()
        psnr = plot_ts(
            ts_dset, 0, renderer, log_dir,
            iteration=tot_step, batch_size=batch_size, image_id=0, verbose=True,
            summary_writer=TB_WRITER, render_fn=default_render_fn(renderer), plot_type="imageio")
        TB_WRITER.add_scalar(f"TestPSNR", psnr, tot_step)
        torch.save({
            'epoch': e,
            'tot_step': tot_step,
            'scheduler': lr_sched.state_dict() if lr_sched is not None else None,
            'optimizer': optim.state_dict(),
            'model': renderer.state_dict(),
        }, os.path.join(log_dir, "model.pt"))
        print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")


def init_model(cfg, tr_dset, checkpoint_data=None):
    renderer = VBrNerf(reso_list=[2**5, 2**6, 2**7], cb_bits=6, scene_radius=tr_dset.radius)
    if checkpoint_data is not None and checkpoint_data.get('model', None) is not None:
        renderer.load_state_dict(checkpoint_data['model'])
        print("=> Loaded model state from checkpoint")
    return renderer


# noinspection PyUnresolvedReferences,PyProtectedMember
def init_lr_scheduler(cfg, optim, num_batches_per_dset: int, checkpoint_data=None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    return None


def init_optim(cfg, model, checkpoint_data=None):
    optim = torch.optim.Adam([
        {'params': [vbr_layer.grid for vbr_layer in model.grids], 'lr': 0.1},
        {'params': model.renderer.parameters(), 'lr': 0.001},
        {'params': [vbr_layer.codebook for vbr_layer in model.grids], 'lr': 0.001},
    ], lr=0.001)
    if checkpoint_data is not None and checkpoint_data.get("optimizer", None) is not None:
        optim.load_state_dict(checkpoint_data['optimizer'])
        print("=> Loaded optimizer state from checkpoint")
    return optim


if __name__ == "__main__":
    num_epochs_ = 600
    cfg_ = vbr_config.get_cfg_defaults()

    torch.manual_seed(cfg_.seed)
    train_log_dir = os.path.join(cfg_.logdir, cfg_.expname)
    os.makedirs(train_log_dir, exist_ok=True)
    TB_WRITER = SummaryWriter(log_dir=train_log_dir)

    tr_dset_, tr_loader_, ts_dset_ = init_data_single_dset(cfg_)
    checkpoint_data_ = {
        'epoch': -1,
        'tot_step': -1,
    }

    model_ = init_model(cfg_, tr_dset=tr_dset_, checkpoint_data=checkpoint_data_)
    optim_ = init_optim(cfg_, model_, checkpoint_data=checkpoint_data_)
    sched_ = init_lr_scheduler(cfg_, optim_, 1, checkpoint_data=checkpoint_data_)
    train_epoch(
        renderer=model_, tr_loader=tr_loader_, ts_dset=ts_dset_, optim=optim_,
        max_epochs=num_epochs_, log_dir=train_log_dir,
        batch_size=cfg_.optim.batch_size, lr_sched=sched_,
        start_epoch=checkpoint_data_['epoch'] + 1,
        start_step=checkpoint_data_['tot_step'] + 1)
