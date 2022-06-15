import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.configs import parse_config, tensorrf_config
from plenoxels.ema import EMA
from plenoxels.models.tensor_rf import TensorRf
from plenoxels.runners.utils import *
from plenoxels.tc_harmonics import plenoxel_sh_encoder

"""
Training procedure


Regularizers:
 - ortho
 - l1
 - TV on sigma and rgb

update_Alphamask_list (default [2000,4000])
    call updateAlphaMask, and shrink bounding box (only at 2000).

upsamp_list ([2000,3000,4000,5500,7000])
    call upsample_volume_grid
    reset learning rate? (default true)
    upsampling is log between n_voxel_init (128^3) to n_voxel_final (300^3)
"""

TB_WRITER = None


def losses_to_postfix(losses: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in losses.items())


def get_lr_factor(cfg):
    lr_decay_iters = cfg.optim.lr_decay_iters if cfg.optim.lr_decay_iters > 0 else cfg.optim.max_steps
    lr_factor = cfg.optim.lr_decay_target_ratio ** (1 / lr_decay_iters)
    print(f"lr decay targets ratio of {cfg.optim.lr_decay_target_ratio} in {lr_decay_iters} steps.")
    return lr_factor


def render_wrap(renderer):
    def render(ro, rd):
        return renderer(ro, rd)[0]

    return render


def train(renderer: TensorRf, tr_loader, ts_dset, optim, log_dir, cfg, batch_size,
          start_step: int = 0):
    renderer.cuda()
    renderer.train()
    losses = defaultdict(lambda: EMA(.3))

    upsample_iters = cfg.optim.upsample_iters
    reso_list = torch.logspace(
        math.log(cfg.model.reso_init), math.log(cfg.model.reso_final), len(upsample_iters) + 1,
        base=math.e
    ).round().long().tolist()[1:]
    # Exclude resolutions which we may have already reached (if resuming from checkpoint).
    reso_list = [r for r in reso_list if r > renderer.resolution]
    lr_factor = get_lr_factor(cfg)

    tr_iterator = iter(tr_loader)
    pb = tqdm(range(start_step, cfg.optim.max_steps))
    for tot_step in pb:
        try:
            batch = next(tr_iterator)
        except StopIteration:
            tr_iterator = iter(tr_loader)
            batch = next(tr_iterator)

        rays_o, rays_d, imgs = batch
        imgs = imgs.cuda()
        rays_o = rays_o.cuda()
        rays_d = rays_d.cuda()

        rgb_preds, depth = renderer(rays_o, rays_d)
        diff_losses = {}
        diff_losses["mse"] = F.mse_loss(rgb_preds, imgs)
        loss = sum(diff_losses.values())
        optim.zero_grad()
        loss.backward()
        optim.step()

        diff_losses["mse"] = diff_losses["mse"].detach().item()
        diff_losses["psnr"] = -10.0 * np.log(diff_losses["mse"]) / np.log(10.0)
        for loss_name, loss_val in diff_losses.items():
            try:
                loss_val_ = loss_val.detach().item()
            except AttributeError:
                loss_val_ = loss_val
            losses[loss_name].update(loss_val_)
            TB_WRITER.add_scalar(f"TensorRF/{loss_name}", loss_val_, tot_step)
            pb.set_postfix_str(losses_to_postfix(losses), refresh=False)
        TB_WRITER.add_scalar("TensorRF/lr", optim.param_groups[0]['lr'],
                             tot_step)  # one lr per parameter-group
        # Save and evaluate model
        if tot_step % cfg.optim.test_every == cfg.optim.test_every - 1:
            time_s = time.time()
            renderer.eval()
            psnr = plot_ts(
                ts_dset, 0, renderer, log_dir, render_fn=render_wrap(renderer),
                iteration=tot_step, batch_size=batch_size, image_id=0, verbose=True,
                summary_writer=TB_WRITER, plot_type="imageio")
            TB_WRITER.add_scalar(f"TensorRF/TestPSNR", psnr, tot_step)
            renderer.train()
            torch.save({
                'tot_step': tot_step,
                'optimizer': optim.state_dict(),
                'model': {
                    'weights': renderer.state_dict(),
                    'resolution': renderer.resolution,
                }
            }, os.path.join(log_dir, "model.pt"))
            print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")
        tot_step += 1
        pb.update(1)

        # Update optimization and model parameters
        for param_group in optim.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
        if tot_step in upsample_iters:
            reso_new = reso_list.pop(0)
            renderer.upsample_volume_grid(reso_new)
            optim = init_optim(cfg, renderer, None)
    pb.close()


def test_model(renderer, ts_dset, log_dir, batch_size, num_test_imgs=1):
    renderer.cuda()
    renderer.eval()
    psnrs = []
    for image_id in tqdm(range(num_test_imgs), desc="test-dataset evaluation"):
        psnr = plot_ts(ts_dset, 0, renderer, log_dir, iteration="test",
                       batch_size=batch_size, image_id=image_id, verbose=False,
                       render_fn=render_wrap(renderer), plot_type="matplotlib")
        psnrs.append(psnr)
    print(f"Average PSNR over {num_test_imgs} poses: {np.mean(psnrs):.2f}")


def init_model(cfg, tr_dset, checkpoint_data=None):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    renderer = TensorRf(radius=tr_dset.radius, resolution=cfg.model.reso_init,
                        sh_encoder=sh_encoder, n_rgb_comp=cfg.model.n_rgb_comp,
                        n_sigma_comp=cfg.model.n_sigma_comp,
                        sh_deg=cfg.sh.degree, abs_light_thresh=cfg.model.abs_light_thresh)
    if checkpoint_data is not None and checkpoint_data.get('model', None) is not None:
        renderer.load_state_dict(checkpoint_data['model']['weights'])
        renderer.resolution = checkpoint_data['model']['resolution']
        renderer.update_stepsize()
        print("=> Loaded model state from checkpoint")
    return renderer


def init_optim(cfg, model: TensorRf, checkpoint_data=None):
    spatial_params = (
                list(model.density_plane.parameters()) + list(model.density_line.parameters()) +
                list(model.rgb_plane.parameters()) + list(model.rgb_line.parameters()))
    optim = torch.optim.Adam([
        {'params': spatial_params, 'lr': cfg.optim.lr_init_spatial},
        {'params': model.basis_mat.parameters(), 'lr': cfg.optim.lr_init_network}
    ], betas=(0.9, 0.99))
    if checkpoint_data is not None and checkpoint_data.get("optimizer", None) is not None:
        optim.load_state_dict(checkpoint_data['optimizer'])
        print("=> Loaded optimizer state from checkpoint")
    return optim


if __name__ == "__main__":
    train_cfg, reload_cfg = parse_config(tensorrf_config.get_cfg_defaults())
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
    checkpoint_data_ = {
        'epoch': -1,
        'tot_step': -1,
    }
    if reload_cfg is not None:
        assert train_cfg is None, "run_tensorrf.py does not support setting train_cfg and reload_cfg"
        chosen_opt = user_ask_options(
            "Restart model training from checkpoint or evaluate model?", "train", "test")
        checkpoint_data_ = torch.load(os.path.join(train_log_dir, "model.pt"), map_location='cpu')
        if chosen_opt == "test":
            print("Running tests only.")
            model_ = init_model(cfg_, tr_dset=tr_dset_, checkpoint_data=checkpoint_data_)
            test_model(renderer=model_, ts_dset=ts_dset_, log_dir=train_log_dir,
                       batch_size=reload_cfg.optim.batch_size, num_test_imgs=10)
            sys.exit(0)
        else:
            print(f"Resuming training from epoch {checkpoint_data_['epoch'] + 1}")
    else:
        # normal training
        with open(os.path.join(train_log_dir, "config.yaml"), "w") as fh:
            fh.write(cfg_.dump())

    model_ = init_model(cfg_, tr_dset=tr_dset_, checkpoint_data=checkpoint_data_)
    optim_ = init_optim(cfg_, model_, checkpoint_data=checkpoint_data_)
    train(renderer=model_, tr_loader=tr_loader_, ts_dset=ts_dset_, optim=optim_,
          log_dir=train_log_dir, cfg=cfg_, batch_size=cfg_.optim.batch_size,
          start_step=checkpoint_data_['tot_step'] + 1)
