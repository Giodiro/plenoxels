from typing import Dict, List, Optional
import os
from collections import defaultdict
import time

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.configs import multiscene_config, parse_config
from plenoxels.ema import EMA
from plenoxels.models import DictPlenoxels
from plenoxels.tc_harmonics import plenoxel_sh_encoder

from plenoxels.runners.utils import *

TB_WRITER = None


def losses_to_postfix(losses: List[Dict[str, EMA]]) -> str:
    pfix_list = []
    for dset_id, loss_dict in enumerate(losses):
        pfix_inner = ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())
        pfix_list.append(f"D{dset_id}({pfix_inner})")
    return '  '.join(pfix_list)


# noinspection PyUnresolvedReferences,PyProtectedMember
def train_epoch(renderer,
                tr_loaders,
                ts_dsets,
                optim,
                lr_sched: Optional[torch.optim.lr_scheduler._LRScheduler],
                l1_coef,
                tv_coef,
                consistency_coef,
                batches_per_epoch,
                epochs,
                log_dir,
                batch_size,
                start_epoch=0,
                start_tot_step=0,
                train_fp16: bool = False,
                ):
    batches_per_dset = 10
    ema_weight = 0.3  # this is only for printing of loss to screen.
    num_dsets = len(tr_loaders)

    tr_iterators = [[iter(dl) for dl in tr_loader] for tr_loader in tr_loaders]
    n_loader_levels = len(tr_loaders[0]) - 1
    renderer.cuda()

    grad_scaler = torch.cuda.amp.GradScaler()

    tot_step = start_tot_step
    TB_WRITER.add_scalar("lr", optim.param_groups[0]["lr"], tot_step)
    for e in range(start_epoch, epochs):
        losses = [defaultdict(lambda: EMA(ema_weight)) for _ in range(num_dsets)]
        pb = tqdm(total=batches_per_epoch * num_dsets, desc=f"Epoch {e + 1}")
        level = -1
        renderer.train()
        for _ in range(0, batches_per_epoch, batches_per_dset):
            # Each epoch, rotate what resolution is being focused on
            for dset_id in range(num_dsets):
                for i in range(batches_per_dset):
                    level = (level + 1) % len(renderer.atoms)
                    try:
                        rays_o, rays_d, imgs = next(tr_iterators[dset_id][min(level, n_loader_levels)])
                        imgs = imgs.cuda()
                        rays_o = rays_o.cuda()
                        rays_d = rays_d.cuda()
                        optim.zero_grad()
                        rgb_preds, alpha, depth, consistency_loss = renderer(
                            rays_o, rays_d, grid_id=dset_id, consistency_coef=consistency_coef,
                            level=level, run_fp16=train_fp16)

                        # Compute and re-weight all the losses
                        diff_losses = dict(mse=F.mse_loss(rgb_preds, imgs))
                        if l1_coef > 0:
                            diff_losses["l1"] = l1_coef * torch.abs(renderer.grids[dset_id]).mean()
                        if tv_coef > 0:
                            diff_losses["tv"] = tv_coef * renderer.tv_loss(dset_id)
                        if consistency_coef > 0 and consistency_loss is not None:
                            diff_losses["consistency"] = consistency_coef * consistency_loss
                        loss = sum(diff_losses.values())
                        if train_fp16:
                            grad_scaler.scale(loss).backward()
                            grad_scaler.step(optim)
                            grad_scaler.update()
                        else:
                            loss.backward()
                            optim.step()

                        # Clip all the weights to be nonnegative
                        # for grid in renderer.grids:
                        #     grid.data = torch.clamp(grid.data, min=0.0)
                        # # Normalize all the atoms to be unit norm
                        # for atoms in renderer.atoms:
                        #     # Compute the atom norms
                        #     shape = atoms.shape # [..., n_atoms, data_dim]
                        #     dims = [i for i in range(len(shape))]
                        #     newdims = [dims[-2]] + dims[0:-2] + dims[-1:]
                        #     norms = torch.permute(atoms, newdims) # [n_atoms, ..., data_dim]
                        #     norms = torch.reshape(norms, (len(norms), -1))
                        #     norms = torch.linalg.vector_norm(norms, dim=-1)
                        #     # Clip so we don't divide by zero
                        #     norms = torch.clamp(norms, min=1e-5)
                        #     broadcastable_norms = torch.empty(size=[1]*(len(shape)-2) + [len(norms)] + [1])
                        #     broadcastable_norms[...,:,0] = norms
                        #     # Normalize each atom
                        #     atoms.data = torch.div(atoms, broadcastable_norms.to(atoms.device))

                        for loss_name, loss_val in diff_losses.items():
                            losses[dset_id][loss_name].update(loss_val.item())
                            TB_WRITER.add_scalar(
                                f"{loss_name}/D{dset_id}", loss_val.item(), tot_step)
                            pb.set_postfix_str(losses_to_postfix(losses), refresh=False)
                        pb.update(1)
                        tot_step += 1
                    except StopIteration:
                        # Reset the training-iterator which has no more samples
                        tr_iterators[dset_id][min(level, n_loader_levels)] = iter(tr_loaders[dset_id][min(level, n_loader_levels)])
            if lr_sched is not None:
                lr_sched.step()
                TB_WRITER.add_scalar("lr", lr_sched.get_last_lr()[0], tot_step)  # one lr per parameter-group
        pb.close()
        # Save and evaluate model
        time_s = time.time()
        renderer.eval()
        for ts_dset_id, ts_dset in enumerate(ts_dsets):
            psnr = plot_ts_imageio(
                ts_dset, ts_dset_id, renderer, log_dir,
                iteration=tot_step, batch_size=batch_size, image_id=0, verbose=True,
                summary_writer=TB_WRITER)
            render_patches(renderer, patch_level=0, log_dir=log_dir, iteration=tot_step,
                           summary_writer=TB_WRITER)
            if len(renderer.atoms) > 1:
                render_patches(renderer, patch_level=1, log_dir=log_dir, iteration=tot_step,
                               summary_writer=TB_WRITER)
            TB_WRITER.add_scalar(f"TestPSNR/D{ts_dset_id}", psnr, tot_step)
        torch.save({
            'epoch': e,
            'tot_step': tot_step,
            'scheduler': lr_sched.state_dict() if lr_sched is not None else None,
            'optimizer': optim.state_dict(),
            'model': renderer.state_dict(),
        }, os.path.join(log_dir, "model.pt"))
        print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")


def test_model(renderer, ts_dsets, log_dir, batch_size, num_test_imgs=1):
    renderer.cuda()
    renderer.eval()
    for ts_dset_id, ts_dset in enumerate(ts_dsets):
        psnrs = []
        for image_id in tqdm(range(num_test_imgs), desc=f"test-dataset {ts_dset_id} evaluation"):
            psnr = plot_ts_imageio(ts_dset, ts_dset_id, renderer, log_dir, iteration="test",
                                   batch_size=batch_size, image_id=image_id, verbose=False)
            psnrs.append(psnr)
        print(f"Average PSNR (over {num_test_imgs} poses) for "
              f"dataset {ts_dset_id}: {np.mean(psnrs):.2f}")


def init_model(cfg, tr_dsets, efficient_dict, checkpoint_data=None):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    radii = [dset[0].radius for dset in tr_dsets]
    renderer = DictPlenoxels(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder,
        radius=radii, num_atoms=cfg.model.num_atoms, num_scenes=len(tr_dsets),
        fine_reso=cfg.model.fine_reso, coarse_reso=cfg.model.coarse_reso,
        efficient_dict=efficient_dict, noise_std=cfg.model.noise_std, use_csrc=cfg.use_csrc)
    if checkpoint_data is not None:
        renderer.load_state_dict(checkpoint_data['model'])
        print("=> Loaded model state from checkpoint")
    return renderer


# noinspection PyUnresolvedReferences,PyProtectedMember
def init_lr_scheduler(cfg, optim, checkpoint_data=None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    eta_min = 1e-4
    num_batches_per_dset = 10
    lr_sched = None
    if cfg_.optim.cosine:
        num_dsets = len(cfg.data.datadirs)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.optim.num_epochs * cfg.optim.batches_per_epoch * num_dsets // num_batches_per_dset,
            eta_min=eta_min)
        if checkpoint_data is not None:
            lr_sched.load_state_dict(checkpoint_data['scheduler'])
            print("=> Loaded scheduler state from checkpoint")
    return lr_sched


def init_optim(cfg, model, transfer_learning=False, checkpoint_data=None) -> torch.optim.Optimizer:
    if transfer_learning:
        optim = torch.optim.Adam(model.grids, lr=cfg.optim.lr)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    if checkpoint_data is not None:
        optim.load_state_dict(checkpoint_data['optimizer'])
        print("=> Loaded optimizer state from checkpoint")
    return optim


if __name__ == "__main__":
    train_cfg, reload_cfg = parse_config(multiscene_config.get_cfg_defaults())
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
    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(cfg_)

    if reload_cfg is not None and train_cfg is None:
        # We're reloading an existing model for either training or testing.
        # Here cfg_ is reload_cfg.
        chosen_opt = user_ask_options(
            "Restart model training from checkpoint or evaluate model?", "train", "test")
        checkpoint_data = torch.load(os.path.join(train_log_dir, "model.pt"), map_location='cpu')
        model_ = init_model(cfg_, tr_dsets=tr_dsets_, efficient_dict=False, checkpoint_data=checkpoint_data)
        if chosen_opt == "test":
            print("Running tests only.")
            test_model(renderer=model_, ts_dsets=ts_dsets_, log_dir=train_log_dir,
                       batch_size=reload_cfg.optim.batch_size, num_test_imgs=len(ts_dsets_[0]))
        else:
            print(f"Resuming training from epoch {checkpoint_data['epoch'] + 1}")
            optim_ = init_optim(cfg_, model_, checkpoint_data=checkpoint_data)
            sched_ = init_lr_scheduler(cfg_, optim_, checkpoint_data=checkpoint_data)
            train_epoch(
                renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                batches_per_epoch=cfg_.optim.batches_per_epoch, epochs=cfg_.optim.num_epochs,
                log_dir=train_log_dir, batch_size=cfg_.optim.batch_size,
                l1_coef=cfg_.optim.regularization.l1_weight, tv_coef=cfg_.optim.regularization.tv_weight,
                consistency_coef=cfg_.optim.regularization.consistency_weight, lr_sched=sched_,
                start_epoch=checkpoint_data['epoch'] + 1, start_tot_step=checkpoint_data['tot_step'] + 1,
                train_fp16=cfg_.optim.train_fp16)
    elif reload_cfg is not None:
        # We're doing a transfer learning experiment. Loading a pretrained model, and fine-tuning on
        # new data.
        print("Applying pretrained patch dicts to new scenes")
        reload_log_dir = os.path.join(reload_cfg.logdir, reload_cfg.expname)
        checkpoint_data = torch.load(os.path.join(reload_log_dir, "model.pt"), map_location='cpu')

        # Pretrained model
        pretrained_model = init_model(
            reload_cfg, tr_dsets=tr_dsets_, efficient_dict=False, checkpoint_data=checkpoint_data)
        # Initialize a new model, but use the trained patch dictionaries.
        fresh_model = init_model(
            train_cfg, tr_dsets=tr_dsets_, efficient_dict=False, checkpoint_data=None)
        assert fresh_model.atoms.shape == pretrained_model.atoms.shape, "Can't transfer due to config mismatch."
        fresh_model.atoms = pretrained_model.atoms
        pretrained_model = fresh_model
        # Only optimize the coarse grids and don't reload any optimizer state.
        optim_ = init_optim(train_cfg, pretrained_model, transfer_learning=True, checkpoint_data=None)
        sched_ = init_lr_scheduler(train_cfg, optim_, checkpoint_data=None)
        train_epoch(renderer=pretrained_model, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                    batches_per_epoch=train_cfg.optim.batches_per_epoch,
                    epochs=train_cfg.optim.num_epochs,
                    log_dir=train_log_dir, batch_size=train_cfg.optim.batch_size,
                    l1_coef=train_cfg.optim.regularization.l1_weight,
                    tv_coef=train_cfg.optim.regularization.tv_weight,
                    consistency_coef=train_cfg.optim.regularization.consistency_weight,
                    lr_sched=sched_, train_fp16=train_cfg.optim.train_fp16)
    else:
        # Normal training.
        with open(os.path.join(train_log_dir, "config.yaml"), "w") as fh:
            fh.write(cfg_.dump())
        model_ = init_model(cfg_, tr_dsets=tr_dsets_, efficient_dict=False)
        optim_ = init_optim(cfg_, model_)
        sched_ = init_lr_scheduler(cfg_, optim_)
        train_epoch(
            renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
            batches_per_epoch=cfg_.optim.batches_per_epoch, epochs=cfg_.optim.num_epochs,
            log_dir=train_log_dir, batch_size=cfg_.optim.batch_size,
            l1_coef=cfg_.optim.regularization.l1_weight, tv_coef=cfg_.optim.regularization.tv_weight,
            consistency_coef=cfg_.optim.regularization.consistency_weight, lr_sched=sched_,
            train_fp16=cfg_.optim.train_fp16)
