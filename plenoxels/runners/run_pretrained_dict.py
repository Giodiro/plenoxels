from typing import Dict, List, Optional
import os
from collections import defaultdict
import time

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.configs import multiscene_config, parse_config, optimize_dict_config
from plenoxels.ema import EMA
from plenoxels.models import DictPlenoxels, make_weights_unit_norm
from plenoxels.models.single_res_multi_scene import SingleResoDictPlenoxels
from plenoxels.tc_harmonics import plenoxel_sh_encoder
from plenoxels.models import single_res_multi_scene

from plenoxels.runners.utils import *

TB_WRITER = None


def default_render_fn(renderer, dset_id):
    def render_fn(ro, rd):
        return renderer(ro, rd, dset_id)
    return render_fn


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
                l2_coef,
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
                        if l2_coef > 0:
                            diff_losses["l2"] += l2_coef * (renderer.cgrids[dset_id].square().sum(-1) - 1).mean()
                        if l1_coef > 0:
                            diff_losses["l1"] = 0
                            for scene_grid in renderer.grids[dset_id]:
                                diff_losses["l1"] += l1_coef * torch.abs(scene_grid).mean()
                        if tv_coef > 0:
                            diff_losses["tv"] = tv_coef * renderer.tv_loss(dset_id)
                        if consistency_coef > 0:
                            diff_losses["consistency"] = consistency_coef * renderer.closs_v2(dset_id, rays_d[:1])
                        loss = sum(diff_losses.values())
                        loss.backward()
                        optim.step()

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
        with torch.autograd.no_grad():
            for ts_dset_id, ts_dset in enumerate(ts_dsets):
                psnr = plot_ts(
                    ts_dset, ts_dset_id, renderer, log_dir, render_fn=default_render_fn(renderer, ts_dset_id),
                    iteration=tot_step, batch_size=batch_size, image_id=0, verbose=True,
                    summary_writer=TB_WRITER, plot_type="imageio")
                render_patches(renderer, patch_level=0, log_dir=log_dir, iteration=tot_step,
                               summary_writer=TB_WRITER)
                if len(renderer.atoms) > 1:
                    render_patches(renderer, patch_level=1, log_dir=log_dir, iteration=tot_step,
                                   summary_writer=TB_WRITER)
                TB_WRITER.add_scalar(f"TestPSNR/D{ts_dset_id}", psnr, tot_step)
            # TB_WRITER.add_histogram(f"patches/sigma", renderer.atoms[0][..., -1].view(-1), tot_step)
            # TB_WRITER.add_histogram(f"patches/R", renderer.atoms[0][..., 0].view(-1), tot_step)
            # TB_WRITER.add_histogram(f"patches/G", renderer.atoms[0][..., 1].view(-1), tot_step)
            # TB_WRITER.add_histogram(f"patches/B", renderer.atoms[0][..., 2].view(-1), tot_step)
            # TB_WRITER.add_histogram(f"weights/white", renderer.grids[0][0].view(renderer.coarse_reso, renderer.coarse_reso, renderer.coarse_reso, -1)[:5, :5, :5].reshape(-1), tot_step)
            torch.save({
                'epoch': e,
                'tot_step': tot_step,
                'scheduler': lr_sched.state_dict() if lr_sched is not None else None,
                'optimizer': optim.state_dict(),
                'model': renderer.state_dict(),
            }, os.path.join(log_dir, "model.pt"))
            print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")

# checkpoint_data is a pretrained dictionary
def init_model(cfg, tr_dsets, checkpoint_data):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    radii = [dset[0].radius for dset in tr_dsets]
    renderer = SingleResoDictPlenoxels(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder, radius=radii,
        num_atoms=cfg.model.num_atoms[0], num_scenes=len(tr_dsets),
        fine_reso=cfg.model.fine_reso[0], coarse_reso=cfg.model.coarse_reso,
        dict_only_sigma=False)
    atoms = checkpoint_data['model']['dictionary']  # [data_dim, patch_reso, patch_reso, patch_reso, num_atoms]
    # renderer.atoms.data = atoms.reshape(atoms.shape[0], atoms.shape[1]**3, -1).permute(1, 2, 0)  # [patch_reso**3, num_atoms, data_dim]
    print("=> Loaded model state from checkpoint")
    return renderer


# noinspection PyUnresolvedReferences,PyProtectedMember
def init_lr_scheduler(cfg, optim, checkpoint_data=None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    eta_min = 1e-4
    num_batches_per_dset = 10
    lr_sched = None
    if cfg.optim.cosine:
        num_dsets = len(cfg.data.datadirs)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.optim.num_epochs * cfg.optim.batches_per_epoch * num_dsets // num_batches_per_dset,
            eta_min=eta_min)
    return lr_sched


def init_optim(cfg, model, transfer_learning=False, checkpoint_data=None) -> torch.optim.Optimizer:
    # optim = torch.optim.Adam(model.cgrids, lr=cfg.optim.lr)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    return optim


if __name__ == "__main__":
    reload_cfg = optimize_dict_config.get_cfg_defaults()
    train_cfg, _ = parse_config(multiscene_config.get_cfg_defaults())
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
    train_log_dir = os.path.join(train_cfg.logdir, train_cfg.expname)
    os.makedirs(train_log_dir, exist_ok=True)
    TB_WRITER = SummaryWriter(log_dir=train_log_dir)
    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(train_cfg)

    assert reload_cfg is not None and train_cfg is not None
    reload_log_dir = os.path.join(reload_cfg.logdir, reload_cfg.expname)
    checkpoint_data = torch.load(os.path.join(reload_log_dir, "model.pt"), map_location='cpu')

    # Pretrained model
    model = init_model(
        train_cfg, tr_dsets=tr_dsets_, checkpoint_data=checkpoint_data)
    # Only optimize the coarse grids and don't reload any optimizer state.
    optim_ = init_optim(train_cfg, model, transfer_learning=True, checkpoint_data=None)
    sched_ = init_lr_scheduler(train_cfg, optim_, checkpoint_data=None)
    train_epoch(renderer=model, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                batches_per_epoch=train_cfg.optim.batches_per_epoch,
                epochs=train_cfg.optim.num_epochs,
                log_dir=train_log_dir, batch_size=train_cfg.optim.batch_size,
                l1_coef=train_cfg.optim.regularization.l1_weight,
                tv_coef=train_cfg.optim.regularization.tv_weight,
                consistency_coef=train_cfg.optim.regularization.consistency_weight,
                lr_sched=sched_, train_fp16=train_cfg.optim.train_fp16,
                l2_coef=train_cfg.optim.regularization.l2_weight)