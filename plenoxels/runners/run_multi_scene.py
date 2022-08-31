import math
from typing import Dict, List, Optional
import os
from collections import defaultdict
import time

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.configs import singlescene_config, multiscene_config, parse_config
from plenoxels.ema import EMA
from plenoxels.models import DictPlenoxels
# from plenoxels.models.grids.learnable_hash import LearnableHash
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from plenoxels.models.single_res_multi_scene import SingleResoDictPlenoxels
from plenoxels.models.superres import SuperResoPlenoxel
from plenoxels.tc_harmonics import plenoxel_sh_encoder

from plenoxels.runners.utils import *

TB_WRITER = None
GRID_CONFIG = """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": 128,
        "rank": 10,
        "init_std": 0.15,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": 8,
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""


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

    tr_iterators = [iter(tr_loader) for tr_loader in tr_loaders]

    grad_scaler = torch.cuda.amp.GradScaler(enabled=train_fp16)
    renderer.cuda()
    tot_step = start_tot_step
    TB_WRITER.add_scalar("lr", optim.param_groups[0]["lr"], tot_step)
    for e in range(start_epoch, epochs):
        losses = [defaultdict(lambda: EMA(ema_weight)) for _ in range(num_dsets)]
        pb = tqdm(total=batches_per_epoch * num_dsets, desc=f"Epoch {e + 1}")
        renderer.train()
        for _ in range(0, batches_per_epoch, batches_per_dset):
            # Each epoch, rotate what resolution is being focused on
            for dset_id in range(num_dsets):
                for i in range(batches_per_dset):
                    try:
                        rays_o, rays_d, imgs = next(tr_iterators[dset_id])
                        imgs = imgs.cuda()
                        rays_o = rays_o.cuda()
                        rays_d = rays_d.cuda()
                        optim.zero_grad()

                        with torch.cuda.amp.autocast(enabled=train_fp16):
                            rgb_preds = renderer(rays_o, rays_d, grid_id=dset_id)
                            loss = F.mse_loss(rgb_preds, imgs)

                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optim)
                        grad_scaler.update()

                        loss_val = loss.item()
                        losses[dset_id]["mse"].update(loss_val)
                        losses[dset_id]["psnr"].update(-10 * math.log10(loss_val))
                        TB_WRITER.add_scalar(
                            f"mse/D{dset_id}", loss_val, tot_step)
                        pb.set_postfix_str(losses_to_postfix(losses), refresh=False)
                        pb.update(1)
                        tot_step += 1
                    except StopIteration:
                        # Reset the training-iterator which has no more samples
                        tr_iterators[dset_id] = iter(tr_loaders[dset_id])
            if lr_sched is not None:
                lr_sched.step()
                TB_WRITER.add_scalar("lr", lr_sched.get_last_lr()[0], tot_step)  # one lr per parameter-group
        pb.close()
        # Save and evaluate model
        time_s = time.time()
        renderer.eval()
        with torch.autograd.no_grad():
            for ts_dset_id, ts_dset in enumerate(ts_dsets):
                for image_id in [0, 3, 6, 9]:
                    psnr = plot_ts(
                        ts_dset, ts_dset_id, renderer, log_dir, render_fn=default_render_fn(renderer, ts_dset_id),
                        iteration=tot_step, batch_size=batch_size, image_id=image_id, verbose=True,
                        summary_writer=TB_WRITER, plot_type="imageio")
                TB_WRITER.add_scalar(f"TestPSNR/D{ts_dset_id}", psnr, tot_step)
            torch.save({
                'epoch': e,
                'tot_step': tot_step,
                'scheduler': lr_sched.state_dict() if lr_sched is not None else None,
                'optimizer': optim.state_dict(),
                'model': renderer.state_dict(),
            }, os.path.join(log_dir, "model.pt"))
            print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")


def init_model(cfg, tr_dsets, checkpoint_data=None):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    radii = [dset.radius for dset in tr_dsets]
    if cfg.model.type == "lowrank_learnable_hash":
        voxel_size = (tr_dsets[0].radius * 2) / cfg.model.resolution
        # step-size and n-intersections are scaled to artificially increment resolution of model
        step_size = voxel_size / cfg.optim.samples_per_voxel
        n_intersections=int(math.sqrt(3.) * cfg.optim.samples_per_voxel * cfg.model.resolution)
        n_intersections = 256
        print("n_intersections: ", n_intersections)
        renderer = LowrankLearnableHash(
            num_scenes=len(tr_dsets),
            grid_config=GRID_CONFIG, radius=tr_dsets[0].radius, n_intersections=n_intersections)
    elif cfg.model.type == "learnable_hash":
        voxel_size = (tr_dsets[0].radius * 2) / cfg.model.resolution
        # step-size and n-intersections are scaled to artificially increment resolution of model
        step_size = voxel_size / cfg.optim.samples_per_voxel
        n_intersections = math.sqrt(3.) * cfg.optim.samples_per_voxel * cfg.model.resolution
        renderer = LearnableHash(
            resolution=cfg.model.resolution, num_features=cfg.model.num_features,
            feature_dim=cfg.model.feature_dim, radius=tr_dsets[0].radius,
            n_intersections=n_intersections, step_size=step_size, second_G=cfg.model.second_G,
            grid_dim=cfg.model.grid_dim, num_scenes=len(tr_dsets))
    elif cfg.model.type == "multi_reso":
        renderer = DictPlenoxels(
            sh_deg=cfg.sh.degree, sh_encoder=sh_encoder,
            radius=radii, num_atoms=cfg.model.num_atoms, num_scenes=len(tr_dsets),
            fine_reso=cfg.model.fine_reso, coarse_reso=cfg.model.coarse_reso,
            efficient_dict=cfg.model.efficient_dict, noise_std=cfg.model.noise_std, use_csrc=cfg.use_csrc)
    elif cfg.model.type == "single_reso":
        renderer = SingleResoDictPlenoxels(
            sh_deg=cfg.sh.degree, sh_encoder=sh_encoder, radius=radii,
            num_atoms=cfg.model.num_atoms[0], num_scenes=len(tr_dsets),
            fine_reso=cfg.model.fine_reso[0], coarse_reso=cfg.model.coarse_reso,
            dict_only_sigma=False)
    elif cfg.model_type == "super_reso":
        renderer = SuperResoPlenoxel(
            coarse_reso=cfg.model.coarse_reso, reso_multiplier=4, param_dim=4,
            radius=radii, num_scenes=len(tr_dsets), sh_deg=cfg.sh.degree, sh_encoder=sh_encoder)
    else:
        raise ValueError(f"Model type {cfg.model.type} invalid")
    if checkpoint_data is not None:
        renderer.load_state_dict(checkpoint_data['model'])
        print("=> Loaded model state from checkpoint")
    return renderer


# noinspection PyUnresolvedReferences,PyProtectedMember
def init_lr_scheduler(cfg, optim, batches_per_epoch, checkpoint_data=None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    eta_min = 1e-4
    num_batches_per_dset = 10
    lr_sched = None
    if cfg_.optim.cosine:
        num_dsets = len(cfg.data.datadirs)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.optim.num_epochs * batches_per_epoch * num_dsets // num_batches_per_dset,
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
    #train_cfg, reload_cfg = parse_config(multiscene_config.get_cfg_defaults())
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
    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(cfg_)
    batches_per_epoch = len(tr_dsets_[0]) // cfg_.optim.batch_size
    print("Will do %d batches per epoch" % (batches_per_epoch))

    if reload_cfg is not None and train_cfg is None:
        # We're reloading an existing model for either training or testing.
        # Here cfg_ is reload_cfg.
        chosen_opt = user_ask_options(
            "Restart model training from checkpoint or evaluate model? ", "train", "test")
        checkpoint_data = torch.load(os.path.join(train_log_dir, "model.pt"), map_location='cpu')
        model_ = init_model(cfg_, tr_dsets=tr_dsets_, checkpoint_data=checkpoint_data)
        if chosen_opt == "test":
            print("Running tests only.")
            for ts_dset_id, ts_dset in enumerate(ts_dsets_):
                print(f"Testing dataset {ts_dset_id}.")
                test_model(model_, ts_dset, train_log_dir, reload_cfg.optim.batch_size,
                           render_fn=default_render_fn(model_, ts_dset_id), plot_type="imageio")
        else:
            print(f"Resuming training from epoch {checkpoint_data['epoch'] + 1}")
            optim_ = init_optim(cfg_, model_, checkpoint_data=checkpoint_data)
            sched_ = init_lr_scheduler(cfg_, optim_, batches_per_epoch=batches_per_epoch, checkpoint_data=checkpoint_data)
            train_epoch(
                renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                batches_per_epoch=batches_per_epoch, epochs=cfg_.optim.num_epochs,
                log_dir=train_log_dir, batch_size=cfg_.optim.batch_size, lr_sched=sched_,
                start_epoch=checkpoint_data['epoch'] + 1, start_tot_step=checkpoint_data['tot_step'] + 1,
                train_fp16=cfg_.optim.train_f16)
    elif reload_cfg is not None:
        # We're doing a transfer learning experiment. Loading a pretrained model, and fine-tuning on
        # new data.
        print("Applying pretrained patch dicts to new scenes")
        reload_log_dir = os.path.join(reload_cfg.logdir, reload_cfg.expname)
        checkpoint_data = torch.load(os.path.join(reload_log_dir, "model.pt"), map_location='cpu')

        # Pretrained model
        pretrained_model = init_model(
            reload_cfg, tr_dsets=tr_dsets_, checkpoint_data=checkpoint_data)
        # Initialize a new model, but use the trained patch dictionaries.
        fresh_model = init_model(
            train_cfg, tr_dsets=tr_dsets_, checkpoint_data=None)
        assert fresh_model.atoms.shape == pretrained_model.atoms.shape, "Can't transfer due to config mismatch."
        fresh_model.atoms = pretrained_model.atoms
        pretrained_model = fresh_model
        # Only optimize the coarse grids and don't reload any optimizer state.
        optim_ = init_optim(train_cfg, pretrained_model, transfer_learning=True, checkpoint_data=None)
        sched_ = init_lr_scheduler(train_cfg, optim_, batches_per_epoch=batches_per_epoch, checkpoint_data=None)
        train_epoch(renderer=pretrained_model, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                    batches_per_epoch=batches_per_epoch,
                    epochs=train_cfg.optim.num_epochs,
                    log_dir=train_log_dir, batch_size=train_cfg.optim.batch_size,
                    lr_sched=sched_, train_fp16=train_cfg.optim.train_f16)
    else:
        # Normal training.
        with open(os.path.join(train_log_dir, "config.yaml"), "w") as fh:
            fh.write(cfg_.dump())
        model_ = init_model(cfg_, tr_dsets=tr_dsets_)
        optim_ = init_optim(cfg_, model_)
        sched_ = init_lr_scheduler(cfg_, optim_, batches_per_epoch=batches_per_epoch)
        train_epoch(
            renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
            batches_per_epoch=batches_per_epoch, epochs=cfg_.optim.num_epochs,
            log_dir=train_log_dir, batch_size=cfg_.optim.batch_size, lr_sched=sched_,
            train_fp16=cfg_.optim.train_f16)
