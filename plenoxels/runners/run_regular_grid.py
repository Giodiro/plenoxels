from typing import Dict, List
import os
from collections import defaultdict
import time

import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels import multiscene_config
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


def train_epoch(renderer, tr_loaders, ts_dsets, optim, l1_coef, tv_coef, consistency_coef, batches_per_epoch, epochs, log_dir, batch_size, cosine):
    batches_per_dset = 10
    eta_min = 1e-4
    ema_weight = 0.3
    num_dsets = len(tr_loaders)

    lr_sched = None
    if cosine:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs * batches_per_epoch * num_dsets // batches_per_dset, eta_min=eta_min)
    TB_WRITER.add_scalar("lr", optim.param_groups[0]["lr"], 0)

    tr_iterators = [iter(dl) for dl in tr_loaders]
    renderer.cuda()

    tot_step = 0
    for e in range(epochs):
        # pb = tqdm(range(0, batches_per_epoch * num_dsets, batches_per_dset), desc=f"epoch {e + 1}")
        losses = [defaultdict(lambda: EMA(ema_weight)) for _ in range(num_dsets)]
        pb = tqdm(total=batches_per_epoch * num_dsets, desc=f"Epoch {e + 1}")
        for _ in range(0, batches_per_epoch, batches_per_dset):
            for dset_id in range(num_dsets):
                for i in range(batches_per_dset):
                    try:
                        rays_o, rays_d, imgs = next(tr_iterators[dset_id])
                        imgs = imgs.cuda()
                        rays_o = rays_o.cuda()
                        rays_d = rays_d.cuda()
                        optim.zero_grad()
                        rgb_preds, alpha, depth, consistency_loss = renderer(rays_o, rays_d, grid_id=dset_id)

                        diff_losses = dict(mse=F.mse_loss(rgb_preds, imgs))
                        if l1_coef > 0:
                            diff_losses["l1"] = l1_coef * torch.abs(renderer.grids[dset_id]).mean()
                        if tv_coef > 0:
                            diff_losses["tv"] = tv_coef * renderer.tv_loss(dset_id)
                        if consistency_coef > 0:
                            diff_losses["consistency"] = consistency_coef * consistency_loss
                        loss = sum(diff_losses.values())
                        loss.backward()
                        optim.step()

                        for loss_name, loss_val in diff_losses.items():
                            losses[dset_id][loss_name].update(loss_val.item())
                            TB_WRITER.add_scalar(f"{loss_name}/D{dset_id}", loss_val.item(), tot_step)
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
        for ts_dset_id, ts_dset in enumerate(ts_dsets):
            psnr = plot_ts_imageio(
                ts_dset, ts_dset_id, renderer, log_dir,
                iteration=e, batch_size=batch_size, image_id=0, verbose=True)
            TB_WRITER.add_scalar(f"TestPSNR/D{ts_dset_id}", psnr, tot_step)
        model_save_path = os.path.join(log_dir, "model.pt")
        torch.save(renderer.state_dict(), model_save_path)
        print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")


def test_model(renderer, ts_dsets, log_dir, batch_size, num_test_imgs=1):
    renderer.cuda()
    for ts_dset_id, ts_dset in enumerate(ts_dsets):
        psnrs = []
        for image_id in range(num_test_imgs):
            psnr = plot_ts_imageio(ts_dset, ts_dset_id, renderer, log_dir, iteration="test",
                                   batch_size=batch_size, image_id=image_id, verbose=False)
            psnrs.append(psnr)
        print(f"Average PSNR (over {num_test_imgs} poses) for "
              f"dataset {ts_dset_id}: {np.mean(psnrs):.2f}")


def load_model_from_logdir(cfg, logdir, tr_dsets, efficient_dict) -> DictPlenoxels:
    renderer = init_model(cfg, tr_dsets, efficient_dict)
    model_save_path = os.path.join(logdir, "model.pt")
    model_data = torch.load(model_save_path, map_location='cpu')
    renderer.load_state_dict(model_data)
    print(f"loaded model from {model_save_path}")
    return renderer


def init_model(cfg, tr_dsets, efficient_dict):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    radii = [dset.radius for dset in tr_dsets]
    render = DictPlenoxels(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder,
        radius=radii, num_atoms=cfg.model.num_atoms, num_scenes=len(tr_dsets),
        fine_reso=cfg.model.fine_reso, coarse_reso=cfg.model.coarse_reso,
        efficient_dict=efficient_dict, noise_std=cfg.model.noise_std, use_csrc=cfg.use_csrc)
    return render


def init_optim(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)


if __name__ == "__main__":
    cfg_, run_test_ = multiscene_config.parse_config()
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f'gpu is {gpu}')

    log_dir_ = os.path.join(cfg_.logdir, cfg_.expname)
    os.makedirs(log_dir_, exist_ok=True)
    # Make config immutable
    cfg_.freeze()
    # Initialize tensorboard
    TB_WRITER = SummaryWriter(log_dir=log_dir_)
    print(cfg_)
    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(cfg_)
    if run_test_:
        print("Running tests only.")
        model_ = load_model_from_logdir(cfg_, logdir=log_dir_, tr_dsets=tr_dsets_, efficient_dict=False)
        test_model(renderer=model_, ts_dsets=ts_dsets_, log_dir=log_dir_,
                   batch_size=cfg_.optim.batch_size, num_test_imgs=1)
    else:
        # Save configuration as yaml into logdir
        with open(os.path.join(log_dir_, "config.yaml"), "w") as fh:
            fh.write(cfg_.dump())
        model_ = init_model(cfg_, tr_dsets=tr_dsets_, efficient_dict=False)
        optim_ = init_optim(cfg_, model_)
        train_epoch(renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                    batches_per_epoch=cfg_.optim.batches_per_epoch, epochs=cfg_.optim.num_epochs,
                    log_dir=log_dir_, batch_size=cfg_.optim.batch_size,
                    l1_coef=cfg_.optim.regularization.l1_weight, tv_coef=cfg_.optim.regularization.tv_weight,
                    consistency_coef=cfg_.optim.regularization.consistency_weight, cosine=cfg_.optim.cosine)
