import os
import time

import torch
import torch.utils.data
import torch.nn.functional as F

from plenoxels import multiscene_config
from plenoxels.ema import EMA
from plenoxels.models import DictPlenoxels
from plenoxels.tc_harmonics import plenoxel_sh_encoder

from plenoxels.runners.utils import *


def train_epoch(renderer, tr_loaders, ts_dsets, optim, l1_loss_coef, max_steps, log_dir, batch_size):
    batches_per_dset = 10
    max_tot_steps = max_steps * len(tr_loaders)
    eta_min = 1e-5
    ema_weight = 0.3

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps // batches_per_dset, eta_min=eta_min)

    tr_iterators = [iter(dl) for dl in tr_loaders]
    tot_steps = 0
    time_s = time.time()
    losses = [{"mse": EMA(ema_weight), "l1": EMA(ema_weight)} for _ in range(len(tr_loaders))]
    renderer.cuda()
    while len([it for it in tr_iterators if it is not None]) > 0:
        for dset_id, dset in enumerate(tr_iterators):
            if dset is None:
                continue
            try:
                for i in range(batches_per_dset):
                    if tot_steps >= max_tot_steps:
                        raise StopIteration("Maximum steps reached")
                    rays_o, rays_d, imgs = next(dset)
                    imgs = imgs.cuda()
                    rays_o = rays_o.cuda()
                    rays_d = rays_d.cuda()

                    optim.zero_grad()
                    rgb_preds, alpha, depth = renderer(rays_o, rays_d, grid_id=dset_id)
                    rec_loss = F.mse_loss(rgb_preds, imgs)
                    l1_loss = l1_loss_coef * torch.abs(renderer.grids[dset_id]).mean()
                    loss = rec_loss + l1_loss
                    loss.backward()
                    optim.step()
                    losses[dset_id]["mse"].update(rec_loss.item())
                    losses[dset_id]["l1"].update(l1_loss.item())
                    tot_steps += 1

                    if tot_steps % 100 == 0:
                        print(f"{tot_steps:6d}  ", end="")
                        for did, loss in enumerate(losses):
                            print(f"D{did}: ", end="")
                            for lname, lval in loss.items():
                                print(f"{lname}={lval.value:.4f} ", end="")
                            print("  ", end="")
                        print()
                    if tot_steps % 1000 == 0:
                        print(f"Time for 1000 steps: {time.time() - time_s:.2f}s")
                        time_s = time.time()
                        for ts_dset_id, ts_dset in enumerate(ts_dsets):
                            plot_ts_imageio(
                                ts_dset, ts_dset_id, renderer, log_dir,
                                iteration=tot_steps, batch_size=batch_size)
                        model_save_path = os.path.join(log_dir, "model.pt")
                        torch.save(renderer.state_dict(), model_save_path)
                        print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")
                        time_s = time.time()
            except StopIteration:
                print(f"Dataset {dset_id} finished")
                tr_iterators[dset_id] = None
        lr_sched.step()


def init_model(cfg, tr_dsets, efficient_dict):
    sh_encoder = plenoxel_sh_encoder(cfg.sh.degree)
    render = DictPlenoxels(
        sh_deg=cfg.sh.degree, sh_encoder=sh_encoder,
        radius=1.3, num_atoms=cfg.model.num_atoms, num_scenes=len(tr_dsets),
        fine_reso=cfg.model.fine_reso, coarse_reso=cfg.model.coarse_reso,
        efficient_dict=efficient_dict)
    return render


def init_optim(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)


if __name__ == "__main__":
    cfg_ = multiscene_config.parse_config()
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f'gpu is {gpu}')

    log_dir_ = os.path.join(cfg_.logdir, cfg_.expname)
    os.makedirs(log_dir_, exist_ok=True)
    # Make config immutable
    cfg_.freeze()
    # Save configuration as yaml into logdir
    with open(os.path.join(log_dir_, "config.yaml"), "w+") as fh:
        fh.write(cfg_.dump())
    print(cfg_)

    tr_dsets_, tr_loaders_, ts_dsets_ = init_data(cfg_)
    model_ = init_model(cfg_, tr_dsets=tr_dsets_, efficient_dict=False)
    optim_ = init_optim(cfg_, model_)
    train_epoch(renderer=model_, tr_loaders=tr_loaders_, ts_dsets=ts_dsets_, optim=optim_,
                max_steps=cfg_.optim.max_steps, log_dir=log_dir_,
                batch_size=cfg_.optim.batch_size, l1_loss_coef=cfg_.optim.regularization.l1_weight)
