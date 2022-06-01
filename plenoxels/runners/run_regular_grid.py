import os
from collections import defaultdict
import time

import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm

from plenoxels import multiscene_config
from plenoxels.ema import EMA
from plenoxels.models import DictPlenoxels
from plenoxels.tc_harmonics import plenoxel_sh_encoder

from plenoxels.runners.utils import *


def train_epoch(renderer, tr_loaders, ts_dsets, optim, l1_coef, tv_coef, batches_per_epoch, epochs, log_dir, batch_size):
    batches_per_dset = 10
    eta_min = 1e-5
    ema_weight = 0.3
    num_dsets = len(tr_loaders)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * batches_per_epoch * num_dsets // batches_per_dset, eta_min=eta_min)

    tr_iterators = [iter(dl) for dl in tr_loaders]
    renderer.cuda()

    for e in range(epochs):
        pb = tqdm(range(0, batches_per_epoch * num_dsets, batches_per_dset), desc=f"epoch {e + 1}")
        losses = [defaultdict(lambda: EMA(ema_weight)) for _ in range(num_dsets)]
        for _ in pb:
            for dset_id in range(num_dsets):
                for i in range(batches_per_dset):
                    try:
                        rays_o, rays_d, imgs = next(tr_iterators[dset_id])
                        imgs = imgs.cuda()
                        rays_o = rays_o.cuda()
                        rays_d = rays_d.cuda()
                        optim.zero_grad()
                        rgb_preds, alpha, depth = renderer(rays_o, rays_d, grid_id=dset_id)

                        diff_losses = dict(mse=F.mse_loss(rgb_preds, imgs))
                        if l1_coef > 0:
                            diff_losses["l1"] = l1_coef * torch.abs(renderer.grids[dset_id].data).mean()
                        if tv_coef > 0:
                            diff_losses["tv"] = tv_coef * renderer.tv_loss(dset_id)

                        loss = 0.0
                        for l in diff_losses.values(): loss = loss + l
                        loss.backward()
                        optim.step()

                        for loss_name, loss_val in diff_losses.items():
                            losses[dset_id][loss_name].update(loss_val.item())
                    except StopIteration:
                        # Reset the training-iterator which has no more samples
                        tr_iterators[dset_id] = iter(tr_loaders[dset_id])
            lr_sched.step()
            pb_postfix = {}
            for dset_id, loss_dict in enumerate(losses):
                for loss_name, loss_val in loss_dict.items():
                    pb_postfix[f"{loss_name}-D{dset_id}"] = loss_val
            pb.set_postfix(pb_postfix)
        # Save and evaluate model
        time_s = time.time()
        for ts_dset_id, ts_dset in enumerate(ts_dsets):
            plot_ts_imageio(
                ts_dset, ts_dset_id, renderer, log_dir,
                iteration=e, batch_size=batch_size)
        model_save_path = os.path.join(log_dir, "model.pt")
        torch.save(renderer.state_dict(), model_save_path)
        print(f"Plot test images & saved model to {log_dir} in {time.time() - time_s:.2f}s")


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
                batches_per_epoch=cfg_.optim.batches_per_epoch, epochs=cfg_.optim.num_epochs,
                log_dir=log_dir_, batch_size=cfg_.optim.batch_size,
                l1_coef=cfg_.optim.regularization.l1_weight, tv_coef=cfg_.optim.regularization.tv_weight)
