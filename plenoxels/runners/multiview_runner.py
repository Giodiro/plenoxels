import math
import os
import random
import logging as log
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base_runner import BaseTrainer
from ..ema import EMA
from ..ops.image.metrics import psnr, ssim
from ..ops.image.io import write_png, write_exr
from ..core.rays import Rays


class MultiviewTrainer(BaseTrainer):
    def __init__(self, tracer, datasets, num_epochs, optim_cls, lr_scheduler_type, lr, weight_decay,
                 grid_lr_weight, optim_params, log_dir, device, num_batches_per_scene=10,
                 exp_name=None, render_every=-1, save_every=-1, **kwargs):
        super().__init__(tracer=tracer, datasets=datasets, num_epochs=num_epochs,
                         optim_cls=optim_cls, lr_scheduler_type=lr_scheduler_type, lr=lr,
                         weight_decay=weight_decay, grid_lr_weight=grid_lr_weight,
                         optim_params=optim_params, log_dir=log_dir, device=device,
                         num_batches_per_scene=num_batches_per_scene, exp_name=exp_name,
                         render_every=render_every, save_every=save_every, **kwargs)

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        os.makedirs(self.valid_log_dir, exist_ok=True)

        self.loss_info = None

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = [defaultdict(lambda: EMA(ema_weight)) for _ in range(self.num_scenes)]

    def pre_epoch(self):
        super().pre_epoch()
        self.init_epoch_info()

    def step(self, n_iter, data, scene_idx):
        """Implement the optimization over image-space loss.
        """
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        self.optimizer.zero_grad(set_to_none=True)

        if self.extra_args["random_lod"]:
            # Sample from a geometric distribution
            population = [i for i in range(self.tracer.nef.num_lods)]
            weights = [2**i for i in range(self.tracer.nef.num_lods)]
            weights = [i/sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None

        with torch.cuda.amp.autocast():
            rb = self.tracer(rays=rays,
                             lod_idx=lod_idx,
                             scene_idx=scene_idx,
                             num_steps=self.extra_args["num_steps"])
            # RGB Loss
            # TODO: This should either default to MSE or be a parameter.
            # rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
            loss = F.l1_loss(rb.rgb[..., :3], img_gts[..., :3], reduction='mean')

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()
        self.loss_info[scene_idx]["rgb_loss"].update(loss.item())
        mse = F.mse_loss(rb.rgb[..., :3].detach(), img_gts[..., :3])
        self.loss_info[scene_idx]["mse"].update(mse.item())
        self.loss_info[scene_idx]["psnr"].update(10 * math.log10(1.0 / mse))

        return scale <= self.scaler.get_scale()

    def post_step(self, n_iter, pb):
        pfix_list = []
        for dset_id, loss_dict in enumerate(self.loss_info):
            pfix_inner = ", ".join(f"{lname}={lval:.2e}" for lname, lval in loss_dict.items())
            pfix_list.append(f"S{dset_id}({pfix_inner})")
        pfix_str = ' '.join(pfix_list)
        pb.set_postfix_str(pfix_str, refresh=False)
        pb.update(1)

    def post_epoch(self):
        super().post_epoch()

    def validate(self, use_test_set=False):
        log.info("Beginning validation...")

        val_metrics = []
        lods = list(range(self.tracer.nef.grid.num_lods))[::-1]
        split_name = "test" if use_test_set else "val"
        for scene_idx, dataset in enumerate(self.datasets):
            if use_test_set:
                val_data = dataset.get_full_test_set()
            else:
                val_data = dataset.get_full_val_set()
            imgs = list(val_data["imgs"])
            img_shape = imgs[0].shape
            log.info(f"Loaded validation set for dataset {scene_idx} with {len(imgs)} images "
                     f"at resolution {img_shape[0]}x{img_shape[1]}")

            log.info(f"Saving validation result to {self.valid_log_dir}")

            for lod in lods:
                metrics = self.evaluate_metrics(
                    self.epoch, val_data["rays"], imgs, lod, scene_idx, name=f"{split_name}_lod{lod}")
                metrics['lod_idx'] = lod
                metrics['scene_idx'] = scene_idx
                val_metrics.append(metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.valid_log_dir, f"{split_name}_metrics_epoch{self.epoch}.csv"))

    def evaluate_metrics(self, epoch, rays, imgs, lod_idx, scene_idx, name=None):
        ray_os = list(rays.origins)
        ray_ds = list(rays.dirs)

        psnr_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for idx, (img, ray_o, ray_d) in tqdm(enumerate(zip(imgs, ray_os, ray_ds))):
                rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                rays = rays.reshape(-1, 3)
                rays = rays.to('cuda')
                rb = self.renderer.render(self.tracer, rays, lod_idx=lod_idx, scene_idx=scene_idx)
                rb.view = None
                rb.hit = None
                rb = rb.reshape(*img.shape[:2], -1).cpu()
                rb.gts = img
                rb.err = (rb.gts[..., :3] - rb.rgb[..., :3]) ** 2
                psnr_total += psnr(rb.rgb[..., :3], rb.gts[..., :3])
                ssim_total += ssim(rb.rgb[..., :3], rb.gts[..., :3])

                exrdict = rb.exr_dict()

                out_name = f"epoch{epoch}-D{scene_idx}-{idx}"
                if name is not None:
                    out_name += "-" + name

                write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
                write_png(os.path.join(self.valid_log_dir, out_name + ".png"), rb.image().byte().rgb.numpy())
        psnr_total /= len(imgs)
        ssim_total /= len(imgs)

        log_text = f"EPOCH {epoch}/{self.num_epochs} | scene {scene_idx} lod {lod_idx}"
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)
        log.info(log_text)

        return {"psnr": psnr_total, "ssim": ssim_total}
