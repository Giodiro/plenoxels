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
from ..core import RenderBuffer


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
        self.criterion = torch.nn.MSELoss(reduction='none')
        os.makedirs(self.valid_log_dir, exist_ok=True)

        self.loss_info = None
        self.test_datasets = None

    def init_test_datasets(self):
        if self.test_datasets is None:
            self.test_datasets = [
                dset.get_different_split(
                    'test', device='cpu', max_frames=self.extra_args.get("max_test_frames", None))
                for dset in self.datasets]
            for dset in self.test_datasets:
                dset.batch_size = -1

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = [defaultdict(lambda: EMA(ema_weight)) for _ in range(self.num_scenes)]

    def pre_epoch(self):
        super().pre_epoch()
        self.init_epoch_info()

    def get_training_lod(self):
        if self.extra_args["random_lod"]:
            # Sample from a geometric distribution
            population = [i for i in range(self.tracer.nef.num_lods)]
            weights = [2**i for i in range(self.tracer.nef.num_lods)]
            weights = [i/sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None
        return lod_idx

    def step(self, n_iter, data, scene_idx):
        """Implement the optimization over image-space loss.
        """
        self.optimizer.zero_grad(set_to_none=True)
        lod_idx = self.get_training_lod()

        with torch.cuda.amp.autocast():
            rays_o = data['rays_o'][0]
            rays_d = data['rays_d'][0]
            imgs = data['images'][0]

            N, C = imgs.shape

            if C == 3:
                bg_color = 1
            else:
                bg_color = torch.rand_like(imgs[..., :3])
            if C == 4:
                imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1 - imgs[..., 3:])

            rb = self.tracer(rays_o=rays_o,
                             rays_d=rays_d,
                             bg_color=bg_color,
                             lod_idx=lod_idx,
                             scene_idx=scene_idx,
                             perturb=True,
                             num_steps=self.extra_args["num_steps"])
            loss = self.criterion(rb.rgb, imgs).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()

        self.loss_info[scene_idx]["rgb_loss"].update(loss.item())
        mse = F.mse_loss(rb.rgb.detach(), imgs)
        self.loss_info[scene_idx]["mse"].update(mse.item())
        self.loss_info[scene_idx]["psnr"].update(10 * math.log10(1.0 / mse))

        return scale <= self.scaler.get_scale()

    def eval_step(self, data, lod_idx, scene_idx):
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast():
            rays_o = data['rays_o'][0]
            rays_d = data['rays_d'][0]

            rb = RenderBuffer(xyz=None, hit=None, normal=None, shadow=None, dirs=None)
            for b in range(math.ceil(rays_o.shape[0] / self.batch_size)):
                rays_o_b = rays_o[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rays_d_b = rays_d[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rb += self.tracer(rays_o=rays_o_b,
                                 rays_d=rays_d_b,
                                 bg_color=None,
                                 lod_idx=lod_idx,
                                 scene_idx=scene_idx,
                                 perturb=False,
                                 num_steps=self.extra_args["num_steps"])
        return rb


    def post_step(self, n_iter, pb):
        pfix_list = []
        for dset_id, loss_dict in enumerate(self.loss_info):
            pfix_inner = ", ".join(f"{lname}={lval.value:.2e}" for lname, lval in loss_dict.items())
            pfix_list.append(f"S{dset_id}({pfix_inner})")
        pfix_str = ' '.join(pfix_list)
        pb.set_postfix_str(pfix_str, refresh=False)
        pb.update(1)

    def post_epoch(self):
        super().post_epoch()

    def validate(self):
        log.info("Beginning validation...")
        self.init_test_datasets()
        val_metrics = []
        with torch.no_grad():
            for scene_idx, dataset in enumerate(self.test_datasets):
                per_scene_metrics = {
                    "psnr": 0,
                    "ssim": 0,
                    "scene_idx": scene_idx,
                    "lod_idx": "max",
                }
                for img_idx, data in enumerate(tqdm(dataset.dataloader(), desc=f"Test({scene_idx})")):
                    preds = self.eval_step(data, lod_idx=None, scene_idx=scene_idx)
                    if "images" in data:
                        gt = data["images"][0]
                    else:
                        gt = None
                    metrics = self.evaluate_metrics(gt, preds, scene_idx=scene_idx,
                                                    img_idx=img_idx, name="")
                    per_scene_metrics["psnr"] += metrics["psnr"]
                    per_scene_metrics["ssim"] += metrics["ssim"]
                per_scene_metrics["psnr"] /= len(dataset)
                per_scene_metrics["ssim"] /= len(dataset)
                log_text = f"EPOCH {self.epoch}/{self.num_epochs} | scene {scene_idx} lod max"
                log_text += f" | D{scene_idx} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{scene_idx} SSIM: {per_scene_metrics['ssim']:.6f}"
                log.info(log_text)
                val_metrics.append(per_scene_metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.valid_log_dir, f"test_metrics_epoch{self.epoch}.csv"))

    def evaluate_metrics(self, gt, preds: RenderBuffer, scene_idx, img_idx, name=None):
        preds.view = None
        preds.hit = None
        preds = preds.reshape(*gt.shape[:2], -1).cpu()
        preds.gts = gt[..., :3]
        preds.err = (preds.gts - preds.rgb) ** 2
        print("test mse: ", preds.err.mean())
        psnr_ = psnr(preds.rgb, preds.gts)
        ssim_ = ssim(preds.rgb, preds.gts)

        exrdict = preds.exr_dict()
        out_name = f"epoch{self.epoch}-D{scene_idx}-{img_idx}"
        if name is not None:
            out_name += "-" + name

        write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
        write_png(os.path.join(self.valid_log_dir, out_name + ".png"), preds.image().byte().rgb.numpy())

        return {"psnr": psnr_, "ssim": ssim_}
