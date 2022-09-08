import logging
import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.utils.data

from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from .multiscene_trainer import Trainer


class VideoTrainer(Trainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.Dataset,
                 num_epochs: int,
                 scheduler_type: Optional[str],
                 model_type: str,
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 **kwargs
                 ):
        # Keys we wish to ignore
        kwargs.pop('transfer_learning', None)
        kwargs.pop('num_batches_per_dset', None)
        super().__init__(tr_loaders=[tr_loader],
                         ts_dsets=[ts_dset],
                         num_batches_per_dset=1,
                         num_epochs=num_epochs,
                         scheduler_type=scheduler_type,
                         model_type=model_type,
                         optim_type=optim_type,
                         logdir=logdir,
                         expname=expname,
                         train_fp16=train_fp16,
                         save_every=save_every,
                         valid_every=valid_every,
                         transfer_learning=False,  # No transfer with video
                         **kwargs)

    def eval_step(self, data, dset_id) -> torch.Tensor:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data[0]
            rays_d = data[1]
            timestamp = data[3]
            # timestamp = 0
            preds = []
            for b in range(math.ceil(rays_o.shape[0] / self.batch_size)):
                rays_o_b = rays_o[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rays_d_b = rays_d[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                timestamps_d_b = torch.ones(len(rays_o_b)).cuda() * timestamp
                preds.append(self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=1))
            preds = torch.cat(preds, 0)
        return preds

    def step(self, data, dset_id):
        rays_o = data[0].cuda()
        rays_d = data[1].cuda()
        timestamps = data[3].cuda()
        imgs = data[2].cuda()
        self.optimizer.zero_grad(set_to_none=True)

        C = imgs.shape[-1]
        # Random bg-color
        if C == 3:
            bg_color = 1
        else:
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rgb_preds = self.model(rays_o, rays_d, timestamps, bg_color=bg_color)
            loss = self.criterion(rgb_preds, imgs)

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        loss_val = loss.item()
        self.loss_info["mse"].update(loss_val)
        self.loss_info["psnr"].update(-10 * math.log10(loss_val))
        return scale <= self.gscaler.get_scale()

    def post_step(self, dset_id, progress_bar):
        self.writer.add_scalar(f"mse: ", self.loss_info["mse"].value, self.global_step)
        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info), refresh=False)
        progress_bar.update(1)

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.train_data_loaders[0].dataset
        if self.model_type == "learnable_hash":
            model = LowrankVideo(
                aabb=dset.scene_bbox,
                len_time=dset.len_time,
                **kwargs)
        else:
            raise ValueError(f"Model type {self.model_type} invalid")
        logging.info(f"Initialized model of type {self.model_type} with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model


def losses_to_postfix(loss_dict: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())
