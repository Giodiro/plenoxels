import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.base_trainer import BaseTrainer
from plenoxels.runners.regularization import (
    VideoPlaneTV, TimeSmoothness, HistogramLoss,
    L1AppearancePlanes, DistortionLoss
)
from plenoxels.runners.utils import init_dloader_random


class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=bg_color,
                                     near_far=near_far)
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        super().train_step(data, **kwargs)
        data = self._move_data_to_device(data)

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(
                data['rays_o'], data['rays_d'], data['timestamps'], bg_color=data['bg_color'],
                near_far=data['near_fars'])
            # Reconstruction loss
            recon_loss = self.criterion(fwd_out['rgb'], data['imgs'])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                loss = loss + reg_loss
        # Update weights
        self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                recon_loss_val = recon_loss.item()
                self.loss_info[f"mse"].update(recon_loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                for r in self.regularizers:
                    r.report(self.loss_info)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            if len(out_depths) > 0:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                    out_depths
                )
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=None),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data):
        super().load_model(checkpoint_data)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankVideo:
        dset = self.test_dataset
        try:
            global_translation = dset.global_translation
        except AttributeError:
            global_translation = None
        try:
            global_scale = dset.global_scale
        except AttributeError:
            global_scale = None
        model = LowrankVideo(
            aabb=dset.scene_bbox,
            len_time=dset.len_time,
            is_ndc=dset.is_ndc,
            is_contracted=dset.is_contracted,
            lookup_time=dset.lookup_time,
            proposal_sampling=self.extra_args.get('histogram_loss_weight', 0.0) > 0.0,
            global_scale=global_scale,
            global_translation=global_translation,
            **kwargs)
        log.info(f"Initialized {model.__class__} model with "
                 f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
                 f"using ndc {model.is_ndc} and contraction {model.is_contracted}.")
        return model

    def get_regularizers(self, **kwargs):
        return [
            VideoPlaneTV(kwargs.get('plane_tv_weight', 0.0)),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0)),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            L1AppearancePlanes(kwargs.get('l1_appearance_planes_reg', 0.0)),
            DistortionLoss(kwargs.get('distortion_reg', 0.0))
        ]

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir, split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras'),
        max_tsteps=kwargs.get('max_train_tsteps') if keyframes else None,
        isg=isg, keyframes=keyframes, is_contracted=False, is_ndc=False
    )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, **kwargs):
    ts_dset = Video360Dataset(
        data_dir, split='test', downsample=1,
        max_cameras=kwargs.get('max_test_cameras'),
        max_tsteps=kwargs.get('max_test_tsteps'),
        is_contracted=False, is_ndc=False,
    )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    od.update(init_ts_data(data_dirs[0], **kwargs))
    return od
