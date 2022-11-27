import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, Optional, Sequence, Union
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from nerfacc import ContractionType, OccupancyGrid

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.base_trainer import BaseTrainer, NerfaccHelper, RenderResult
from plenoxels.runners.regularization import (
    VideoPlaneTV, TimeSmoothness, L1PlaneDensityVideo,
    L1AppearancePlanes, Regularizer
)
from plenoxels.runners.utils import init_dloader_random


class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: Video360Dataset,
                 ts_dset: Video360Dataset,
                 num_steps: int,
                 scheduler_type: Optional[str],
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device,
                 sample_batch_size: int,
                 n_samples: int,
                 ist_step: int,
                 isg_step: int,
                 batch_size_queue,
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        if self.test_dataset.is_contracted:
            self.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        else:
            self.contraction_type = ContractionType.AABB
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        self.batch_size_queue = batch_size_queue

        self.nerfacc_helper = NerfaccHelper(
            target_sample_batch_size=sample_batch_size,
            render_n_samples=n_samples,
            cone_angle=kwargs['cone_angle'],
            density_threshold=kwargs['density_threshold'],
            alpha_threshold=kwargs['alpha_threshold'],
        )

        super().__init__(train_data_loader=tr_loader,
                         num_steps=num_steps,
                         scheduler_type=scheduler_type,
                         optim_type=optim_type,
                         logdir=logdir,
                         expname=expname,
                         train_fp16=train_fp16,
                         save_every=save_every,
                         valid_every=valid_every,
                         save_outputs=False,  # False since we're saving video
                         device=device,
                         **kwargs)
        # self.criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.occupancy_grid = self.init_occupancy_grid(**self.extra_args)
        self.occupancy_grid.to(device=self.device)

    def eval_step(self,
                  data: Dict[str, Union[int, torch.Tensor]],
                  **kwargs) -> RenderResult:
        super().eval_step(data, **kwargs)
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            return self.nerfacc_helper.render(
                self.model,
                self.occupancy_grid,
                data,
                self.device,
                step_size=self.model.step_size(self.nerfacc_helper.render_n_samples),
                is_training=False)

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        super().train_step(data, **kwargs)
        imgs = data["imgs"].to(self.device)
        data["timestamps"] = data["timestamps"].to(self.device)
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # update occupancy grid
            self.occupancy_grid.every_n_step(
                step=self.global_step,
                occ_eval_fn=lambda x: self.model.query_opacity(
                    x, data["timestamps"], self.train_dataset
                ),
                occ_thre=self.nerfacc_helper.density_threshold,
            )
            # render
            rendered = self.nerfacc_helper.render(
                self.model,
                self.occupancy_grid,
                data,
                self.device,
                step_size=self.model.step_size(self.nerfacc_helper.render_n_samples),
                is_training=True)
            if rendered.n_rendering_samples == 0:
                self.loss_info[f"n_rendered_samples"].update(0.0)
                return False

            # dynamic batch size for rays to keep sample batch size constant.
            new_batch_size = self.nerfacc_helper.calc_batch_size(
                old_batch_size=len(imgs), n_rendered_samples=rendered.n_rendering_samples
            )
            self.batch_size_queue.put(new_batch_size, block=False)
            self.train_dataset.update_num_rays(new_batch_size)
            alive_ray_mask = rendered.acc.squeeze(-1) > 0
            # compute loss and add regularizers
            recon_loss = self.criterion(rendered.rgb[alive_ray_mask], imgs[alive_ray_mask])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, grid_id=0)
                loss = loss + reg_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        # Update weights
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                mse = F.mse_loss(rendered.rgb, imgs).item()
                self.loss_info["psnr"].update(-10 * math.log10(mse))
                self.loss_info["mse"].update(mse)
                self.loss_info["alive_ray_mask"].update(float(alive_ray_mask.long().sum().item()))
                self.loss_info["n_rendering_samples"].update(float(rendered.n_rendering_samples))
                self.loss_info["n_rays"].update(float(len(imgs)))
                for r in self.regularizers:
                    r.report(self.loss_info)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration

        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        self.nerfacc_helper.step_cb(self.global_step)
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()
        self.nerfacc_helper.step_cb(self.global_step)

    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data, dset_id=0)
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
        base_save_dict["occupancy_grid"] = self.occupancy_grid.state_dict()
        return base_save_dict

    def load_model(self, checkpoint_data):
        super().load_model(checkpoint_data)
        self.occupancy_grid.load_state_dict(checkpoint_data["occupancy_grid"])
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
        model = LowrankVideo(
            aabb=dset.scene_bbox,
            len_time=dset.len_time,
            render_n_samples=self.nerfacc_helper.render_n_samples,
            grid_config=kwargs.pop("grid_config"),
            global_scale=None,
            global_translation=None,
            **kwargs)
        log.info(f"Initialized LowrankVideo model with "
                 f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        return model

    def init_occupancy_grid(self, **kwargs) -> OccupancyGrid:
        og_resolution = torch.tensor(kwargs.get('occupancy_grid_resolution'), dtype=torch.long)
        og = OccupancyGrid(
            roi_aabb=self.model.aabb().view(-1),
            resolution=og_resolution,
            contraction_type=self.contraction_type,
        )
        log.info("Initialized OccupancyGrid. resolution: %s - #parameters: %d" % (
            og.resolution.tolist(), sum(np.prod(p.shape) for p in og.parameters()),
        ))
        return og

    def get_regularizers(self, **kwargs) -> Sequence[Regularizer]:
        return (
            VideoPlaneTV(kwargs.get('plane_tv_weight', 0.0)),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0)),
            L1PlaneDensityVideo(kwargs.get('l1_plane_density_reg', 0.0)),
            L1AppearancePlanes(kwargs.get('l1_appearance_planes_reg', 0.0)),
        )

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    tr_queue = mp.Queue(maxsize=1000)
    if "lego" in data_dir:
        log.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            batch_size=batch_size,
            max_cameras=kwargs.get('max_train_cameras'),
            max_tsteps=kwargs.get('max_train_tsteps') if keyframes else None,
            isg=isg, keyframes=keyframes, is_contracted=False, is_ndc=False,
            batch_size_queue=tr_queue
        )
        if ist:
            tr_dset.switch_isg2ist()  # this should only happen in case we're reloading
    elif "sacre" in data_dir or "trevi" in data_dir or "brandenburg" in data_dir:
        tr_dset = PhotoTourismDataset(
            data_dir, split='train', downsample=data_downsample, batch_size=batch_size,
        )
        tr_loader = torch.utils.data.DataLoader(
            tr_dset, batch_size=batch_size, num_workers=4,
            prefetch_factor=4, pin_memory=True, shuffle=True,)
        return {"tr_loader": tr_loader, "tr_dset": tr_dset}
    else:
        log.info(f"Loading contracted Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample, batch_size=batch_size,
            keyframes=keyframes, isg=isg, is_contracted=True, is_ndc=False
        )
        if ist:
            tr_dset.switch_isg2ist()  # this should only happen in case we're reloading
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=2,
        prefetch_factor=4, pin_memory=True, worker_init_fn=init_dloader_random)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset, "batch_size_queue": tr_queue}


def init_ts_data(data_dir, **kwargs):
    if "lego" in data_dir:
        ts_dset = Video360Dataset(
            data_dir, split='test', downsample=1,
            max_cameras=kwargs.get('max_test_cameras'),
            max_tsteps=kwargs.get('max_test_tsteps'),
            is_contracted=False, is_ndc=False,
        )
    elif "sacre" in data_dir or "trevi" in data_dir or "brandenburg" in data_dir:
        ts_dset = PhotoTourismDataset(
            data_dir, split='test', downsample=1,
        )
    else:
        ts_dset = Video360Dataset(
            data_dir, split='test', downsample=2, keyframes=kwargs.get('keyframes', False),
            is_contracted=True
        )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, **kwargs):
    assert len(data_dirs) == 1
    od = {}
    if not validate_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    od.update(init_ts_data(data_dirs[0], **kwargs))
    return od
