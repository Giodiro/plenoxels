import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union, Sequence, Any
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from nerfacc import ContractionType, OccupancyGrid

from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from .base_trainer import BaseTrainer, RenderResult, NerfaccHelper
from .regularization import (
    PlaneTV, DensityPlaneTV, VolumeTV, L1PlaneColor, L1PlaneDensity,
    L1Density,
)
from .utils import init_dloader_random
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..datasets.base_dataset import BaseDataset
from ..datasets.multi_dataset_sampler import MultiSceneSampler
from ..my_tqdm import tqdm
from ..utils import parse_optint


class MultisceneTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dsets: List[BaseDataset],
                 tr_dsets: List[BaseDataset],
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
                 batch_size_queue,
                 **kwargs
                 ):
        self.test_datasets = ts_dsets
        self.train_datasets = tr_dsets
        if self.test_datasets[0].is_contracted:
            self.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        else:
            self.contraction_type = ContractionType.AABB
        self.num_dsets = len(self.train_datasets)
        self.batch_size_queue = batch_size_queue
        self.num_trloader_workers = tr_loader.num_workers

        self.nerfacc_helper = NerfaccHelper(
            target_sample_batch_size=sample_batch_size,
            render_n_samples=n_samples,
            cone_angle=kwargs['cone_angle'],
            density_threshold=kwargs['density_threshold'],
            alpha_threshold=kwargs['alpha_threshold'],
        )

        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            scheduler_type=scheduler_type,
            optim_type=optim_type,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=save_outputs,
            device=device,
            **kwargs
        )

        # self.criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.occupancy_grids = self.init_occupancy_grid(**self.extra_args)
        for og in self.occupancy_grids:
            og.to(device=self.device)

    def eval_step(self,
                  data: Dict[str, Union[int, torch.Tensor]],
                  **kwargs) -> RenderResult:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        dset_id = data["dset_id"]
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            return self.nerfacc_helper.render(
                self.model,
                self.occupancy_grids[dset_id],
                data,
                self.device,
                step_size=self.model.step_size(self.nerfacc_helper.render_n_samples, grid_id=dset_id),
                is_training=False)

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        super().train_step(data, **kwargs)
        dset_id = data["dset_id"]
        imgs = data["imgs"].to(self.device)
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # update occupancy grid
            self.occupancy_grids[dset_id].every_n_step(
                step=self.global_step,
                occ_eval_fn=lambda x: self.model.query_opacity(
                    x, dset_id, self.train_datasets[dset_id]
                ),
                occ_thre=self.nerfacc_helper.density_threshold,
            )
            # render
            rendered = self.nerfacc_helper.render(
                self.model,
                self.occupancy_grids[dset_id],
                data,
                self.device,
                step_size=self.model.step_size(self.nerfacc_helper.render_n_samples, grid_id=dset_id),
                is_training=True)
            if rendered.n_rendering_samples == 0:
                self.loss_info[f"n_rendered_samples_{dset_id}"].update(0.0)
                return False

            # dynamic batch size for rays to keep sample batch size constant.
            new_batch_size = self.nerfacc_helper.calc_batch_size(
                old_batch_size=len(imgs), n_rendered_samples=rendered.n_rendering_samples
            )
            self.batch_size_queue.put(new_batch_size, block=False)
            self.train_datasets[dset_id].update_num_rays(new_batch_size)
            alive_ray_mask = rendered.acc.squeeze(-1) > 0
            # compute loss and add regularizers
            recon_loss = self.criterion(rendered.rgb[alive_ray_mask], imgs[alive_ray_mask])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, grid_id=dset_id)
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
                self.loss_info[f"psnr_{dset_id}"].update(-10 * math.log10(mse))
                self.loss_info[f"mse_{dset_id}"].update(mse)
                self.loss_info[f"alive_ray_mask_{dset_id}"].update(
                    float(alive_ray_mask.long().sum().item()))
                self.loss_info[f"n_rendered_samples_{dset_id}"].update(float(rendered.n_rendering_samples))
                self.loss_info[f"n_rays_{dset_id}"].update(float(len(imgs)))
                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        self.nerfacc_helper.step_cb(self.global_step)
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in every train-dataset
        for d in self.train_datasets:
            d.reset_iter()
        self.nerfacc_helper.step_cb(self.global_step)

    @torch.no_grad()
    def validate(self):
        val_metrics = []
        for dset_id, dataset in enumerate(self.test_datasets):
            pb = tqdm(total=len(dataset), desc=f"Test scene {dset_id} ({dataset.name})")
            per_scene_metrics = defaultdict(list)
            for img_idx, data in enumerate(dataset):
                ts_render = self.eval_step(data, dset_id=dset_id)
                out_metrics, _, _ = self.evaluate_metrics(
                    data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
                    name=f"D{dset_id}", save_outputs=self.save_outputs)
                for k, v in out_metrics.items():
                    per_scene_metrics[k].append(v)
                pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                pb.update(1)
            pb.close()
            val_metrics.append(
                self.report_test_metrics(per_scene_metrics, extra_name=f"scene_{dset_id}"))

        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        base_save_dict["occupancy_grids"] = [og.state_dict() for og in self.occupancy_grids]
        return base_save_dict

    def load_model(self, checkpoint_data):
        super().load_model(checkpoint_data)
        for i in range(len(self.occupancy_grids)):
            self.occupancy_grids[i].load_state_dict(checkpoint_data["occupancy_grids"][i])

    def init_epoch_info(self):
        ema_weight = 0.9  # higher places higher weight to new observations
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankLearnableHash:
        aabbs = [d.scene_bbox for d in self.train_datasets]
        model = LowrankLearnableHash(
            num_scenes=self.num_dsets,
            grid_config=kwargs.pop("grid_config"),
            aabb=aabbs,
            render_n_samples=self.nerfacc_helper.render_n_samples,
            **kwargs)
        log.info(f"Initialized LowrankLearnableHash model with "
                 f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        return model

    def init_occupancy_grid(self, **kwargs) -> List[OccupancyGrid]:
        og_resolution = torch.tensor(kwargs.get('occupancy_grid_resolution'), dtype=torch.long)
        occupancy_grids = []
        for scene in range(self.num_dsets):
            og = OccupancyGrid(
                roi_aabb=self.model.aabb(scene).view(-1),
                resolution=og_resolution,
                contraction_type=self.contraction_type,
            )
            occupancy_grids.append(og)
            log.info("Initialized OccupancyGrid(dset=%d). resolution: %s - #parameters: %d" % (
                scene, og.resolution.tolist(), sum(np.prod(p.shape) for p in og.parameters()),
            ))
        return occupancy_grids

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), features='all'),
            PlaneTV(kwargs.get('plane_tv_weight_sigma', 0.0), features='sigma'),
            PlaneTV(kwargs.get('plane_tv_weight_sh', 0.0), features='sh'),
            DensityPlaneTV(kwargs.get('density_plane_tv_weight', 0.0)),
            VolumeTV(
                kwargs.get('volume_tv_weight', 0.0),
                what=kwargs.get('volume_tv_what'),
                patch_size=kwargs.get('volume_tv_patch_size', 3),
                batch_size=kwargs.get('volume_tv_npts', 100),
            ),
            L1PlaneColor(kwargs.get('l1_plane_color_weight', 0.0)),
            L1PlaneDensity(kwargs.get('l1_plane_density_weight', 0.0)),
            L1Density(kwargs.get('l1density_weight', 0.0), max_voxels=100_000),
        ]

    @property
    def calc_metrics_every(self):
        return self.num_dsets * 2 + 1


def decide_dset_type(dd: str) -> str:
    if ("chair" in dd or "drums" in dd or "ficus" in dd or "hotdog" in dd or "lego" in dd or
            "materials" in dd or "mic" in dd or "ship" in dd):
        return "synthetic"
    elif ("fern" in dd or "flower" in dd or "fortress" in dd or "horns" in dd or "leaves" in dd or
          "orchids" in dd or "room" in dd or "trex" in dd):
        return "llff"
    else:
        raise RuntimeError(f"data_dir {dd} not recognized as LLFF or Synthetic dataset.")


def init_tr_data(data_downsample: float, data_dirs: Sequence[str], **kwargs):
    initial_batch_size = int(kwargs['sample_batch_size']) // int(kwargs['n_samples'])
    dsets = []
    tr_queue = mp.Queue(maxsize=1000)

    for i, data_dir in enumerate(data_dirs):
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample,
                max_frames=max_tr_frames, batch_size=initial_batch_size, dset_id=i,
                batch_size_queue=tr_queue))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            log.info(f"About to load LLFF data downsampled by {data_downsample} times.")
            dsets.append(LLFFDataset(
                data_dir, split='train', downsample=int(data_downsample), hold_every=hold_every,
                batch_size=initial_batch_size, dset_id=i))
        dsets[-1].reset_iter()

    tr_sampler = MultiSceneSampler(dsets, num_samples_per_dataset=1)
    cat_tr_dset = torch.utils.data.ConcatDataset(dsets)
    tr_loader = torch.utils.data.DataLoader(
        cat_tr_dset, num_workers=4, prefetch_factor=2, pin_memory=True,
        batch_size=None, sampler=tr_sampler, worker_init_fn=init_dloader_random)

    return {
        "tr_dsets": dsets,
        "tr_loader": tr_loader,
        "batch_size_queue": tr_queue,
    }


def init_ts_data(data_dirs: Sequence[str], **kwargs):
    dsets = []
    for i, data_dir in enumerate(data_dirs):
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
            dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, max_frames=max_ts_frames, dset_id=i))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            dsets.append(LLFFDataset(
                data_dir, split='test', downsample=4, hold_every=hold_every, dset_id=i))
    return {"ts_dsets": dsets}


def load_data(data_downsample, data_dirs, validate_only, **kwargs):
    od: Dict[str, Any] = {}
    if not validate_only:
        od.update(init_tr_data(data_downsample, data_dirs, **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    od.update(init_ts_data(data_dirs, **kwargs))
    return od
