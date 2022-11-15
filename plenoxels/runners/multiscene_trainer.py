import logging
import math
import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, MutableMapping, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from nerfacc import ContractionType, OccupancyGrid
from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_exr, write_png
from .regularization import (
    PlaneTV, DensityPlaneTV, VolumeTV, L1PlaneColor, L1PlaneDensity,
    L1Density
)
from .utils import render_image
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..my_tqdm import tqdm
from ..utils import parse_optint
from .utils import get_cosine_schedule_with_warmup, get_step_schedule_with_warmup


class Trainer():
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dsets: List[torch.utils.data.TensorDataset],
                 tr_dsets: List[torch.utils.data.TensorDataset],
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
                 **kwargs
                 ):
        self.train_data_loader = tr_loader
        self.test_datasets = ts_dsets
        self.train_datasets = tr_dsets
        self.is_ndc = self.test_datasets[0].is_ndc

        self.extra_args = kwargs
        self.num_dsets = len(self.train_datasets)

        self.num_steps = num_steps

        self.scheduler_type = scheduler_type
        self.optim_type = optim_type
        self.transfer_learning = kwargs['transfer_learning']
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.gradient_acc = kwargs.get('gradient_acc', False)
        self.target_sample_batch_size = sample_batch_size
        self.render_n_samples = n_samples
        self.cone_angle = kwargs['cone_angle']
        self._density_threshold = kwargs['density_threshold']
        self._alpha_threshold = kwargs['alpha_threshold']

        # Set initial batch-size
        for dset in self.train_datasets:
            dset.update_num_rays(self.target_sample_batch_size // self.render_n_samples)
        # All 'steps' things must be multiplied by the number of datasets
        # to get consistent amount of training independently of how many dsets.
        # This is not needed if gradient_acc is set, since in that case the
        # global_step is only updated once every time we cycle through each scene
        step_multiplier = 1 if self.gradient_acc else self.num_dsets
        self.shrink_steps = [s * step_multiplier for s in kwargs.get('shrink_steps', [])]
        self.upsample_steps = [s * step_multiplier for s in kwargs.get('upsample_steps', [])]
        self.upsample_resolution_list = list(kwargs.get('upsample_resolution', []))
        assert len(self.upsample_resolution_list) == len(self.upsample_steps), \
            f"Got {len(self.upsample_steps)} upsample_steps and {len(self.upsample_resolution_list)} upsample_resolution."

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step = None
        self.loss_info = None
        self.train_iterators = None
        self.contraction_type = ContractionType.UN_BOUNDED_SPHERE
        self.is_unbounded = self.contraction_type != ContractionType.AABB

        # self.criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean')

        self.model = self.init_model(**self.extra_args)
        self.occupancy_grids = self.init_occupancy_grid(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)
        self.regularizers = self.init_regularizers(**self.extra_args)

        self.device = device
        self.model.to(device=self.device)
        if False:
            for dset in self.train_datasets:
                dset.to(device=self.device)
            for dset in self.test_datasets:
                dset.to(device=self.device)

    def eval_step(self, data, dset_id) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rgb, acc, depth, _ = render_image(
                self.model,
                self.occupancy_grids[dset_id],
                dset_id,
                data['rays_o'],
                data['rays_d'],
                aabb=self.model.aabb(dset_id).view(-1),
                near_plane=self.train_datasets[dset_id].near,
                far_plane=self.train_datasets[dset_id].far,
                #near_plane=None,
                #far_plane=None,
                render_bkgd=data['color_bkgd'],
                cone_angle=self.cone_angle,
                render_step_size=self.model.step_size(self.render_n_samples, dset_id),
                alpha_thresh=self.alpha_threshold,
                device=self.device,
            )
        return {
            "rgb": rgb,
            "depth": depth,
        }

    def step(self, data: Dict[str, Union[int, torch.Tensor]], do_update: bool):
        dset_id = data["dset_id"]
        imgs = data["pixels"]

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # update occupancy grid
            self.occupancy_grids[dset_id].every_n_step(
                step=self.global_step,
                occ_eval_fn=lambda x: self.model.query_opacity(
                    x, dset_id, self.train_datasets[dset_id]
                ),
                occ_thre=self.density_threshold,
            )
            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                self.model,
                self.occupancy_grids[dset_id],
                dset_id,
                data["rays_o"],
                data["rays_d"],
                # rendering options
                aabb=self.model.aabb(dset_id).view(-1),
                near_plane=self.train_datasets[dset_id].near,
                far_plane=self.train_datasets[dset_id].far,
                render_bkgd=data["color_bkgd"],
                cone_angle=self.cone_angle,
                render_step_size=self.model.step_size(self.render_n_samples, dset_id),
                alpha_thresh=self.alpha_threshold,
                device=self.device,
            )
            if n_rendering_samples == 0:
                return False
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(imgs)
            num_rays = min(self.max_rays, int(
                num_rays
                * (self.target_sample_batch_size / float(n_rendering_samples))
            ))
            self.train_datasets[dset_id].update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0
            # compute loss and add regularizers
            recon_loss = self.criterion(rgb[alive_ray_mask], imgs[alive_ray_mask])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, grid_id=dset_id)
                loss = loss + reg_loss

        if do_update:
            self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()

        # Report on losses
        if self.global_step % 10 == 0:
            with torch.no_grad():
                mse = F.mse_loss(rgb[alive_ray_mask], imgs[alive_ray_mask]).item()
                self.loss_info[dset_id]["psnr"].update(-10 * math.log10(mse))
                self.loss_info[dset_id]["mse"].update(mse)
                self.loss_info[dset_id]["alive_ray_mask"].update(float(alive_ray_mask.long().sum().item()))
                self.loss_info[dset_id]["n_rendering_samples"].update(float(n_rendering_samples))
                self.loss_info[dset_id]["n_rays"].update(float(len(imgs)))
                for r in self.regularizers:
                    r.report(self.loss_info[dset_id])

        # Update weights
        if do_update:
            self.gscaler.step(self.optimizer)
            scale = self.gscaler.get_scale()
            self.gscaler.update()

            # Run grid-update routines (masking, shrinking, upscaling)
            opt_reset_required = False
            if self.global_step in self.shrink_steps:
                print()
                for u_dset_id in range(self.num_dsets):
                    self.model.shrink(self.occupancy_grids[dset_id], dset_id)
                    opt_reset_required = True
            try:
                upsample_step_idx = self.upsample_steps.index(self.global_step)  # if not an upsample step will raise
                new_num_voxels = self.upsample_resolution_list[upsample_step_idx]
                print()
                logging.info(f"Upsampling all datasets at step {self.global_step} to {new_num_voxels} voxels.")
                for u_dset_id in range(self.num_dsets):
                    new_reso = N_to_reso(new_num_voxels, self.model.aabb(u_dset_id))
                    self.model.upsample(new_reso, u_dset_id)
                opt_reset_required = True
            except ValueError:
                pass

            # We reset the optimizer in case some of the parameters in model were changed.
            if opt_reset_required:
                self.optimizer = self.init_optim(**self.extra_args)

            return scale <= self.gscaler.get_scale()
        return True

    def post_step(self, data, progress_bar):
        dset_id = data["dset_id"]
        if self.global_step % 30 == 0:
            self.writer.add_scalar(
                f"mse/D{dset_id}", self.loss_info[dset_id]["mse"].value, self.global_step)
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info, lr=self.lr), refresh=False)
        progress_bar.update(1)

        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        logging.info(f"Starting training from step {self.global_step + 1}")
        data_iter = iter(self.train_data_loader)
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            self.init_loss_info()
            while self.global_step < self.num_steps:
                self.model.train()
                data = next(data_iter)
                step_successful = self.step(data, do_update=True)
                self.global_step += 1
                if step_successful and self.scheduler is not None:
                    self.scheduler.step()
                for r in self.regularizers:
                    r.step(self.global_step)
                self.post_step(data=data, progress_bar=pb)
        finally:
            pb.close()
            self.writer.close()

    def validate(self):
        val_metrics = []
        self.model.eval()
        with torch.no_grad():
            for dset_id, dataset in enumerate(self.test_datasets):
                per_scene_metrics = {
                    "psnr": 0,
                    "ssim": 0,
                    "dset_id": dset_id,
                }
                pb = tqdm(total=len(dataset), desc=f"Test scene {dset_id} ({dataset.name})")
                dset = self.train_datasets[0]
                dset.training = False
                dset.split = 'test'
                data = dset[0]
                preds = self.eval_step(data, dset_id=0)
                out_metrics = self.evaluate_metrics(
                    data['pixels'], preds, dset_id=0, dset=dset, img_idx=0, name='train',
                    save_outputs=True)
                print('train psnr', out_metrics['psnr'])
                dset.training = True
                dset.split = 'train'
                for img_idx, data in enumerate(dataset):
                    preds = self.eval_step(data, dset_id=dset_id)
                    out_metrics = self.evaluate_metrics(
                        data["pixels"], preds, dset_id=dset_id, dset=dataset, img_idx=img_idx, name=None,
                        save_outputs=self.save_outputs)
                    per_scene_metrics["psnr"] += out_metrics["psnr"]
                    per_scene_metrics["ssim"] += out_metrics["ssim"]
                    pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                    pb.update(1)
                pb.close()
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"step {self.global_step}/{self.num_steps} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)

        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def evaluate_metrics(self, gt, preds: MutableMapping[str, torch.Tensor], dset, dset_id: int, img_idx: int,
                         name: Optional[str] = None, save_outputs: bool = True):
        preds_rgb = (
            preds["rgb"].reshape(dset.intrinsics.height, dset.intrinsics.width, 3)
                        .cpu()
                        .clamp(0, 1)
        )
        exrdict = {
            "preds": preds_rgb.numpy(),
        }
        summary = dict()

        if "depth" in preds:
            # normalize depth and add to exrdict
            depth = preds["depth"]
            depth = depth - depth.min()
            depth = depth / depth.max()
            depth = depth.cpu().reshape(dset.intrinsics.height, dset.intrinsics.width)[..., None]
            preds["depth"] = depth
            exrdict["depth"] = preds["depth"].numpy()

        if gt is not None:
            gt = gt.reshape(dset.intrinsics.height, dset.intrinsics.width, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            exrdict["gt"] = gt.numpy()

            err = (gt - preds_rgb) ** 2
            exrdict["err"] = err.numpy()
            summary["mse"] = torch.mean(err)
            summary["psnr"] = metrics.psnr(preds_rgb, gt)
            summary["ssim"] = metrics.ssim(preds_rgb, gt)

        if save_outputs:
            out_name = f"step{self.global_step}-D{dset_id}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
            write_png(os.path.join(self.log_dir, out_name + ".png"), (preds_rgb * 255.0).byte().numpy())
            if "depth" in preds:
                out_name = f"step{self.global_step}-D{dset_id}-{img_idx}-depth"
                depth = preds["depth"].repeat(1, 1, 3)
                write_png(os.path.join(self.log_dir, out_name + ".png"), (depth * 255.0).byte().numpy())

        return summary

    def save_model(self):
        model_fname = os.path.join(self.log_dir, f'model.pth')
        logging.info(f'Saving model checkpoint to: {model_fname}')

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "occupancy_grids": [og.state_dict() for og in self.occupancy_grids],
            "global_step": self.global_step
        }, model_fname)

    def load_model(self, checkpoint_data):
        if self.transfer_learning:
            # Only reload model components that are not scene-specific, and don't reload the optimizer or scheduler
            for key in checkpoint_data["model"].keys():
                if 'scene' in key or 'density' in key or 'aabb' in key or 'resolution' in key:
                    continue
                self.model.load_state_dict(OrderedDict({key: checkpoint_data["model"][key]}), strict=False)
                logging.info(f"=> Loaded model state {key} with shape {checkpoint_data['model'][key].shape} from checkpoint")
        else:
            # Loading model grids is complicated due to possible shrinkage.
            for k, v in checkpoint_data['model'].items():
                if 'resolution' in k:
                    grid_id = int(k[-1])  # TODO: won't work with more than 10 scenes
                    self.model.upsample(v.cpu().tolist(), grid_id)
            self.model.load_state_dict(checkpoint_data["model"])
            logging.info("=> Loaded model state from checkpoint")

            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            logging.info("=> Loaded optimizer state from checkpoint")

            assert len(self.occupancy_grids) == len(checkpoint_data["occupancy_grids"])
            for og, og_sd in zip(self.occupancy_grids, checkpoint_data["occupancy_grids"]):
                og.load_state_dict(og_sd)
            logging.info("=> Loaded occupancy grids state from checkpoint")

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
                logging.info("=> Loaded scheduler state from checkpoint")
            self.global_step = checkpoint_data["global_step"]
            logging.info(f"=> Loaded step {self.global_step} from checkpoints")

    def init_loss_info(self):
        ema_weight = 1.0
        self.loss_info = [defaultdict(lambda: EMA(ema_weight)) for _ in range(self.num_dsets)]

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        logging.info(f"Initializing LR Scheduler of type {self.scheduler_type} with "
                     f"{max_steps} maximum steps.")
        if self.scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif self.scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif self.scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
        elif self.scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        if self.optim_type == 'adam':
            optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr']))
        else:
            raise NotImplementedError()
        return optim

    def init_model(self, **kwargs) -> torch.nn.Module:
        aabbs = [d.scene_bbox for d in self.train_datasets]
        model = LowrankLearnableHash(
            num_scenes=self.num_dsets,
            grid_config=kwargs.pop("grid_config"),
            aabb=aabbs,
            is_ndc=self.is_ndc,  # TODO: This should also be per-scene
            render_n_samples=self.render_n_samples,
            **kwargs)
        logging.info(f"Initialized LowrankLearnableHash model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        return model

    def init_occupancy_grid(self, **kwargs):
        occupancy_grids = []
        for scene in range(self.num_dsets):
            occupancy_grid = OccupancyGrid(
                roi_aabb=self.model.aabb(scene).view(-1),
                resolution=self.model.resolution(scene)[:3],
                contraction_type=self.contraction_type,
            ).cuda()
            occupancy_grids.append(occupancy_grid)
        return occupancy_grids

    def init_regularizers(self, **kwargs):
        regularizers = [
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
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in regularizers if r.weight > 0]
        return regularizers

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def density_threshold(self):
        if self.global_step < 512:
            return self._density_threshold / 10
        return self._density_threshold

    @property
    def alpha_threshold(self):
        if self.global_step < 512:
            return self._alpha_threshold
        return self._alpha_threshold

    @property
    def max_rays(self):
        if self.global_step < 512:
            return 15_000
        elif self.global_step < 1024:
            return 100_000
        return 1_000_000


def losses_to_postfix(losses: List[Dict[str, EMA]], lr: float) -> str:
    pfix_list = []
    for dset_id, loss_dict in enumerate(losses):
        pfix_inner = []
        for lname, lval in loss_dict.items():
            pfix_inner.append(f"{lname}={lval}")
        pfix_list.append(f"D{dset_id}({', '.join(pfix_inner)})")
    if lr is not None:
        pfix_list.append(f"lr={lr:.2e}")
    return '  '.join(pfix_list)


def load_data(data_downsample, data_dirs, batch_size, **kwargs):
    # TODO: multiple different dataset types are currently not supported well.
    def decide_dset_type(dd) -> str:
        if ("chair" in dd or "drums" in dd or "ficus" in dd or "hotdog" in dd
                or "lego" in dd or "materials" in dd or "mic" in dd
                or "ship" in dd):
            return "synthetic"
        elif ("fern" in dd or "flower" in dd or "fortress" in dd
              or "horns" in dd or "leaves" in dd or "orchids" in dd
              or "room" in dd or "trex" in dd):
            return "llff"
        else:
            raise RuntimeError(f"data_dir {dd} not recognized as LLFF or Synthetic dataset.")

    data_resolution = parse_optint(kwargs.get('data_resolution'))

    tr_dsets, ts_dsets = [], []
    for i, data_dir in enumerate(data_dirs):
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample,
                max_frames=max_tr_frames, batch_size=batch_size,
                dset_id=i))
            ts_dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, max_frames=max_ts_frames,
                dset_id=i))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            logging.info(f"About to load LLFF data downsampled by {data_downsample} times.")
            tr_dsets.append(LLFFDataset(
                data_dir, split='train', downsample=data_downsample, hold_every=hold_every,
                batch_size=batch_size, dset_id=i))
            ts_dsets.append(LLFFDataset(
                data_dir, split='test', downsample=4, hold_every=hold_every, dset_id=i))
        else:
            raise ValueError(dset_type)

    cat_tr_dset = MultiSceneDataset(tr_dsets)
    tr_loader = torch.utils.data.DataLoader(
        cat_tr_dset, num_workers=0, prefetch_factor=2, pin_memory=False,
        batch_size=None, sampler=None)

    return {"ts_dsets": ts_dsets, "tr_dsets": tr_dsets, "tr_loader": tr_loader}


def N_to_reso(num_voxels, aabb):
    voxel_size = ((aabb[1] - aabb[0]).prod() / num_voxels).pow(1 / 3)
    return ((aabb[1] - aabb[0]) / voxel_size).long().cpu().tolist()
