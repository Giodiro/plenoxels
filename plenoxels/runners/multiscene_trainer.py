import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, MutableMapping, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash, DensityMask
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_exr, write_png
from .regularization import (
    PlaneTV, L1PlaneColor, L1PlaneDensity, VolumeTV, L1Density, FloaterLoss,
    HistogramLoss
)
from .timer import CudaTimer
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..datasets.multi_dataset_sampler import MultiSceneSampler
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
                 **kwargs
                 ):
        self.train_data_loader = tr_loader
        self.test_datasets = ts_dsets
        self.train_datasets = tr_dsets
        self.is_ndc = self.test_datasets[0].is_ndc
        self.is_contracted = self.test_datasets[0].is_contracted
        self.eval_batch_size = kwargs.get('eval_batch_size', 8192)

        self.extra_args = kwargs
        self.num_dsets = len(self.train_datasets)

        self.num_steps = num_steps

        self.scheduler_type = scheduler_type
        self.optim_type = optim_type
        self.transfer_learning = kwargs.get('transfer_learning')
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        step_multiplier = self.num_dsets
        self.density_mask_update_steps = [s * step_multiplier for s in kwargs.get('dmask_update', [])]
        self.upsample_steps = [s * step_multiplier for s in kwargs.get('upsample_steps', [])]
        self.upsample_resolution_list = list(kwargs.get('upsample_resolution', []))
        self.upsample_F_steps = list(kwargs.get('upsample_F_steps', []))

        assert len(self.upsample_resolution_list) == len(self.upsample_steps), \
            f"Got {len(self.upsample_steps)} upsample_steps and {len(self.upsample_resolution_list)} upsample_resolution."

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step = None
        self.loss_info = None
        self.train_iterators = None
        self.timer = CudaTimer(enabled=False)

        self.model = self.init_model(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.regularizers = self.init_regularizers(**self.extra_args)
        #self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

    def eval_step(self, data, dset_id) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].cuda()
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].cuda()
                if self.is_ndc:
                    bg_color = None
                else:
                    bg_color = 1
                outputs = self.model(rays_o_b, rays_d_b, grid_id=dset_id, bg_color=bg_color, channels={"rgb", "depth"})
                for k, v in outputs.items():
                    preds[k].append(v)
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def step(self, data: Dict[str, Union[int, torch.Tensor]]):
        self.timer.reset()
        dset_id = data["dset_id"]
        rays_o = data["rays_o"].cuda()
        rays_d = data["rays_d"].cuda()
        near_far = data["near_far"].cuda() if "near_far" in data else None
        imgs = data["imgs"].cuda()

        C = imgs.shape[-1]
        if self.is_ndc:
            bg_color = None
        elif C == 3:
            bg_color = 1
        else:  # Random bg-color
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])
        self.timer.check("step_prepare")

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(rays_o, rays_d, grid_id=dset_id, bg_color=bg_color, channels={"rgb"}, near_far=near_far)
            self.timer.check("step_model")
            rgb_preds = fwd_out["rgb"]
            # Reconstruction loss
            recon_loss = self.criterion(rgb_preds, imgs)
            loss = recon_loss
            # Regularization
            for r in self.regularizers:
                loss = loss + r.regularize(self.model, grid_id=dset_id, model_out=fwd_out)
            self.timer.check("step_loss")
        self.gscaler.scale(loss).backward()

        # Update weights
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.timer.check("step_backprop")

        # Report on losses
        recon_loss_val = recon_loss.item()
        self.loss_info[dset_id]["mse"].update(recon_loss_val)
        self.loss_info[dset_id]["psnr"].update(-10 * math.log10(recon_loss_val))
        for r in self.regularizers:
            r.report(self.loss_info[dset_id])

        # Run grid-update routines (masking, shrinking, upscaling)
        opt_reset_required = False
        if self.global_step in self.density_mask_update_steps:
            logging.info(f"Updating alpha-mask for all datasets at step {self.global_step}.")
            for u_dset_id in range(self.num_dsets):
                new_aabb = self.model.update_alpha_mask(grid_id=u_dset_id)
                sorted_updates = sorted(self.density_mask_update_steps)
                if self.global_step == sorted_updates[0] or self.global_step == sorted_updates[1]:
                    self.model.shrink(new_aabb, grid_id=u_dset_id)
                    opt_reset_required = True
        try:
            upsample_step_idx = self.upsample_steps.index(self.global_step)  # if not an upsample step will raise
            new_num_voxels = self.upsample_resolution_list[upsample_step_idx]
            logging.info(f"Upsampling all datasets at step {self.global_step} to {new_num_voxels} voxels.")
            for u_dset_id in range(self.num_dsets):
                new_reso = N_to_reso(new_num_voxels, self.model.aabb(u_dset_id))
                self.model.upsample(new_reso, u_dset_id)
            opt_reset_required = True
        except ValueError:
            pass
        if self.global_step in self.upsample_F_steps:
            self.model.upsample_F(new_reso = self.model.config[1]["resolution"][0] * 2)  # Double the resolution in each dimension of F
            self.model.config[1]["resolution"] = [r * 2 for r in self.model.config[1]["resolution"]]  # Update the config
            opt_reset_required = True

        # We reset the optimizer in case some of the parameters in model were changed.
        if opt_reset_required:
            self.optimizer = self.init_optim(**self.extra_args)

        self.timer.check("step_remaining")
        return scale <= self.gscaler.get_scale()

    def post_step(self, data, progress_bar):
        dset_id = data["dset_id"]
        self.writer.add_scalar(f"mse/D{dset_id}", self.loss_info[dset_id]["mse"].value, self.global_step)

        for key in self.timer.timings:
            self.writer.add_scalar(f"timer/{key}", self.timer.timings[key], self.global_step)
        for key in self.model.timer.timings:
            self.writer.add_scalar(f"timer/{key}", self.model.timer.timings[key], self.global_step)

        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info, lr=self.cur_lr()), refresh=False)
        progress_bar.update(1)

    def pre_epoch(self):
        for d in self.train_datasets:
            d.reset_iter()
        self.init_epoch_info()
        self.model.train()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        logging.info(f"Starting training from step {self.global_step + 1}")
        self.pre_epoch()
        pb = tqdm(total=self.num_steps, desc=f"")
        batch_iter = iter(self.train_data_loader)
        while self.global_step < self.num_steps:
            try:
                # Get a batch of data
                self.timer.reset()
                data = next(batch_iter)
                self.timer.check("data")

                step_successful = self.step(data)
                # Update the progress bar
                self.post_step(data=data, progress_bar=pb)
                if step_successful and self.scheduler is not None:
                    self.scheduler.step()

                # Check if we need to save model at this step
                if self.save_every > -1 and self.global_step % self.save_every == 0 and self.global_step > 0:
                    self.model.eval()
                    self.save_model()
                    self.model.train()
                if self.valid_every > -1 and self.global_step % self.valid_every == 0 and self.global_step > 0:
                    self.model.eval()
                    self.validate()
                    self.model.train()

                self.global_step += 1
            except StopIteration as e:
                logging.info(str(e))
                logging.info(f'resetting after a full pass through the data, or when the dataset changed')
                self.pre_epoch()
                batch_iter = iter(self.train_data_loader)
        pb.close()

    def validate(self):
        val_metrics = []
        with torch.no_grad():
            for dset_id, dataset in enumerate(self.test_datasets):
                per_scene_metrics = {
                    "psnr": 0,
                    "ssim": 0,
                    "dset_id": dset_id,
                }
                pb = tqdm(total=len(dataset), desc=f"Test scene {dset_id} ({dataset.name})")
                for img_idx, data in enumerate(dataset):
                    preds = self.eval_step(data, dset_id=dset_id)
                    out_metrics = self.evaluate_metrics(
                        data["imgs"], preds, dset_id=dset_id, dset=dataset, img_idx=img_idx, name=None,
                        save_outputs=self.save_outputs)
                    per_scene_metrics["psnr"] += out_metrics["psnr"]
                    per_scene_metrics["ssim"] += out_metrics["ssim"]
                    pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                    pb.update(1)
                if False:  # Save a training image as well
                    dset = self.train_datasets[0]
                    data = dict(rays_o=dset.rays_o.view(-1, gt.shape[0], 3)[0],
                                rays_d=dset.rays_d.view(-1, gt.shape[0], 3)[0],
                                imgs=dset.imgs.view(-1, gt.shape[0], gt.shape[1])[0])
                    preds = self.eval_step(data, dset_id=dset_id)
                    out_metrics = self.evaluate_metrics(
                        data["imgs"], preds, dset_id=dset_id, dset=dset, img_idx=0, name="train",
                        save_outputs=True)
                    print(f"train img 0 PSNR={out_metrics['psnr']}")
                pb.close()
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"step {self.global_step}/{self.num_steps} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)

            # visualize planes
            if self.save_outputs:
                visualize_planes(self.model, self.log_dir, f"step{self.global_step}-D{dset_id}")

        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def evaluate_metrics(self, gt, preds: MutableMapping[str, torch.Tensor], dset, dset_id: int, img_idx: int,
                         name: Optional[str] = None, save_outputs: bool = True):
        preds_rgb = preds["rgb"].reshape(dset.img_h, dset.img_w, 3).cpu()
        exrdict = {
            "preds": preds_rgb.numpy(),
        }
        summary = dict()

        if "depth" in preds:
            # normalize depth and add to exrdict
            depth = preds["depth"]
            depth = depth - depth.min()
            depth = depth / depth.max()
            depth = depth.cpu().reshape(dset.img_h, dset.img_w)[..., None]
            preds["depth"] = depth
            exrdict["depth"] = preds["depth"].numpy()

        if gt is not None:
            gt = gt.reshape(dset.img_h, dset.img_w, -1).cpu()
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
        """Override this function to change model saving."""
        model_fname = os.path.join(self.log_dir, f'model.pth')
        logging.info(f'Saving model checkpoint to: {model_fname}')

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step
        }, model_fname)

    def load_model(self, checkpoint_data):
        if self.transfer_learning:
            # Only reload model components that are not scene-specific, and don't reload the optimizer or scheduler
            for key in checkpoint_data["model"].keys():
                if 'scene' in key or 'density' in key or 'aabb' in key or 'resolution' in key:
                    continue
                self.model.load_state_dict({key: checkpoint_data["model"][key]}, strict=False)
                logging.info(f"=> Loaded model state {key} with shape {checkpoint_data['model'][key].shape} from checkpoint")
        else:
            # Loading model grids is complicated due to possible shrinkage.
            for k, v in checkpoint_data['model'].items():
                if 'resolution' in k:
                    grid_id = int(k[-1])  # TODO: won't work with more than 10 scenes
                    self.model.upsample(v.cpu().tolist(), grid_id)
                if 'density_volume' in k:
                    grid_id = int(k.split('.')[1])  # 'density_mask.0.density_volume'
                    self.model.density_mask[grid_id] = DensityMask(
                        density_volume=v, aabb=torch.empty(2, 3, dtype=torch.float32, device=v.device))
            self.model.load_state_dict(checkpoint_data["model"])

            logging.info("=> Loaded model state from checkpoint")
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            logging.info("=> Loaded optimizer state from checkpoint")
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint_data['scheduler'])
                logging.info("=> Loaded scheduler state from checkpoint")
            self.global_step = checkpoint_data["global_step"]
            logging.info(f"=> Loaded step {self.global_step} from checkpoints")

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = [defaultdict(lambda: EMA(ema_weight)) for _ in range(self.num_dsets)]

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 1e-5
        lr_sched = None
        max_steps = self.num_steps
        if self.scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
            logging.info(f"Initialized CosineAnnealing LR Scheduler with {max_steps} maximum "
                         f"steps.")
        elif self.scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=512,
                num_training_steps=max_steps,
                eta_min=eta_min)
            logging.info(f"Initialized CosineAnnealing LR Scheduler with {max_steps} maximum "
                         f"steps and {512} warmup steps.")
        elif self.scheduler_type == "step_many":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
            logging.info(f"Initialized Many-step LR Scheduler with {max_steps} maximum "
                         f"steps.")
        elif self.scheduler_type == "warmup_step_many":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=512,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
            logging.info(f"Initialized Many-step LR Scheduler with {max_steps} maximum "
                         f"steps and {512} warmup steps.")
        elif self.scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                ],
                gamma=0.1)
        elif self.scheduler_type is not None:
            raise ValueError(self.scheduler_type)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        if self.optim_type == 'adam':
            optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr']))
        else:
            raise NotImplementedError()
        return optim

    def init_model(self, **kwargs) -> torch.nn.Module:
        aabbs = [d.scene_bbox for d in self.test_datasets]

        model = LowrankLearnableHash(
            num_scenes=self.num_dsets,
            grid_config=kwargs.pop("grid_config"),
            aabb=aabbs,
            is_ndc=self.is_ndc,
            is_contracted=self.is_contracted,
            proposal_sampling=self.extra_args.get('histogram_loss_weight', 0.0) > 0.0,
            **kwargs)
        logging.info(f"Initialized LowrankLearnableHash model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model

    def init_regularizers(self, **kwargs):
        regularizers = [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0)),
            VolumeTV(
                kwargs.get('volume_tv_weight', 0.0),
                what=kwargs.get('volume_tv_what'),
                patch_size=kwargs.get('volume_tv_patch_size', 3),
                batch_size=kwargs.get('volume_tv_npts', 100),
            ),
            L1PlaneColor(kwargs.get('l1_plane_color_weight', 0.0)),
            L1PlaneDensity(kwargs.get('l1_plane_density_weight', 0.0)),
            L1Density(kwargs.get('l1density_weight', 0.0), max_voxels=100_000),
            FloaterLoss(kwargs.get('floater_loss_weight', 0.0)),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
        ]
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in regularizers if r.weight > 0]
        return regularizers

    def cur_lr(self):
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return None


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
    num_batches_per_dataset = int(kwargs['num_batches_per_dset'])

    tr_dsets, ts_dsets = [], []
    for i, data_dir in enumerate(data_dirs):
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                max_frames=max_tr_frames, batch_size=batch_size, dset_id=i))
            ts_dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, resolution=800, max_frames=max_ts_frames,
                dset_id=i))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            logging.info(f"About to load LLFF data downsampled by {data_downsample} times.")
            tr_dsets.append(LLFFDataset(
                data_dir, split='train', downsample=data_downsample, hold_every=hold_every,
                batch_size=batch_size, dset_id=i))
            # Note that LLFF has same downsampling applied to train and test datasets
            ts_dsets.append(LLFFDataset(
                data_dir, split='test', downsample=4, hold_every=hold_every,
                dset_id=i))
        else:
            raise ValueError(dset_type)

    tr_sampler = MultiSceneSampler(
        tr_dsets, num_samples_per_dataset=num_batches_per_dataset)
    cat_tr_dset = torch.utils.data.ConcatDataset(tr_dsets)
    tr_loader = torch.utils.data.DataLoader(
        cat_tr_dset, num_workers=4, prefetch_factor=4, pin_memory=True,
        batch_size=None, sampler=tr_sampler)

    return {"ts_dsets": ts_dsets, "tr_dsets": tr_dsets, "tr_loader": tr_loader}


def N_to_reso(num_voxels, aabb):
    voxel_size = ((aabb[1] - aabb[0]).prod() / num_voxels).pow(1 / 3)
    return ((aabb[1] - aabb[0]) / voxel_size).long().cpu().tolist()


@torch.no_grad()
def visualize_planes(model, save_dir: str, name: str):
    rank = model.config[0]["rank"]
    dim = model.feature_dim
    if hasattr(model, 'scene_grids'):  # LowrankLearnableHash
        multi_scale_grids = model.scene_grids[0]
    elif hasattr(model, 'grids'):  # LowrankVideo
        multi_scale_grids = model.grids
    else:
        raise RuntimeError(f"Cannot find grids in model {model}.")

    for scale_id, grids in enumerate(multi_scale_grids):
        if hasattr(model, 'scene_grids'):
            grids = grids[0]
        n_planes = len(grids)
        fig, ax = plt.subplots(nrows=n_planes, ncols=2*rank, figsize=(3*n_planes,3*2*rank))
        for plane_idx, grid in enumerate(grids):
            _, c, h, w = grid.data.shape

            grid = grid.data.view(dim, rank, h, w)
            for r in range(rank):
                density = model.density_act(
                    grid[-1, r, :, :].cpu()
                ).numpy()
                # density = grid[-1, r, :, :].cpu().numpy()

                im = ax[plane_idx, r].imshow(density, norm=LogNorm(vmin=1e-6, vmax=density.max()))
                ax[plane_idx, r].axis("off")
                plt.colorbar(im, ax=ax[plane_idx, r], aspect=20, fraction=0.04)

                rays_d = torch.ones((h*w, 3), device=grid.device)
                rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
                features = grid[:, r, :, :].view(dim, h*w).permute(1,0)
                color = (
                        torch.sigmoid(model.decoder.compute_color(features, rays_d))
                ).view(h, w, 3).cpu().numpy()
                ax[plane_idx, r+rank].imshow(color)
                ax[plane_idx, r+rank].axis("off")

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}-planes-scale-{scale_id}.png"))
        plt.cla()
        plt.clf()
        plt.close()

