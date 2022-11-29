import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, MutableMapping, Union, Sequence, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from matplotlib.colors import LogNorm
from torch.utils.tensorboard import SummaryWriter

from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from .base_trainer import BaseTrainer
from .regularization import (
    PlaneTV, L1PlaneColor, L1PlaneDensity, VolumeTV, L1Density, FloaterLoss,
    HistogramLoss, DensityPlaneTV
)
from .utils import (
    init_dloader_random
)
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..datasets.multi_dataset_sampler import MultiSceneSampler
from ..my_tqdm import tqdm
from ..utils import parse_optint


class Trainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dsets: List[torch.utils.data.TensorDataset],
                 tr_dsets: List[torch.utils.data.TensorDataset],
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.test_datasets = ts_dsets
        self.train_datasets = tr_dsets
        self.is_ndc = self.test_datasets[0].is_ndc
        self.is_contracted = self.test_datasets[0].is_contracted
        self.num_dsets = len(self.train_datasets)

        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=save_outputs,
            device=device,
            **kwargs
        )

        self.transfer_learning = kwargs.get('transfer_learning')

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        dset_id = data["dset_id"]
        batch_size = self.eval_batch_size
        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            near_far = data["near_far"].to(self.device) if "near_far" in data else None
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(rays_o_b, rays_d_b, grid_id=dset_id, near_far=near_far,
                                     bg_color=bg_color, channels=channels)
                for k, v in outputs.items():
                    preds[k].append(v)
        return {k: torch.cat(v, 0) for k, v in preds.items() if (
                    k in channels or k.startswith("proposal_depth"))}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        super().train_step(data, **kwargs)
        dset_id = data["dset_id"]
        rays_o = data["rays_o"].to(self.device)
        rays_d = data["rays_d"].to(self.device)
        near_far = data["near_far"].to(self.device) if "near_far" in data else None
        imgs = data["imgs"].to(self.device)
        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(rays_o, rays_d, grid_id=dset_id, bg_color=bg_color,
                                 channels={"rgb"}, near_far=near_far)
            rgb_preds = fwd_out["rgb"]
            # Reconstruction loss
            recon_loss = self.criterion(rgb_preds, imgs)

            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, grid_id=dset_id, model_out=fwd_out)
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
                self.loss_info[f"mse_{dset_id}"].update(recon_loss_val)
                self.loss_info[f"psnr_{dset_id}"].update(-10 * math.log10(recon_loss_val))
                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in every train-dataset
        for d in self.train_datasets:
            d.reset_iter()

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
        return base_save_dict

    def load_model(self, checkpoint_data):
        super().load_model(checkpoint_data)

    def init_epoch_info(self):
        ema_weight = 0.9  # higher places higher weight to new observations
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> torch.nn.Module:
        aabbs = [d.scene_bbox for d in self.test_datasets]
        try:
            global_translation = self.test_datasets[0].global_translation
        except AttributeError:
            global_translation = None
        try:
            global_scale = self.test_datasets[0].global_scale
        except AttributeError:
            global_scale = None

        model = LowrankLearnableHash(
            num_scenes=self.num_dsets,
            grid_config=kwargs.pop("grid_config"),
            aabb=aabbs,
            is_ndc=self.is_ndc,
            is_contracted=self.is_contracted,
            proposal_sampling=self.extra_args.get('histogram_loss_weight', 0.0) > 0.0,
            global_translation=global_translation,
            global_scale=global_scale,
            **kwargs)
        logging.info(f"Initialized LowrankLearnableHash model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        return model

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
            FloaterLoss(kwargs.get('floater_loss_weight', 0.0)),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return self.num_dsets * 2 + 1


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


def init_tr_data(data_downsample: float, data_dirs: Sequence[str], **kwargs):
    batch_size = int(kwargs['batch_size'])
    dsets = []

    for i, data_dir in enumerate(data_dirs):
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample,
                max_frames=max_tr_frames, batch_size=batch_size, dset_id=i))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            dsets.append(LLFFDataset(
                data_dir, split='train', downsample=int(data_downsample), hold_every=hold_every,
                batch_size=batch_size, dset_id=i))
        dsets[-1].reset_iter()

    tr_sampler = MultiSceneSampler(dsets, num_samples_per_dataset=1)
    cat_tr_dset = torch.utils.data.ConcatDataset(dsets)
    tr_loader = torch.utils.data.DataLoader(
        cat_tr_dset, num_workers=4, prefetch_factor=2, pin_memory=True,
        batch_size=None, sampler=tr_sampler, worker_init_fn=init_dloader_random)

    return {
        "tr_dsets": dsets,
        "tr_loader": tr_loader,
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


@torch.no_grad()
def visualize_planes_withF(model, save_dir: str, name: str):
    MAX_RANK = 3
    rank = model.config[0]["rank"]
    used_rank = min(model.config[0]["rank"], MAX_RANK)
    dim = model.config[0]["output_coordinate_dim"]

    # For each plane get the n-d coordinates and plot the density
    # corresponding to those coordinates in F.
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
        fig, ax = plt.subplots(ncols=used_rank, nrows=n_planes, figsize=(3 * used_rank, 3 * n_planes))
        for plane_idx, grid in enumerate(grids):
            _, c, h, w = grid.data.shape
            grid = grid.data.view(dim, rank, h, w)
            for r in range(used_rank):
                grid_r = grid[:, r, ...].view(dim, -1).transpose(0, 1)  # h*w, dim
                multi_scale_interp = (grid_r - model.pt_min) / (model.pt_max - model.pt_min)
                multi_scale_interp = multi_scale_interp * 2 - 1
                from plenoxels.models.utils import grid_sample_wrapper
                out = grid_sample_wrapper(model.features, multi_scale_interp).view(h, w, -1)
                density = model.density_act(
                    out[..., -1].cpu()
                ).numpy()
                im = ax[plane_idx, r].imshow(density, norm=LogNorm(vmin=1e-6, vmax=density.max()))
                ax[plane_idx, r].axis("off")
                plt.colorbar(im, ax=ax[plane_idx, r], aspect=20, fraction=0.04)
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}-planes-scale-{scale_id}.png"))
        plt.cla()
        plt.clf()
        plt.close()


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
        fig, ax = plt.subplots(nrows=n_planes, ncols=2*rank, figsize=(3*2*rank, 3*n_planes))
        for plane_idx, grid in enumerate(grids):
            _, c, h, w = grid.data.shape

            grid = grid.data.view(dim, rank, h, w)
            for r in range(rank):
                features = grid[:, r, :, :].view(dim, h*w).transpose(0, 1)

                density = (
                    model.density_act(
                        model.decoder.compute_density(
                            features=features, rays_d=None)
                    ).view(h, w)
                     .cpu()
                     .float()
                     .nan_to_num(posinf=99.0, neginf=-99.0)
                     .clamp_min(1e-6)
                ).numpy()

                im = ax[plane_idx, r].imshow(density, norm=LogNorm(vmin=1e-6, vmax=density.max()))
                ax[plane_idx, r].axis("off")
                plt.colorbar(im, ax=ax[plane_idx, r], aspect=20, fraction=0.04)

                rays_d = torch.ones((h*w, 3), device=grid.device)
                rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
                # color = (
                #         torch.sigmoid(model.decoder.compute_color(features, rays_d))
                # ).view(h, w, 3).cpu().float().numpy()
                # ax[plane_idx, r+rank].imshow(color)
                # ax[plane_idx, r+rank].axis("off")

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}-planes-scale-{scale_id}.png"))
        plt.cla()
        plt.clf()
        plt.close()
