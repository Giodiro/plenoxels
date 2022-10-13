import logging
import math
import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, MutableMapping, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from nerfacc import ContractionType, OccupancyGrid, ray_marching, rendering
from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash, DensityMask
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_exr, write_png
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..datasets.synthetic_nerf_dataset import MultiSceneDataset
from ..my_tqdm import tqdm
from ..utils import parse_optint


class Trainer():
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dsets: List[torch.utils.data.TensorDataset],
                 tr_dsets: List[torch.utils.data.TensorDataset],
                 regnerf_weight_start: float,
                 regnerf_weight_end: float,
                 regnerf_weight_max_step: int,
                 plane_tv_weight: float,
                 l1density_weight: float,
                 volume_tv_weight: float,
                 volume_tv_npts: int,
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

        self.extra_args = kwargs
        self.num_dsets = len(self.train_datasets)

        self.regnerf_weight_start = regnerf_weight_start
        self.regnerf_weight_end = regnerf_weight_end
        self.regnerf_weight_max_step = regnerf_weight_max_step
        self.cur_regnerf_weight = self.regnerf_weight_start
        self.plane_tv_weight = plane_tv_weight
        self.plane_tv_what = kwargs.get('plane_tv_what', 'features')
        self.l1density_weight = l1density_weight
        self.volume_tv_weight = volume_tv_weight
        self.volume_tv_npts = volume_tv_npts
        self.volume_tv_what = kwargs.get('volume_tv_what', 'Gcoords')
        self.volume_tv_patch_size = kwargs.get('volume_tv_patch_size', 3)

        self.num_steps = num_steps

        self.scheduler_type = scheduler_type
        self.optim_type = optim_type
        self.transfer_learning = kwargs.get('transfer_learning')
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.gradient_acc = kwargs.get('gradient_acc', False)
        self.target_sample_batch_size = 1 << 18
        # All 'steps' things must be multiplied by the number of datasets
        # to get consistent amount of training independently of how many dsets.
        # This is not needed if gradient_acc is set, since in that case the
        # global_step is only updated once every time we cycle through each scene
        step_multiplier = 1 if self.gradient_acc else self.num_dsets
        self.density_mask_update_steps = [s * step_multiplier for s in kwargs.get('dmask_update', [])]
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
        self.contraction_type = ContractionType.AABB

        self.model = self.init_model(**self.extra_args)
        self.occupancy_grids = self.init_occupancy_grid(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        #self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

    def eval_step(self, data, dset_id) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        batch_size = self.train_datasets[dset_id].batch_size
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

    def step(self, data: Dict[str, Union[int, torch.Tensor]], do_update: bool):
        dset_id = data["dset_id"]
        rays_o = data["rays_o"].cuda()
        rays_d = data["rays_d"].cuda()
        imgs = data["imgs"].cuda()
        bg_color = data["bg_color"].cuda()

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # update occupancy grid
            self.occupancy_grids[dset_id].every_n_step(
                step=self.global_step,
                occ_eval_fn=lambda x: self.model.query_opacity(
                    x, dset_id
                ),
                warmup_steps=512,
                n=64,
                occ_thre=self.extra_args['density_threshold'],
            )
            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                self.model,
                self.occupancy_grids[dset_id],
                dset_id,
                rays_o,
                rays_d,
                # rendering options
                near_plane=self.train_datasets[dset_id].near_far[0],
                far_plane=self.train_datasets[dset_id].near_far[1],
                render_bkgd=bg_color,
                cone_angle=0.0,
            )
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(imgs)
            num_rays = int(
                num_rays
                * (self.target_sample_batch_size / float(n_rendering_samples))
            )
            self.train_datasets[dset_id].update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            recon_loss = self.criterion(rgb[alive_ray_mask], imgs[alive_ray_mask])
            loss = recon_loss

        if do_update:
            self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()

        # Report on losses
        recon_loss_val = recon_loss.item()
        if len(self.test_datasets) < 5:
            self.loss_info[dset_id]["mse"].update(recon_loss_val)
        self.loss_info[dset_id]["psnr"].update(-10 * math.log10(recon_loss_val))

        # Update weights
        if do_update:
            self.gscaler.step(self.optimizer)
            scale = self.gscaler.get_scale()
            self.gscaler.update()

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

            # We reset the optimizer in case some of the parameters in model were changed.
            if opt_reset_required:
                self.optimizer = self.init_optim(**self.extra_args)

            return scale <= self.gscaler.get_scale()
        return True

    def post_step(self, data, progress_bar):
        # Anneal regularization weights
        if self.regnerf_weight_start > 0:
            w = np.clip(self.global_step / (1 if self.regnerf_weight_max_step < 1 else self.regnerf_weight_max_step), 0, 1)
            self.cur_regnerf_weight = self.regnerf_weight_start * (1 - w) + w * self.regnerf_weight_end

        dset_id = data["dset_id"]
        self.writer.add_scalar(f"mse/D{dset_id}", self.loss_info[dset_id]["mse"].value, self.global_step)
        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info, lr=self.cur_lr()), refresh=False)
        progress_bar.update(1)

        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()
        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()

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
                if step_successful:
                    self.scheduler.step()
                self.post_step(data=data, progress_bar=pb)
        finally:
            pb.close()
            self.writer.close()

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
                pb.close()
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"STEP {self.global_step}/{self.num_steps} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)

        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.num_steps}.csv"))

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
                self.model.load_state_dict(OrderedDict({key: checkpoint_data["model"][key]}), strict=False)
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
                self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
                logging.info("=> Loaded scheduler state from checkpoint")
            self.global_step = checkpoint_data["global_step"]
            logging.info(f"=> Loaded step {self.global_step} from checkpoints")

    def init_loss_info(self):
        ema_weight = 0.1
        self.loss_info = [defaultdict(lambda: EMA(ema_weight)) for _ in range(self.num_dsets)]

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = int(self.num_epochs * len(self.train_data_loader))
        if self.scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
            logging.info(f"Initialized CosineAnnealing LR Scheduler with {max_steps} maximum steps.")
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
            render_n_samples=1024,
            **kwargs)
        logging.info(f"Initialized LowrankLearnableHash model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model

    def init_occupancy_grid(self, **kwargs):
        occupancy_grids = []
        for scene in range(self.num_dsets):
            occupancy_grid = OccupancyGrid(
                roi_aabb=self.model.aabb(scene).view(-1),
                resolution=self.model.resolution(scene),
                contraction_type=self.contraction_type,
            ).cuda()
            occupancy_grids.append(occupancy_grid)
        return occupancy_grids

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
    regnerf_weight = float(kwargs.get('regnerf_weight_start'))
    extra_views = regnerf_weight > 0
    patch_size = 8

    tr_dsets, ts_dsets = [], []
    for i, data_dir in enumerate(data_dirs):
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                max_frames=max_tr_frames, extra_views=extra_views, batch_size=batch_size,
                patch_size=patch_size, dset_id=i, color_bkgd_aug='random'))
            ts_dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, resolution=800, max_frames=max_ts_frames,
                dset_id=i, color_bkgd_aug='random'))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            logging.info(f"About to load LLFF data downsampled by {data_downsample} times.")
            tr_dsets.append(LLFFDataset(
                data_dir, split='train', downsample=data_downsample, hold_every=hold_every,
                extra_views=extra_views, batch_size=batch_size, patch_size=patch_size,
                dset_id=i, color_bkgd_aug='random'))
            # Note that LLFF has same downsampling applied to train and test datasets
            ts_dsets.append(LLFFDataset(
                data_dir, split='test', downsample=4, hold_every=hold_every,
                dset_id=i, color_bkgd_aug='random'))
        else:
            raise ValueError(dset_type)

    cat_tr_dset = MultiSceneDataset(tr_dsets)
    tr_loader = torch.utils.data.DataLoader(
        cat_tr_dset, num_workers=4, prefetch_factor=4, pin_memory=True,
        batch_size=None, sampler=None)

    return {"ts_dsets": ts_dsets, "tr_dsets": tr_dsets, "tr_loader": tr_loader}


def N_to_reso(num_voxels, aabb):
    voxel_size = ((aabb[1] - aabb[0]).prod() / num_voxels).pow(1 / 3)
    return ((aabb[1] - aabb[0]) / voxel_size).long().cpu().tolist()


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    grid_id: int,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    device="cuda:0",
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays_o.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays_o = rays_o.reshape([num_rays] + list(rays_o.shape[2:]))
        rays_d = rays_d.reshape([num_rays] + list(rays_d.shape[2:]))
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field.query_density(positions, grid_id)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs, grid_id)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        packed_info, t_starts, t_ends = ray_marching(
            rays_o[i: i + chunk],
            rays_d[i: i + chunk],
            scene_aabb=radiance_field.aabb(grid_id).view(-1),
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
        )
        rgb, opacity, depth = rendering(
            rgb_sigma_fn,
            packed_info,
            t_starts,
            t_ends,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )
