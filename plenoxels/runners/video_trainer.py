import logging
import math
import os
from collections import defaultdict
from typing import Dict, Optional, MutableMapping

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.runners.base_trainer import BaseTrainer
from plenoxels.runners.regularization import (
    VideoPlaneTV, TimeSmoothness, L1PlaneDensityVideo,
    L1AppearancePlanes
)
from plenoxels.runners.utils import render_image


class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.Dataset,
                 ts_dset: torch.utils.data.Dataset,
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
                 **kwargs
                 ):
        # Keys we wish to ignore
        kwargs.pop('transfer_learning', None)
        kwargs.pop('num_batches_per_dset', None)
        super().__init__(tr_loader=tr_loader,
                         tr_dsets=[tr_dset],
                         ts_dsets=[ts_dset],
                         num_batches_per_dset=1,
                         num_steps=num_steps,
                         scheduler_type=scheduler_type,
                         optim_type=optim_type,
                         logdir=logdir,
                         expname=expname,
                         train_fp16=train_fp16,
                         save_every=save_every,
                         valid_every=valid_every,
                         transfer_learning=False,  # No transfer with video
                         save_outputs=save_outputs,
                         device="cpu",  # We don't want to send all the data to GPU - it's too big.
                         sample_batch_size=sample_batch_size,
                         n_samples=n_samples,
                         **kwargs)
        self.ist_step = ist_step
        self.isg_step = isg_step

        # Override base-class device (which is CPU)
        self.device = device
        self.model = self.model.to(self.device)

    def eval_step(self, data, dset_id=0) -> MutableMapping[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rgb, acc, depth, _ = render_image(
                self.model,
                self.occupancy_grids[0],
                0,
                data['rays_o'],
                data['rays_d'],
                timestamps=data['timestamps'],
                near_plane=data['near_far'][:, 0],
                far_plane=data['near_far'][:, 1],
                render_bkgd=data['color_bkgd'],
                cone_angle=self.cone_angle,
                render_step_size=self.model.step_size(self.render_n_samples),
                alpha_thresh=self.alpha_threshold,
                device=self.device,
            )
        return {
            "rgb": rgb,
            "depth": depth,
        }

    def step(self, data, do_update=True):
        imgs = data["imgs"].to(self.device)
        tstamps = data["timestamps"].to(self.device)

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # update occupancy grid
            self.occupancy_grids[0].every_n_step(
                step=self.global_step,
                occ_eval_fn=lambda x: self.model.query_opacity(
                    x, tstamps, self.train_datasets[0]
                ),
                occ_thre=self.density_threshold,
            )
            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                self.model,
                self.occupancy_grids[0],
                0,
                data["rays_o"],
                data["rays_d"],
                timestamps=tstamps,
                # rendering options
                near_plane=data['near_far'][:, 0],
                far_plane=data['near_far'][:, 1],
                render_bkgd=data["color_bkgd"],
                cone_angle=self.cone_angle,
                render_step_size=self.model.step_size(self.render_n_samples),
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
            self.train_datasets[0].update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0
            # compute loss and add regularizers
            recon_loss = self.criterion(rgb[alive_ray_mask], imgs[alive_ray_mask])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, grid_id=0)
                loss = loss + reg_loss
        if do_update:
            self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()

        # Report on losses
        if self.global_step % 5 == 0:
            with torch.no_grad():
                mse = F.mse_loss(rgb[alive_ray_mask], imgs[alive_ray_mask]).item()
                self.loss_info["psnr"].update(-10 * math.log10(mse))
                self.loss_info["mse"].update(mse)
                self.loss_info["alive_ray_mask"].update(float(alive_ray_mask.long().sum().item()))
                self.loss_info["n_rendering_samples"].update(float(n_rendering_samples))
                self.loss_info["n_rays"].update(float(len(imgs)))
                for r in self.regularizers:
                    r.report(self.loss_info)

        if do_update:
            self.gscaler.step(self.optimizer)
            scale = self.gscaler.get_scale()
            self.gscaler.update()

            if self.global_step == self.isg_step:
                self.train_datasets[0].enable_isg()
                raise StopIteration  # Whenever we change the dataset
            if self.global_step == self.ist_step:
                self.train_datasets[0].switch_isg2ist()
                raise StopIteration  # Whenever we change the dataset

            return scale <= self.gscaler.get_scale()
        return True

    def post_step(self, data, progress_bar):
        progress_bar.set_postfix_str(
            losses_to_postfix(self.loss_info, lr=self.lr), refresh=False)
        for loss_name, loss_val in self.loss_info.items():
            self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)
        progress_bar.update(1)

        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()

    def init_loss_info(self):
        ema_weight = 1.0
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.test_datasets[0]
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
            render_n_samples=self.render_n_samples,
            grid_config=kwargs.pop("grid_config"),
            global_scale=global_scale,
            global_translation=global_translation,
            **kwargs)
        logging.info(f"Initialized LowrankVideo model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        return model

    def init_regularizers(self, **kwargs):
        regularizers = [
            VideoPlaneTV(kwargs.get('plane_tv_weight', 0.0)),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0)),
            L1PlaneDensityVideo(kwargs.get('l1_plane_density_reg', 0.0)),
            L1AppearancePlanes(kwargs.get('l1_appearance_planes_reg', 0.0)),
        ]
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in regularizers if r.weight > 0]
        return regularizers

    def validate(self):
        val_metrics = []
        self.model.eval()
        with torch.no_grad():
            dataset = self.test_datasets[0]
            per_scene_metrics = {
                "psnr": 0, "ssim": 0
            }
            pred_frames = []
            pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
            for img_idx, data in enumerate(dataset):
                preds = self.eval_step(data, dset_id=0)
                out_metrics, out_img = self.evaluate_metrics(
                    data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None)
                pred_frames.append(out_img)
                per_scene_metrics["psnr"] += out_metrics["psnr"]
                per_scene_metrics["ssim"] += out_metrics["ssim"]
                pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                pb.update(1)
            pb.close()
            self.write_video_to_file(pred_frames)
            per_scene_metrics["psnr"] /= len(dataset)  # noqa
            per_scene_metrics["ssim"] /= len(dataset)  # noqa
            log_text = f"EPOCH {self.global_step}/{self.num_steps}"
            log_text += f" | PSNR: {per_scene_metrics['psnr']:.2f}"
            log_text += f" | SSIM: {per_scene_metrics['ssim']:.6f}"
            logging.info(log_text)
            val_metrics.append(per_scene_metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def write_video_to_file(self, frames):
        video_file = os.path.join(self.log_dir, f"step{self.global_step}.mp4")
        logging.info(f"Saving video ({len(frames)} frames) to {video_file}")
        height, width = frames[0].shape[:2]
        video = cv2.VideoWriter(
            video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for img in frames:
            video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()

    def evaluate_metrics(self, gt, preds: MutableMapping[str, torch.Tensor], dset,
                         img_idx, name=None, save_outputs: bool = True):
        preds_rgb = (
            preds["rgb"]
            .reshape(dset.img_h, dset.img_w, 3)
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
            depth = depth.cpu().reshape(dset.img_h, dset.img_w)[..., None]
            preds["depth"] = depth
            exrdict["depth"] = preds["depth"].numpy()

        if gt is not None:
            gt = gt.reshape(dset.intrinsics.height, dset.intrinsics.width, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            err = (gt - preds_rgb) ** 2
            exrdict["err"] = err.numpy()
            summary["mse"] = torch.mean(err)
            summary["psnr"] = metrics.psnr(preds_rgb, gt)
            summary["ssim"] = metrics.ssim(preds_rgb, gt)

        out_name = f"epoch{self.global_step}-{img_idx}"
        if name is not None and name != "":
            out_name += "-" + name

        out_img = preds_rgb
        if "depth" in preds:
            out_img = torch.cat((out_img, preds["depth"].expand_as(out_img)))
        out_img = (out_img * 255.0).byte().numpy()
        return summary, out_img


def losses_to_postfix(loss_dict: Dict[str, EMA], lr: Optional[float]) -> str:
    pfix = [f"{lname}={lval}" for lname, lval in loss_dict.items()]
    if lr is not None:
        pfix.append(f"lr={lr:.2e}")
    return "  ".join(pfix)


def init_dloader_random(worker_id):
    seed = torch.utils.data.get_worker_info().seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    if "lego" in data_dir:
        logging.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            batch_size=batch_size,
            max_cameras=kwargs.get('max_train_cameras'),
            max_tsteps=kwargs.get('max_train_tsteps') if keyframes else None,
            isg=isg, keyframes=keyframes, is_contracted=False, is_ndc=False
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
        logging.info(f"Loading contracted Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample, batch_size=batch_size,
            keyframes=keyframes, isg=isg, is_contracted=True, is_ndc=False
        )
        if ist:
            tr_dset.switch_isg2ist()  # this should only happen in case we're reloading
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=2,
        prefetch_factor=4, pin_memory=True, worker_init_fn=init_dloader_random)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


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


def load_video_model(config, state, validate_only):
    if state is not None:
        global_step = state['global_step']

        if len(config["upsample_time_steps"]) > 0:
            if global_step > config['upsample_time_steps'][0]:
                config.update(keyframes=False)
    data = load_data(**config, validate_only=validate_only)
    config.update(data)
    model = VideoTrainer(**config)

    if state is not None:
        init_hexplane_with_triplane = True
        if init_hexplane_with_triplane:
            keys = state['model'].keys()
            newdict = {}
            for key in keys:
                old_key = key
                if key in ("grids.0.2", "grids.1.2", "grids.2.2", "grids.3.2"):
                    key = key[:-1] + "3"
                newdict[key] = state['model'][old_key]
            newdict["resolution0"] = torch.tensor([640, 320, 160, 1708], device="cuda:0")
            state["model"] = newdict

        model.load_model(state)
    return model, config
