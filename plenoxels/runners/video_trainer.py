import gc
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

from plenoxels.datasets.video_datasets import Video360Dataset, VideoLLFFDataset
from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.runners.utils import render_image


class VideoTrainer(Trainer):
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
                 upsample_time_resolution: [int],
                 upsample_time_steps: [int],
                 ist_step: int,
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
        self.upsample_time_resolution = upsample_time_resolution
        self.upsample_time_steps = upsample_time_steps
        # Override base-class device (which is CPU)
        self.device = device
        self.model.to(device)
        self.ist_step = ist_step
        self.plane_tv_weight = kwargs['plane_tv_weight']
        assert len(upsample_time_resolution) == len(upsample_time_steps)

    def eval_step(self, data, dset_id=0) -> MutableMapping[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rgb, acc, depth, _ = render_image(
                self.model,
                self.occupancy_grids[0],
                0,
                data['rays_o'],
                data['rays_d'],
                timestamps=data['timestamps'],
                near_plane=self.train_datasets[0].near,
                far_plane=self.train_datasets[0].far,
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
        imgs = data["imgs"]

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            # update occupancy grid
            self.occupancy_grids[0].every_n_step(
                step=self.global_step,
                occ_eval_fn=lambda x: self.model.query_opacity(
                    x, data["timestamps"], self.train_datasets[0]
                ),
                occ_thre=self.cur_density_threshold(),
            )
            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                self.model,
                self.occupancy_grids[0],
                0,
                data["rays_o"],
                data["rays_d"],
                timestamps=data["timestamps"],
                # rendering options
                near_plane=self.train_datasets[0].near,
                far_plane=self.train_datasets[0].far,
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
            num_rays = min(self.cur_max_rays(), int(
                num_rays
                * (self.target_sample_batch_size / float(n_rendering_samples))
            ))
            self.train_datasets[0].update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0
            # compute loss and add regularizers
            loss = self.criterion(rgb[alive_ray_mask], imgs[alive_ray_mask])
            plane_tv = None
            if self.plane_tv_weight > 0:
                plane_tv = self.model.compute_plane_tv() * self.plane_tv_weight
                loss = loss + plane_tv

        if do_update:
            self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()

        # Report on losses
        if self.global_step % 30 == 0:
            with torch.no_grad():
                mse = F.mse_loss(rgb[alive_ray_mask], imgs[alive_ray_mask]).item()
                self.loss_info["psnr"].update(-10 * math.log10(mse))
                self.loss_info["mse"].update(mse)
                self.loss_info["alive_ray_mask"].update(float(alive_ray_mask.long().sum().item()))
                self.loss_info["n_rendering_samples"].update(float(n_rendering_samples))
                self.loss_info["n_rays"].update(float(len(imgs)))
                if plane_tv is not None:
                    self.loss_info["plane_tv"].update(plane_tv.item())

        if do_update:
            self.gscaler.step(self.optimizer)
            scale = self.gscaler.get_scale()
            self.gscaler.update()

            if self.global_step in self.upsample_time_steps:
                print()
                # Upsample time resolution
                self.model.upsample_time(self.upsample_time_resolution[self.upsample_time_steps.index(self.global_step)])
                # Reset optimizer
                self.optimizer = self.init_optim(**self.extra_args)
                # After upsampling time, train with full time data
                if self.train_datasets[0].use_keyframes:
                    logging.info('Loading all the training frames. Will now train on full data.')
                    self.train_datasets[0] = reload_dset_no_keyframes(self.train_datasets[0])
                    self.train_data_loader = torch.utils.data.DataLoader(
                        self.train_datasets[0], batch_size=None, num_workers=4,
                        prefetch_factor=4, pin_memory=True)

            if self.global_step == self.ist_step:
                print(f'deleting pre-ist training data to avoid OOM')
                datadir = self.train_data_loader.dataset.datadir
                downsample = self.train_data_loader.dataset.downsample
                keyframes = self.train_data_loader.dataset.keyframes
                extra_views = self.train_data_loader.dataset.extra_views
                batch_size = self.train_data_loader.dataset.batch_size
                del self.train_data_loader, self.train_datasets
                gc.collect()
                print(f'reloading training data with IST importance sampling')
                tr_dset = VideoLLFFDataset(datadir, split='train', downsample=downsample,
                                    keyframes=keyframes, isg=False, ist=True,
                                    extra_views=extra_views, batch_size=batch_size)
                self.train_data_loader = torch.utils.data.DataLoader(
                    tr_dset, batch_size=None, shuffle=True, num_workers=4,
                    prefetch_factor=4, pin_memory=True)
                self.train_datasets = [tr_dset]
            return scale <= self.gscaler.get_scale()
        return True

    def post_step(self, data, progress_bar):
        if self.global_step % 10 == 0:
            self.writer.add_scalar(f"mse: ", self.loss_info["mse"].value, self.global_step)
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info), refresh=False)
        progress_bar.update(1)

        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.train_data_loader.dataset
        model = LowrankVideo(
            aabb=dset.scene_bbox,
            len_time=dset.len_time,
            is_ndc=dset.is_ndc,
            render_n_samples=self.render_n_samples,
            grid_config=kwargs.pop("grid_config"),
            **kwargs)
        logging.info(f"Initialized LowrankVideo model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        return model

    def validate(self):
        val_metrics = []
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
        # if not self.save_video:
        #     write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
        #     write_png(os.path.join(self.log_dir, out_name + ".png"), out_img)

        return summary, out_img


def losses_to_postfix(loss_dict: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())


def reload_dset_no_keyframes(dset):
    if isinstance(dset, Video360Dataset):
        new_dset = Video360Dataset(
            datadir=dset.datadir, split=dset.split, color_bkgd_aug=dset.color_bkgd_aug,
            batch_size=dset.batch_size, generator=dset.generator, downsample=dset.downsample,
            max_cameras=dset.max_cameras, max_tsteps=None, isg=False, ist=True,
        )
    elif isinstance(dset, VideoLLFFDataset):
        new_dset = VideoLLFFDataset(
            datadir=dset.datadir, split=dset.split, keyframes=False,
            batch_size=dset.batch_size, generator=dset.generator, downsample=dset.downsample,
            isg=False, ist=True,
        )
    else:
        raise ValueError(dset)
    return new_dset


def load_data(data_downsample, data_dirs, batch_size, **kwargs):
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]

    keyframes = kwargs.get('keyframes')

    if "lego" in data_dir:
        logging.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', color_bkgd_aug='white', batch_size=1024, generator=None,
            downsample=data_downsample,
            max_cameras=kwargs.get('max_train_cameras'),
            max_tsteps=kwargs.get('max_train_tsteps') if keyframes else None,
            isg=kwargs.get('isg'),
        )
        ts_dset = Video360Dataset(
            data_dir, split='test', color_bkgd_aug='white', batch_size=None, generator=None,
            downsample=1,
            max_cameras=kwargs.get('max_test_cameras'),
            max_tsteps=kwargs.get('max_test_tsteps'))
    else:
        # For LLFF we downsample both train and test unlike 360.
        # For LLFF the test-set is not time-subsampled!
        logging.info(f"Loading VideoLLFFDataset with downsample={data_downsample}")
        tr_dset = VideoLLFFDataset(data_dir, split='train', downsample=data_downsample,
                                   keyframes=kwargs.get('keyframes'), isg=kwargs.get('isg'),  # Always start without ist
                                   batch_size=batch_size)
        ts_dset = VideoLLFFDataset(data_dir, split='test', downsample=4,
                                   keyframes=False, batch_size=batch_size)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,
        prefetch_factor=4, pin_memory=True)
    return {"tr_loader": tr_loader, "ts_dset": ts_dset}
