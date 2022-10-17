import logging
import math
import os
from collections import defaultdict
from typing import Dict, Optional, MutableMapping

import numpy as np
import pandas as pd

import torch
import torch.utils.data

from plenoxels.datasets.video_datasets import Video360Dataset, VideoLLFFDataset
from plenoxels.my_tqdm import tqdm
import cv2

from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.models.utils import compute_tv_norm
from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_png, write_exr


class VideoTrainer(Trainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.TensorDataset,
                 regnerf_weight_start: float,
                 regnerf_weight_end: float,
                 regnerf_weight_max_step: int,
                 num_epochs: int,
                 scheduler_type: Optional[str],
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_video: bool,  # TODO: Rationalize parameters for saving
                 save_outputs: bool,
                 upsample_time_resolution: [int],
                 upsample_time_steps: [int],
                 ist_step: int,
                 **kwargs
                 ):
        # Keys we wish to ignore
        kwargs.pop('transfer_learning', None)
        kwargs.pop('num_batches_per_dset', None)
        super().__init__(tr_loader=tr_loader,
                         tr_dsets=[tr_loader.dataset],
                         ts_dsets=[ts_dset],
                         regnerf_weight_start=regnerf_weight_start,
                         regnerf_weight_end=regnerf_weight_end,
                         regnerf_weight_max_step=regnerf_weight_max_step,
                         num_batches_per_dset=1,
                         num_epochs=num_epochs,
                         scheduler_type=scheduler_type,
                         optim_type=optim_type,
                         logdir=logdir,
                         expname=expname,
                         train_fp16=train_fp16,
                         save_every=save_every,
                         valid_every=valid_every,
                         transfer_learning=False,  # No transfer with video
                         save_outputs=save_outputs,
                         **kwargs)
        self.save_video = save_video
        self.upsample_time_resolution = upsample_time_resolution
        self.upsample_time_steps = upsample_time_steps
        self.ist_step = ist_step
        assert len(upsample_time_resolution) == len(upsample_time_steps)

    def eval_step(self, data, dset_id) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        batch_size = self.train_datasets[dset_id].batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].cuda()
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].cuda()
                timestamps_d_b = torch.ones(len(rays_o_b)).cuda() * timestamp
                if self.is_ndc:
                    bg_color = None
                else:
                    bg_color = 1
                outputs = self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=bg_color,
                                     channels={"rgb", "depth"})
                for k, v in outputs.items():
                    preds[k].append(v)
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def step(self, data, do_update=True):
        rays_o = data["rays_o"].cuda()
        rays_d = data["rays_d"].cuda()
        imgs = data["imgs"].cuda()
        timestamps = data["timestamps"].cuda()
        patch_rays_o, patch_rays_d, patch_timestamps = None, None, None
        if "patches" in data:
            patch_rays_o = data["patch_rays_o"].cuda()
            patch_rays_d = data["patch_rays_d"].cuda()
            patch_timestamps = data["patch_timestamps"]
            # Broadcast timestamps to match rays
            patch_timestamps = torch.ones(len(patch_timestamps), patch_rays_o.shape[1],
                                          patch_rays_o.shape[2]) * patch_timestamps[:, None, None]
            patch_timestamps = patch_timestamps.cuda()

        self.optimizer.zero_grad(set_to_none=True)

        C = imgs.shape[-1]
        # Random bg-color
        if C == 3:
            bg_color = 1
        else:
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(rays_o, rays_d, timestamps, bg_color=bg_color, channels={"rgb"})
            rgb_preds = fwd_out["rgb"]

            recon_loss = self.criterion(rgb_preds, imgs)

            tv = 0
            if patch_rays_o is not None:
                # Don't randomize bg-color when only interested in depth.
                patch_out = self.model(
                    patch_rays_o.reshape(-1, 3), patch_rays_d.reshape(-1, 3),
                    patch_timestamps.reshape(-1), bg_color=1, channels={"depth"})
                depths = patch_out["depth"].reshape(patch_rays_o.shape[:3])
                tv = compute_tv_norm(depths, weighting=None)
            loss = recon_loss + tv * self.cur_regnerf_weight

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        recon_loss_val = recon_loss.item()
        self.loss_info["mse"].update(recon_loss_val)
        self.loss_info["psnr"].update(-10 * math.log10(recon_loss_val))
        self.loss_info["tv"].update(tv.item() if patch_rays_o is not None else 0.0)

        if self.global_step in self.upsample_time_steps:
            # Upsample time resolution
            self.model.upsample_time(self.upsample_time_resolution[self.upsample_time_steps.index(self.global_step)])
            # Reset optimizer
            self.optimizer = self.init_optim(**self.extra_args)
            # After upsampling time, train with full time data
            if self.train_data_loader.dataset.keyframes:
                print(f'loading all the training frames')
                tr_dset = VideoLLFFDataset(self.train_data_loader.dataset.datadir, split='train', downsample=self.train_data_loader.dataset.downsample,
                                    keyframes=False, isg=self.train_data_loader.dataset.isg, ist=self.train_data_loader.dataset.ist, 
                                    extra_views=self.train_data_loader.dataset.extra_views, batch_size=self.train_data_loader.dataset.batch_size)
                self.train_data_loader = torch.utils.data.DataLoader(
                    tr_dset, batch_size=None, shuffle=True, num_workers=4,
                    prefetch_factor=4, pin_memory=True)
                self.train_datasets = [tr_dset]

        if self.global_step == self.ist_step:
            print(f'enabling IST importance sampling')
            tr_dset = VideoLLFFDataset(self.train_data_loader.dataset.datadir, split='train', downsample=self.train_data_loader.dataset.downsample,
                                keyframes=self.train_data_loader.dataset.keyframes, isg=False, ist=True, 
                                extra_views=self.train_data_loader.dataset.extra_views, batch_size=self.train_data_loader.dataset.batch_size)
            self.train_data_loader = torch.utils.data.DataLoader(
                tr_dset, batch_size=None, shuffle=True, num_workers=4,
                prefetch_factor=4, pin_memory=True)
            self.train_datasets = [tr_dset]

        return scale <= self.gscaler.get_scale()

    def post_step(self, data, progress_bar):
        if self.regnerf_weight_start > 0:
            w = np.clip(self.global_step / (1 if self.regnerf_weight_max_step < 1 else self.regnerf_weight_max_step), 0, 1)
            self.cur_regnerf_weight = self.regnerf_weight_start * (1 - w) + w * self.regnerf_weight_end
        self.writer.add_scalar(f"mse: ", self.loss_info["mse"].value, self.global_step)
        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info), refresh=False)
        progress_bar.update(1)

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.train_data_loader.dataset
        model = LowrankVideo(
            aabb=dset.scene_bbox,
            len_time=dset.len_time,
            is_ndc=dset.is_ndc,
            **kwargs)
        logging.info(f"Initialized LowrankVideo model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model

    def validate(self):
        val_metrics = []
        with torch.no_grad():
            for dset_id, dataset in enumerate(self.test_datasets):
                per_scene_metrics = {
                    "psnr": 0, "ssim": 0, "dset_id": dset_id,
                }

                pred_frames = []
                pb = tqdm(total=len(dataset), desc=f"Test scene {dset_id} ({dataset.name})")
                for img_idx, data in enumerate(dataset):
                    preds = self.eval_step(data, dset_id=dset_id)
                    out_metrics, out_img = self.evaluate_metrics(
                        data["imgs"], preds, dset_id=dset_id, dset=dataset, img_idx=img_idx, name=None)
                    pred_frames.append(out_img)
                    per_scene_metrics["psnr"] += out_metrics["psnr"]
                    per_scene_metrics["ssim"] += out_metrics["ssim"]
                    pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                    pb.update(1)
                pb.close()
                if self.save_video:
                    self.write_video_to_file(pred_frames, dataset)
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"EPOCH {self.epoch}/{self.num_epochs} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_epoch{self.epoch}.csv"))

    def write_video_to_file(self, frames, dataset):
        video_file = os.path.join(self.log_dir, f"epoch{self.epoch}.mp4")
        logging.info(f"Saving video ({len(frames)} frames) to {video_file}")
        height, width = frames[0].shape[:2]
        video = cv2.VideoWriter(
            video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for img in frames:
            video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()

    def evaluate_metrics(self, gt, preds: MutableMapping[str, torch.Tensor], dset, dset_id,
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

        out_name = f"epoch{self.epoch}-D{dset_id}-{img_idx}"
        if name is not None and name != "":
            out_name += "-" + name

        out_img = preds_rgb
        if "depth" in preds:
            out_img = torch.cat((out_img, preds["depth"].expand_as(out_img)))
        out_img = (out_img * 255.0).byte().numpy()

        if not self.save_video:
            write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
            write_png(os.path.join(self.log_dir, out_name + ".png"), out_img)

        return summary, out_img


def losses_to_postfix(loss_dict: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())


def load_data(data_downsample, data_dirs, batch_size, **kwargs):
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]
    regnerf_bool = kwargs.get('regnerf_weight_start') > 0

    if "lego" in data_dir:
        logging.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            resolution=None,  # Don't use resolution for low-pass filtering any more
            max_cameras=kwargs.get('max_train_cameras'), max_tsteps=kwargs.get('max_train_tsteps'),
            extra_views=regnerf_bool, batch_size=batch_size)
        ts_dset = Video360Dataset(
            data_dir, split='test', downsample=1, resolution=None,
            max_cameras=kwargs.get('max_test_cameras'), max_tsteps=kwargs.get('max_test_tsteps'),
            extra_views=False, batch_size=batch_size)
    else:
        # For LLFF we downsample both train and test unlike 360.
        # For LLFF the test-set is not time-subsampled!
        logging.info(f"Loading VideoLLFFDataset with downsample={data_downsample}")
        tr_dset = VideoLLFFDataset(data_dir, split='train', downsample=data_downsample,
                                   keyframes=kwargs.get('keyframes'), isg=kwargs.get('isg'),  # Always start without ist
                                   extra_views=regnerf_bool, batch_size=batch_size)
        ts_dset = VideoLLFFDataset(data_dir, split='test', downsample=data_downsample,
                                   keyframes=False, extra_views=False, batch_size=batch_size)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, shuffle=True, num_workers=4,
        prefetch_factor=4, pin_memory=True)
    return {"tr_loader": tr_loader, "ts_dset": ts_dset}
