import logging
import math
import os
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from ..datasets.video_datasets import Video360Dataset, VideoLLFFDataset
from ..my_tqdm import tqdm
import cv2

from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from .multiscene_trainer import Trainer
from ..ops.image import metrics
from ..ops.image.io import write_png, write_exr
from plenoxels.datasets.patchloader import PatchLoader


class VideoTrainer(Trainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.Dataset,
                 patch_loader: PatchLoader,
                 regnerf_weight: float,
                 num_epochs: int,
                 scheduler_type: Optional[str],
                 model_type: str,
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_video: bool,  # TODO: Rationalize parameters for saving
                 save_outputs: bool,
                 **kwargs
                 ):
        # Keys we wish to ignore
        kwargs.pop('transfer_learning', None)
        kwargs.pop('num_batches_per_dset', None)
        super().__init__(tr_loaders=[tr_loader],
                         ts_dsets=[ts_dset],
                         num_batches_per_dset=1,
                         num_epochs=num_epochs,
                         scheduler_type=scheduler_type,
                         model_type=model_type,
                         optim_type=optim_type,
                         logdir=logdir,
                         expname=expname,
                         train_fp16=train_fp16,
                         save_every=save_every,
                         valid_every=valid_every,
                         transfer_learning=False,  # No transfer with video
                         save_outputs=save_outputs,
                         **kwargs)
        self.patch_loader = patch_loader
        self.regnerf_weight = regnerf_weight
        if self.regnerf_weight > 0:
            assert self.patch_loader is not None
        self.save_video = save_video

    def eval_step(self, data, dset_id) -> torch.Tensor:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data[0]
            rays_d = data[1]
            timestamp = data[3]
            # timestamp = 0
            preds = []
            for b in range(math.ceil(rays_o.shape[0] / self.batch_size)):
                rays_o_b = rays_o[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rays_d_b = rays_d[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                timestamps_d_b = torch.ones(len(rays_o_b)).cuda() * timestamp
                preds.append(self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=1)[0])
            preds = torch.cat(preds, 0)
        return preds

    def step(self, data, dset_id):
        rays_o = data["train"][0].cuda()
        rays_d = data["train"][1].cuda()
        imgs = data["train"][2].cuda()
        timestamps = data["train"][3].cuda()
        patch_rays_o, patch_rays_d, patch_timestamps = None, None, None
        if "patches" in data:
            patch_rays_o = data["patches"][0].cuda()
            patch_rays_d = data["patches"][1].cuda()
            patch_timestamps = data["patches"][2]
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
            rgb_preds, _ = self.model(rays_o, rays_d, timestamps, bg_color=bg_color)
            tv = 0
            if patch_rays_o is not None:
                _, depths = self.model(patch_rays_o.reshape(-1, 3), patch_rays_d.reshape(-1, 3),
                                       patch_timestamps.reshape(-1), bg_color=bg_color)
                depths = depths.reshape(patch_rays_o.shape[0], patch_rays_o.shape[1],
                                        patch_rays_o.shape[2])
                tv = compute_tv_norm(depths)
            recon_loss = self.criterion(rgb_preds, imgs)
            loss = recon_loss + tv * self.regnerf_weight

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        recon_loss_val = recon_loss.item()
        self.loss_info["mse"].update(recon_loss_val)
        self.loss_info["psnr"].update(-10 * math.log10(recon_loss_val))
        self.loss_info["tv"].update(tv.item() if patch_rays_o is not None else 0.0)
        return scale <= self.gscaler.get_scale()

    def post_step(self, dset_id, progress_bar):
        self.writer.add_scalar(f"mse: ", self.loss_info["mse"].value, self.global_step)
        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info), refresh=False)
        progress_bar.update(1)

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.train_data_loaders[0].dataset
        if self.model_type == "learnable_hash":
            model = LowrankVideo(
                aabb=dset.scene_bbox,
                len_time=dset.len_time,
                is_ndc=dset.is_ndc,
                **kwargs)
        else:
            raise ValueError(f"Model type {self.model_type} invalid")
        logging.info(f"Initialized model of type {self.model_type} with "
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
                    gt = data[2]
                    out_metrics, out_img = self.evaluate_metrics(
                        gt, preds, dset_id=dset_id, dset=dataset, img_idx=img_idx, name=None)
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
        logging.info(f"Saving video to {video_file}")
        video = cv2.VideoWriter(
            video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30,
            (2 * dataset.intrinsics.width, dataset.intrinsics.height))
        for img in frames:
            video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()

    def evaluate_metrics(self, gt, preds: torch.Tensor, dset, dset_id, img_idx, name=None,
                         save_outputs: bool = True):
        gt = gt.reshape(dset.intrinsics.height, dset.intrinsics.width, -1).cpu()
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
        preds = preds.reshape(dset.intrinsics.height, dset.intrinsics.width, 3).cpu()
        err = (gt - preds) ** 2
        exrdict = {
            "gt": gt.numpy(),
            "preds": preds.numpy(),
            "err": err.numpy(),
        }
        summary = {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds, gt),
        }

        out_name = f"epoch{self.epoch}-D{dset_id}-{img_idx}"
        if name is not None and name != "":
            out_name += "-" + name

        out_img = (preds * 255.0).byte().numpy()
        if not self.save_video:
            write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
            write_png(os.path.join(self.log_dir, out_name + ".png"), out_img)

        return summary, out_img

    def reset_data_iterators(self, dataset_idx=None):
        """Rewind the iterator for the new epoch.
        Since we have an infinite iterable for patches and a finite iterable for the
        train-data-loader, we zip the two and give each a key ("train" or "patches")
        """

        def imerge_wkeys(patches, train):
            if patches is None:
                for i in train:
                    yield {"train": i}
            for i, j in zip(train, patches):
                yield {"train": i, "patches": j}

        if dataset_idx is None:
            # We always have a single train_data_loader, hence this works!
            patch_iter = iter(self.patch_loader) if self.patch_loader is not None else None
            self.train_iterators = [
                imerge_wkeys(patch_iter, iter(dloader)) for dloader in self.train_data_loaders
            ]
        else:
            self.train_iterators[dataset_idx] = imerge_wkeys(
                patches=iter(self.patch_loader) if self.patch_loader is not None else None,
                train=iter(self.train_data_loaders[dataset_idx]))


def losses_to_postfix(loss_dict: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())


# Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/math.py#L237
def compute_tv_norm(depths, losstype='l1'):
    # depths [n_patches, h, w]
    v00 = depths[:, :-1, :-1]
    v01 = depths[:, :-1, 1:]
    v10 = depths[:, 1:, :-1]

    if losstype == 'l2':
        loss = torch.sqrt(((v00 - v01) ** 2) + ((v00 - v10) ** 2))
    elif losstype == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')

    return torch.mean(loss)


def load_data(data_downsample, data_dirs, batch_size, **kwargs):
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]
    regnerf_weight = kwargs.get('regnerf_weight')

    if "lego" in data_dir:
        logging.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            resolution=None,  # Don't use resolution for low-pass filtering any more
            max_cameras=kwargs.get('max_train_cameras'), max_tsteps=kwargs.get('max_train_tsteps'),
            extra_views=regnerf_weight > 0)
        ts_dset = Video360Dataset(
            data_dir, split='test', downsample=1, resolution=None,
            max_cameras=kwargs.get('max_test_cameras'), max_tsteps=kwargs.get('max_test_tsteps'),
            extra_views=False)
    else:
        # For LLFF we downsample both train and test unlike 360.
        # For LLFF the test-set is not time-subsampled!
        logging.info(f"Loading VideoLLFFDataset with downsample={data_downsample}")
        tr_dset = VideoLLFFDataset(data_dir, split='train', downsample=data_downsample,
                                   subsample_time=kwargs.get('subsample_time'),
                                   extra_views=regnerf_weight > 0)
        ts_dset = VideoLLFFDataset(data_dir, split='test', downsample=data_downsample,
                                   subsample_time=1.0, extra_views=False)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=batch_size, shuffle=True, num_workers=3,
        prefetch_factor=4, pin_memory=True)
    patch_loader = None
    if regnerf_weight > 0:
        patch_loader = PatchLoader(
            rays_o=tr_dset.extra_rays_o, rays_d=tr_dset.extra_rays_d, len_time=tr_dset.len_time)
    return {"tr_loaders": [tr_loader], "ts_dsets": [ts_dset], "patch_loader": patch_loader}
