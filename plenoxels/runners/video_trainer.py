import gc
import logging
import math
import os
from collections import defaultdict
from copy import copy
from typing import Dict, Optional, MutableMapping

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from plenoxels.datasets.video_datasets import Video360Dataset, VideoLLFFDataset
from plenoxels.datasets.photo_tourism_dataset import PhotoTourismDataset

from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.runners.multiscene_trainer import Trainer


class VideoTrainer(Trainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 scheduler_type: Optional[str],
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
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
                        #  regnerf_weight_start=0.0,   # regnerf useless
                        #  regnerf_weight_end=0.0,     # regnerf useless
                        #  regnerf_weight_max_step=0,  # regnerf useless
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
                         **kwargs)
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
            
            if "bounds" in data:
                near = data["bounds"][0].cuda()
                far = data["bounds"][1].cuda()
            else:
                near = None
                far = None
            
            if rays_d.ndim == 3:
                rays_d = rays_d.squeeze()
                rays_o = rays_o.squeeze()
                timestamp = timestamp.squeeze()
                
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].cuda()
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].cuda()
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).cuda()
                if self.is_ndc:
                    bg_color = None
                else:
                    bg_color = 1
                outputs = self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=bg_color,
                                     channels={"rgb", "depth"}, near=near, far=far)
                for k, v in outputs.items():
                    preds[k].append(v)
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def step(self, data, do_update=True):
        rays_o = data["rays_o"].cuda()
        rays_d = data["rays_d"].cuda()
        imgs = data["imgs"].cuda()
        timestamps = data["timestamps"].cuda()
        self.optimizer.zero_grad(set_to_none=True)
        
        # because the dataloader for phototourism is a bit different 
        # we have a batch dimension (always 1)
        if rays_d.ndim == 3:
            rays_d = rays_d.squeeze()
            rays_o = rays_o.squeeze()
            timestamps = timestamps.squeeze()
            imgs = imgs.squeeze()
            
        if "bounds" in data:
            near = data["bounds"][:, 0].cuda()
            far = data["bounds"][:, 1].cuda()
        else:
            near = None
            far = None

        C = imgs.shape[-1]
        # Random bg-color
        if C == 3:
            bg_color = 1
        else:
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(rays_o, rays_d, timestamps, bg_color=bg_color, channels={"rgb"} , near=near, far=far)
            rgb_preds = fwd_out["rgb"]
            recon_loss = self.criterion(rgb_preds, imgs)
            loss = recon_loss
            plane_tv = None
            if self.plane_tv_weight > 0:
                plane_tv = self.model.compute_plane_tv()
                loss = loss + plane_tv * self.plane_tv_weight

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        recon_loss_val = recon_loss.item()
        self.loss_info["mse"].update(recon_loss_val)
        self.loss_info["psnr"].update(-10 * math.log10(recon_loss_val))
        if plane_tv is not None:
            self.loss_info["plane_tv"].update(plane_tv.item() if self.plane_tv_weight > 0 else 0.0)

        if self.global_step in self.upsample_time_steps:
            # Upsample time resolution
            self.model.upsample_time(self.upsample_time_resolution[self.upsample_time_steps.index(self.global_step)])
            # Reset optimizer
            self.optimizer = self.init_optim(**self.extra_args)
            # After upsampling time, train with full time data
            if self.train_datasets[0].keyframes:
                data_dir = self.train_datasets[0].datadir
                del self.train_data_loader, self.train_datasets
                gc.collect()
                self.extra_args.update(keyframes=False)
                tr_dd = init_tr_data(data_dir=data_dir, **self.extra_args)
                self.train_data_loader = tr_dd['tr_loader']
                self.train_datasets = [tr_dd['tr_dset']]
                raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_datasets[0].switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale <= self.gscaler.get_scale()

    def post_step(self, data, progress_bar):
        self.writer.add_scalar(f"mse: ", self.loss_info["mse"].value, self.global_step)
        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info), refresh=False)
        progress_bar.update(1)

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.train_datasets[0]
        model = LowrankVideo(
            aabb=dset.scene_bbox,
            len_time=dset.len_time,
            is_ndc=dset.is_ndc,
            is_contracted=dset.is_contracted,
            lookup_time=dset.lookup_time,
            **kwargs)
        logging.info(f"Initialized LowrankVideo model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, using ndc {dset.is_ndc} and contraction {dset.is_contracted}.")
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
                    
                    imgs = data["imgs"]
                    if imgs.ndim == 3:
                        imgs = imgs.squeeze()
                    
                    out_metrics, out_img = self.evaluate_metrics(
                        imgs, preds, dset_id=dset_id, dset=dataset, img_idx=img_idx, name=None)
                    pred_frames.append(out_img)
                    per_scene_metrics["psnr"] += out_metrics["psnr"]
                    per_scene_metrics["ssim"] += out_metrics["ssim"]
                    pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                    pb.update(1)
                pb.close()
                self.write_video_to_file(pred_frames, dataset)
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"step {self.global_step}/{self.num_steps} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def write_video_to_file(self, frames, dataset):
        video_file = os.path.join(self.log_dir, f"step{self.global_step}.mp4")
        logging.info(f"Saving video ({len(frames)} frames) to {video_file}")
        
        # Photo tourisme the image sizes differs
        sizes = np.array([frame.shape[:2] for frame in frames])
        same_size_frames = np.unique(sizes, axis=0).shape[0] == 1
        if same_size_frames:
            height, width = frames[0].shape[:2]
            video = cv2.VideoWriter(
                video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            for img in frames:
                video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
            cv2.destroyAllWindows()
            video.release()
        else:
            height = sizes[:,0].max()
            width = sizes[:, 1].max()
            video = cv2.VideoWriter(
                video_file, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
            for img in frames:
                image = np.zeros((height, width, 3), dtype=np.uint8)
                h, w = img.shape[:2]
                image[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w, :] = img
                video.write(image[:, :, ::-1])  # opencv uses BGR instead of RGB
            cv2.destroyAllWindows()
            video.release() 
            
            
    def evaluate_metrics(self, gt, preds: MutableMapping[str, torch.Tensor], dset, dset_id,
                         img_idx, name=None, save_outputs: bool = True):
        
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w  
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
            
        preds_rgb = preds["rgb"].reshape(img_h, img_w, 3).cpu()
        exrdict = {
            "preds": preds_rgb.numpy(),
        }
        summary = dict()

        if "depth" in preds:
            # normalize depth and add to exrdict
            depth = preds["depth"]
            depth = depth - depth.min()
            depth = depth / depth.max()
            depth = depth.cpu().reshape(img_h, img_w)[..., None]
            preds["depth"] = depth
            exrdict["depth"] = preds["depth"].numpy()

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            err = (gt - preds_rgb) ** 2
            exrdict["err"] = err.numpy()
            summary["mse"] = torch.mean(err)
            summary["psnr"] = metrics.psnr(preds_rgb, gt)
            summary["ssim"] = metrics.ssim(preds_rgb, gt)

        out_name = f"step{self.global_step}-D{dset_id}-{img_idx}"
        if name is not None and name != "":
            out_name += "-" + name

        out_img = preds_rgb
        if "depth" in preds:
            out_img = torch.cat((out_img, preds["depth"].expand_as(out_img)))
        out_img = (out_img * 255.0).byte().numpy()

        return summary, out_img


def losses_to_postfix(loss_dict: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())


def init_dloader_random(worker_id):
    seed = torch.utils.data.get_worker_info().seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    if "lego" in data_dir:
        logging.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            batch_size=batch_size,
            max_cameras=kwargs.get('max_train_cameras'),
            max_tsteps=kwargs.get('max_train_tsteps') if keyframes else None,
            isg=isg,
        )
    elif "sacre" in data_dir or "notre" in data_dir or "brandenburg" in data_dir or "trevi" in data_dir:
        logging.info(f"About to load Phototourism data downsampled by {data_downsample} times.")
        
        tr_dset = PhotoTourismDataset(data_dir, split='train', downsample=data_downsample, batch_size=batch_size)        
        tr_loader = torch.utils.data.DataLoader(tr_dset, batch_size=1, shuffle=True)
        
        return {"ts_dset" : tr_dset, "tr_loader": tr_loader}
    else:
        logging.info(f"Loading contracted Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            batch_size=batch_size,
            keyframes=keyframes,
            isg=isg,  
            is_contracted=True,
        )
        # For LLFF we downsample both train and test unlike 360.
        # For LLFF the test-set is not time-subsampled!
        # logging.info(f"Loading VideoLLFFDataset with downsample={data_downsample}")
        # tr_dset = VideoLLFFDataset(
        #     data_dir, split='train', downsample=data_downsample,
        #     batch_size=batch_size,
        #     keyframes=keyframes,
        #     isg=isg,  # Always start without ist
        #     ist=ist,
        # )
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
        )
    elif "sacre" in data_dir or "notre" in data_dir or "brandenburg" in data_dir or "trevi" in data_dir:        
        ts_dset = PhotoTourismDataset(data_dir, split='test', downsample=4)
    else:
        ts_dset = Video360Dataset(
            data_dir, split='test', downsample=4, keyframes=False, is_contracted=True
        )
        # ts_dset = VideoLLFFDataset(
        #     data_dir, split='test', downsample=4, keyframes=False
        # )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, **kwargs):
    assert len(data_dirs) == 1
    od = init_tr_data(data_downsample, data_dirs[0], **kwargs)
    od.update(init_ts_data(data_dirs[0], **kwargs))
    return od
