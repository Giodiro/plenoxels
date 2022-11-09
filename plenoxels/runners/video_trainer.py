import gc
import logging
import math
import os
from collections import defaultdict
from typing import Dict, Optional, MutableMapping

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from plenoxels.datasets.photo_tourism import PhotoTourismDataset
from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.ema import EMA
from plenoxels.models.lowrank_appearance import LowrankAppearance
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.multiscene_trainer import Trainer, visualize_planes, visualize_planes_withF
from plenoxels.runners.regularization import VideoPlaneTV, TimeSmoothness, HistogramLoss, L1PlaneDensityVideo, L1AppearancePlanes
import matplotlib.pyplot as plt

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
                 add_rank_steps: [int],
                 isg_step: int,
                 ist_step: int,
                 **kwargs
                 ):
        # Keys we wish to ignore
        kwargs.pop('transfer_learning', None)
        kwargs.pop('num_batches_per_dset', None)
        tr_dset = tr_loader.dataset if tr_loader is not None else None
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
                         **kwargs)
        self.upsample_time_resolution = upsample_time_resolution
        self.upsample_time_steps = upsample_time_steps
        self.add_rank_steps = add_rank_steps
        self.ist_step = ist_step
        self.isg_step = isg_step
        assert len(upsample_time_resolution) == len(upsample_time_steps)

    def eval_step(self, data, dset_id) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            near_far = data["near_far"].cuda() if data["near_far"] is not None else None
            timestamp = data["timestamps"]
            # import pdb; pdb.set_trace()
            if rays_o.ndim == 3:
                rays_o = rays_o.squeeze(0)
                rays_d = rays_d.squeeze(0)

            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].cuda()
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].cuda()
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).cuda()
                if self.is_ndc:
                    bg_color = None
                else:
                    bg_color = torch.tensor([1.0, 1.0, 1.0], device=rays_o_b.device)
                outputs = self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=bg_color,
                                     channels={"rgb", "depth", "proposal_depth"}, near_far=near_far)
                for k, v in outputs.items():
                    if not isinstance(v, list):
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def step(self, data, do_update=True):
        rays_o = data["rays_o"].cuda()
        rays_d = data["rays_d"].cuda()
        imgs = data["imgs"].cuda()
        near_far = data["near_far"].cuda() if data["near_far"] is not None else None
        timestamps = data["timestamps"].cuda()

        # import pdb; pdb.set_trace()
        if rays_o.ndim == 3:
            rays_o = rays_o.squeeze(0)
            rays_d = rays_d.squeeze(0)
            near_far = near_far.squeeze(0) if near_far is not None else None
            timestamps = timestamps.squeeze(0)
            imgs = imgs.squeeze(0)

        self.optimizer.zero_grad(set_to_none=True)

        C = imgs.shape[-1]
        # Random bg-color
        if C == 3:
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=rays_o.device)
        else:
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(rays_o, rays_d, timestamps, bg_color=bg_color, channels={"rgb"}, near_far=near_far)
            rgb_preds = fwd_out["rgb"]
            # Reconstruction loss
            recon_loss = self.criterion(rgb_preds, imgs)
            loss = recon_loss
            # Regularization
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, grid_id=0, model_out=fwd_out)
                loss = loss + reg_loss

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        recon_loss_val = recon_loss.item()
        self.loss_info["mse"].update(recon_loss_val)
        self.loss_info["psnr"].update(-10 * math.log10(recon_loss_val))
        for r in self.regularizers:
            r.report(self.loss_info)

        if self.model.trainable_rank is not None and self.global_step in self.add_rank_steps:
            self.model.trainable_rank = self.model.trainable_rank + 1
            self.model.update_trainable_rank()
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
                # Reload the test dataset also, since it should not use keyframes once the train set is full-time
                ts_dset = Video360Dataset(
                    data_dir, split='test', downsample=4, keyframes=False, is_contracted=self.test_datasets[0].is_contracted
                )
                self.test_datasets = [ts_dset]
                raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.isg_step:
            self.train_datasets[0].enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_datasets[0].switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale <= self.gscaler.get_scale()

    def post_step(self, data, progress_bar):
        progress_bar.set_postfix_str(
            losses_to_postfix(self.loss_info, lr=self.cur_lr()), refresh=False)
        for loss_name, loss_val in self.loss_info.items():
            self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)
        progress_bar.update(1)

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info: Dict[str, EMA] = defaultdict(lambda: EMA(ema_weight))

    def init_model(self, **kwargs) -> torch.nn.Module:
        dset = self.test_datasets[0]
        data_dir = dset.datadir
        try:
            global_translation = dset.global_translation
        except AttributeError:
            global_translation = None
        try:
            global_scale = dset.global_scale
        except AttributeError:
            global_scale = None
        if "sacre" in data_dir or "trevi" in data_dir:
            model = LowrankAppearance(
                aabb=dset.scene_bbox,
                len_time=dset.len_time,
                is_ndc=self.is_ndc,
                is_contracted=self.is_contracted,
                lookup_time=dset.lookup_time,
                proposal_sampling=self.extra_args.get('histogram_loss_weight', 0.0) > 0.0,
                global_scale=global_scale,
                global_translation=global_translation,
                **kwargs)
        else:
            model = LowrankVideo(
                aabb=dset.scene_bbox,
                len_time=dset.len_time,
                is_ndc=self.is_ndc,
                is_contracted=self.is_contracted,
                lookup_time=dset.lookup_time,
                proposal_sampling=self.extra_args.get('histogram_loss_weight', 0.0) > 0.0,
                global_scale=global_scale,
                global_translation=global_translation,
                **kwargs)
        logging.info(f"Initialized {model.__class__} model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
                     f"using ndc {self.is_ndc} and contraction {self.is_contracted}.")
        model.cuda()
        return model

    def init_regularizers(self, **kwargs):
        regularizers = [
            VideoPlaneTV(kwargs.get('plane_tv_weight', 0.0)),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0)),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            L1PlaneDensityVideo(kwargs.get('l1_plane_density_reg', 0.0)),
            L1AppearancePlanes(kwargs.get('l1_appearance_planes_reg', 0.0)),
        ]
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in regularizers if r.weight > 0]
        return regularizers

    def optimize_appearance_step(self, data, batch_size, im_id):
        rays_o = data["rays_o_left"]
        rays_d = data["rays_d_left"]
        imgs = data["imgs_left"]
        near_far = data["near_far"].cuda() if data["near_far"] is not None else None
        timestamp = data["timestamps"]

        if rays_o.ndim == 3:
            rays_o = rays_o.squeeze(0)
            rays_d = rays_d.squeeze(0)
            near_far = near_far.squeeze(0) if near_far is not None else None
            timestamp = timestamp.squeeze(0)
            imgs = imgs.squeeze(0)

        # here we shuffle the rays to make optimization more stable
        n_steps = math.ceil(rays_o.shape[0] / batch_size)
        epochs = 10
        for n in range(epochs):
            idx = torch.randperm(rays_o.shape[0])
            for b in range(n_steps):
                rays_o_b = rays_o[idx[b * batch_size: (b + 1) * batch_size]].cuda()
                rays_d_b = rays_d[idx[b * batch_size: (b + 1) * batch_size]].cuda()
                imgs_b  = imgs[idx[b * batch_size: (b + 1) * batch_size]].cuda()
                timestamps_b = timestamp.expand(rays_o_b.shape[0]).cuda()
                near_far_b = near_far.expand(rays_o_b.shape[0], 2).cuda()

                with torch.cuda.amp.autocast(enabled=self.train_fp16):
                    fwd_out = self.model(rays_o_b, rays_d_b, timestamps_b, bg_color=1,
                                        channels={"rgb"}, near_far=near_far_b)
                    rgb_preds = fwd_out["rgb"]
                    recon_loss = self.criterion(rgb_preds, imgs_b)
                    loss = recon_loss

                    self.gscaler.scale(loss).backward()
                    self.gscaler.step(self.appearance_optimizer)
                    scale = self.gscaler.get_scale()
                    self.gscaler.update()

                    self.appearance_optimizer.zero_grad(set_to_none=True)

                    self.writer.add_scalar(f"appearance_loss_{self.global_step}/recon_loss_{im_id}", recon_loss.item(), b + n * n_steps)

    def optimize_appearance_codes(self):
        # turn gradients off for anything but appearance codes

        # turn gradients on
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.appearance_coef.requires_grad = True

        self.appearance_optimizer = torch.optim.Adam(params=[self.model.appearance_coef], lr=1e-3)

        for dset_id, dataset in enumerate(self.test_datasets):
            pb = tqdm(total=len(dataset), desc=f"Test scene {dset_id} ({dataset.name})")

            # reset the appearance codes for
            test_frames = dataset.__len__()
            mask = torch.ones_like(self.model.appearance_coef)
            mask[: , -test_frames:] = 0
            self.model.appearance_coef.data = self.model.appearance_coef.data * mask + abs(1 - mask)

            batch_size = 4096
            for img_idx, data in enumerate(dataset):
                self.optimize_appearance_step(data, batch_size, img_idx)
                pb.update(1)
            pb.close()

        # turn gradients on
        self.model.grids.requires_grad = True
        for field in self.model.density_fields:
            field.requires_grad = True

    def validate(self):
        if hasattr(self.model, "appearance_coef"):
            self.optimize_appearance_codes()

        val_metrics = []
        with torch.no_grad():
            if self.save_outputs:  # visualize planes
                if self.model.use_F:
                    pass
                    #visualize_planes_withF(self.model, self.log_dir, f"step{self.global_step}")
                else:
                    visualize_planes(self.model, self.log_dir, f"step{self.global_step}")

            for dset_id, dataset in enumerate(self.test_datasets):
                per_scene_metrics = {
                    "psnr": 0, "ssim": 0, "dset_id": dset_id,
                }

                pred_frames, out_depths = [], []
                pb = tqdm(total=len(dataset), desc=f"Test scene {dset_id} ({dataset.name})")
                for img_idx, data in enumerate(dataset):
                    preds = self.eval_step(data, dset_id=dset_id)
                    out_metrics, out_img, out_depth = self.evaluate_metrics(
                        data["imgs"], preds, dset_id=dset_id, dset=dataset, img_idx=img_idx,
                        name=None, save_outputs=False)
                    pred_frames.append(out_img)
                    if out_depth is not None:
                        out_depths.append(out_depth)
                    per_scene_metrics["psnr"] += out_metrics["psnr"]
                    per_scene_metrics["ssim"] += out_metrics["ssim"]
                    pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                    pb.update(1)
                pb.close()
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                    pred_frames
                )
                if len(out_depths) > 0:
                    write_video_to_file(
                        os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                        out_depths
                    )
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"step {self.global_step}/{self.num_steps}"
                log_text += f" | PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)
                self.writer.add_scalar(f"test/loss/psnr", per_scene_metrics['psnr'], self.global_step)
                self.writer.add_scalar(f"test/loss/ssim", per_scene_metrics['ssim'], self.global_step)

        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):
        """
        Compute error metrics. This function gets called by `evaluate_metrics` in the base
        trainer class.
        :param preds:
        :param gt:
        :return:
        """
        # if phototourism then only compute metrics on the right side of the image
        if hasattr(self.model, "appearance_coef"):
            mid = gt.shape[1] // 2
            gt_right = gt[:, mid:]
            preds_rgb_right = preds[:, mid:]

            err = (gt_right - preds_rgb_right) ** 2
            return {
                "mse": torch.mean(err),
                "psnr": metrics.psnr(preds_rgb_right, gt_right),
                "ssim": metrics.ssim(preds_rgb_right, gt_right),
            }
        else:
            err = (gt - preds) ** 2
            return {
                "mse": torch.mean(err),
                "psnr": metrics.psnr(preds, gt),
                "ssim": metrics.ssim(preds, gt),
            }

    def load_model(self, checkpoint_data):
        for k, v in checkpoint_data['model'].items():
            if 'time_resolution' in k:
                self.model.upsample_time(v.cpu())
        self.model.load_state_dict(checkpoint_data["model"])
        logging.info("=> Loaded model state from checkpoint")
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        logging.info("=> Loaded optimizer state from checkpoint")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
            logging.info("=> Loaded scheduler state from checkpoint")
        self.global_step = checkpoint_data["global_step"]
        logging.info(f"=> Loaded step {self.global_step} from checkpoints")


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
    elif "sacre" in data_dir or "trevi" in data_dir:
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
    elif "sacre" in data_dir or "trevi" in data_dir:
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
        model.load_model(state)
    return model, config
