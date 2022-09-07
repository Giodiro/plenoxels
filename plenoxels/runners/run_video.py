import argparse
import importlib.util
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional
import pprint

import numpy as np
np.random.seed(0)
import pandas as pd
import torch
torch.manual_seed(0)
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from plenoxels.ema import EMA
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_exr, write_png
from plenoxels.runners.utils import *
from plenoxels.video_dataset import VideoDataset



class Trainer():
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.Dataset,
                 num_batches_per_dset: int,
                 num_epochs: int,
                 scheduler_type: Optional[str],
                 model_type: str,
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 transfer_learning: bool = False,
                 **kwargs
                 ):
        self.train_data_loader = tr_loader
        self.test_dataset = ts_dset
        self.extra_args = kwargs

        self.num_batches_per_dset = num_batches_per_dset

        self.batch_size = tr_loader.batch_size
        self.num_epochs = num_epochs

        self.scheduler_type = scheduler_type
        self.model_type = model_type
        self.optim_type = optim_type
        self.transfer_learning = transfer_learning
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.epoch = None
        self.global_step = None
        self.loss_info = None
        self.train_iterator = None

        self.model = self.init_model(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

    def eval_step(self, data) -> torch.Tensor:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data[0]
            rays_d = data[1]
            timestamp = data[2]
            preds = []
            for b in range(math.ceil(rays_o.shape[0] / self.batch_size)):
                rays_o_b = rays_o[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rays_d_b = rays_d[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                timestamps_d_b = torch.ones(len(rays_o_b)).cuda() * timestamp
                preds.append(self.model(rays_o_b, rays_d_b, timestamps_d_b, bg_color=1))
            preds = torch.cat(preds, 0)
        return preds

    def step(self, data):
        rays_o = data[0].cuda()
        rays_d = data[1].cuda()
        timestamps = data[2].cuda()
        imgs = data[3].cuda()
        self.optimizer.zero_grad(set_to_none=True)

        C = imgs.shape[-1]
        # Random bg-color
        if C == 3:
            bg_color = 1
        else:
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rgb_preds = self.model(rays_o, rays_d, timestamps, bg_color=bg_color)
            loss = self.criterion(rgb_preds, imgs)

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        loss_val = loss.item()
        self.loss_info["mse"].update(loss_val)
        self.loss_info["psnr"].update(-10 * math.log10(loss_val))
        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        self.writer.add_scalar(f"mse: ", self.loss_info["mse"].value, self.global_step)
        progress_bar.set_postfix_str(losses_to_postfix(self.loss_info), refresh=False)
        progress_bar.update(1)

    def pre_epoch(self):
        self.reset_data_iterators()
        self.init_epoch_info()
        self.model.train()

    def post_epoch(self):
        self.model.eval()
        # Save model
        if self.save_every > -1 and self.epoch % self.save_every == 0:
            self.save_model()
        if self.valid_every > -1 and \
                self.epoch % self.valid_every == 0 and \
                self.epoch != 0:
            self.validate()
        if self.epoch >= self.num_epochs:
            raise StopIteration(f"Finished after {self.num_epochs} epochs.")

    def train_epoch(self):
        self.pre_epoch()
        pb = tqdm(total=self.total_batches_per_epoch(), desc=f"E{self.epoch}")
        try:
            step_successful = False
            while True:
                data = next(self.train_iterator)
                step_successful |= self.step(data)
                self.post_step(progress_bar=pb)
            self.scheduler.step()
        except StopIteration:
            pb.close()
        self.post_epoch()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.epoch is None:
            self.epoch = 0
        if self.global_step is None:
            self.global_step = 0
        logging.info(f"Starting training from epoch {self.epoch + 1}")
        try:
            while True:
                self.epoch += 1
                self.train_epoch()
        except StopIteration as e:
            logging.info(str(e))
        finally:
            self.writer.close()

    def validate(self):
        logging.info("Beginning validation...")
        val_metrics = []
        with torch.no_grad():
            dataset = self.test_dataset
            per_scene_metrics = {
                "psnr": 0,
                "ssim": 0,
            }
            for img_idx, data in enumerate(tqdm(dataset, desc=f"Test")):
                preds = self.eval_step(data)
                gt = data[3]
                out_metrics = self.evaluate_metrics(
                    gt, preds, dset=dataset, img_idx=img_idx, name=None)
                per_scene_metrics["psnr"] += out_metrics["psnr"]
                per_scene_metrics["ssim"] += out_metrics["ssim"]
            per_scene_metrics["psnr"] /= len(dataset)  # noqa
            per_scene_metrics["ssim"] /= len(dataset)  # noqa
            log_text = f"EPOCH {self.epoch}/{self.num_epochs}"
            log_text += f" | PSNR: {per_scene_metrics['psnr']:.2f}"
            log_text += f" | SSIM: {per_scene_metrics['ssim']:.6f}"
            logging.info(log_text)
            val_metrics.append(per_scene_metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_epoch{self.epoch}.csv"))

    def evaluate_metrics(self, gt, preds: torch.Tensor, dset, img_idx, name=None):
        # gt = gt.reshape(dset.img_h, dset.img_w, -1).cpu()
        gt = gt.reshape(dset.img_w, dset.img_h, -1).cpu()
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
        preds = torch.permute(preds.reshape(dset.img_h, dset.img_w, 3), (1, 0, 2)).cpu()
        # preds = preds.reshape(dset.img_w, dset.img_h, 3).cpu()
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

        out_name = f"epoch{self.epoch}-{img_idx}"
        if name is not None and name != "":
            out_name += "-" + name

        write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
        write_png(os.path.join(self.log_dir, out_name + ".png"), (torch.cat((preds, gt), dim=1) * 255.0).byte().numpy())

        return summary

    def save_model(self):
        """Override this function to change model saving."""
        model_fname = os.path.join(self.log_dir, f'model.pth')
        logging.info(f'Saving model checkpoint to: {model_fname}')

        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": self.epoch,
            "global_step": self.global_step
        }, model_fname)

    def load_model(self, checkpoint_data):
        if self.transfer_learning:
            # Only reload model components that are not scene-specific, and don't reload the optimizer or scheduler
            for key in checkpoint_data["model"].keys():
                if 'scene' in key:
                    continue
                self.model.load_state_dict({key: checkpoint_data["model"][key]}, strict=False)
                # Don't optimize parameters that are reloaded (this should be fine, but it doesn't train)
                # for param in self.model.parameters():
                #     if param.shape == checkpoint_data["model"][key].shape:
                #         print(f'setting requires_grad false for param shape {param.shape}')
                #         param.requires_grad = False
                logging.info(f"=> Loaded model state key with shape {checkpoint_data['model'][key].shape} from checkpoint")
        else:
            self.model.load_state_dict(checkpoint_data["model"])
            logging.info("=> Loaded model state from checkpoint")
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            logging.info("=> Loaded optimizer state from checkpoint")
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint_data['scheduler'])
                logging.info("=> Loaded scheduler state from checkpoint")
            self.epoch = checkpoint_data["epoch"]
            self.global_step = checkpoint_data["global_step"]
            logging.info(f"=> Loaded epoch-state {self.epoch}, step {self.global_step} from checkpoints")

    def total_batches_per_epoch(self):
        # noinspection PyTypeChecker
        return len(self.train_data_loader.dataset) // self.batch_size

    def reset_data_iterators(self):
        """Rewind the iterator for the new epoch.
        """
        self.train_iterator = iter(self.train_data_loader)

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = defaultdict(lambda: EMA(ema_weight))

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 1e-4
        lr_sched = None
        if self.scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs * self.total_batches_per_epoch() // self.num_batches_per_dset,
                eta_min=eta_min)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        if self.optim_type == 'adam':
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=kwargs.get("lr"))
        else:
            raise NotImplementedError()
        return optim

    def init_model(self, **kwargs) -> torch.nn.Module:
        radi = self.train_data_loader.dataset.radius
        len_time = self.train_data_loader.dataset.len_time
        if self.model_type == "learnable_hash":
            model = LowrankVideo(
                radi=radi,
                len_time=len_time,
                **kwargs)
        else:
            raise ValueError(f"Model type {self.model_type} invalid")
        logging.info(f"Initialized model of type {self.model_type} with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model


def eval_render_fn(renderer):
    def render_fn(ro, rd):
        return renderer(ro, rd, bg_color=1.0)
    return render_fn


def losses_to_postfix(loss_dict: Dict[str, EMA]) -> str:
    return ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)


def load_data(data_downsample, data_dir, subsample_time_train, batch_size, **kwargs):
    if data_downsample is None:
        data_downsample = 1.0
    # Training datasets are lists of lists, where each inner list is different resolutions for the same scene
    # Test datasets are a single list over the different scenes, all at full resolution
    logging.info(f"About to load data with downsample={data_downsample} and using {subsample_time_train * 100}% of the video frames")
    tr_dset = VideoDataset(
        data_dir, split='train', downsample=data_downsample, 
        subsample_time=subsample_time_train)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=batch_size, shuffle=True, num_workers=3,
        prefetch_factor=4, pin_memory=True)
    ts_dset = VideoDataset(
        data_dir, split='test', downsample=1,
        subsample_time=1)
    return tr_loader, ts_dset


def main():
    setup_logging()
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logging.info(f"Selected GPU {gpu}")

    p = argparse.ArgumentParser(description="")

    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)

    args = p.parse_args()
    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    pprint.pprint(cfg.config)
    tr_loader, ts_dset = load_data(**cfg.config)
    trainer = Trainer(tr_loader=tr_loader, ts_dset=ts_dset, **cfg.config)
    if trainer.transfer_learning:
        # We have reloaded the model learned from args.log_dir
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    if args.log_dir is not None:
        trainer.log_dir = args.log_dir
        checkpoint_path = os.path.join(trainer.log_dir, "model.pth")
        trainer.load_model(torch.load(checkpoint_path))

    if args.validate_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
