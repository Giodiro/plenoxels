import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_exr, write_png
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..my_tqdm import tqdm
from ..utils import parse_optint


class Trainer():
    def __init__(self,
                 tr_loaders: List[torch.utils.data.DataLoader],
                 ts_dsets: List[torch.utils.data.Dataset],
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
                 save_outputs: bool,
                 **kwargs
                 ):
        self.train_data_loaders = tr_loaders
        self.test_datasets = ts_dsets
        self.extra_args = kwargs
        self.num_dsets = len(self.train_data_loaders)
        assert len(self.test_datasets) == self.num_dsets

        self.num_batches_per_dset = num_batches_per_dset
        if self.num_dsets == 1 and self.num_batches_per_dset != 1:
            logging.warning("Changing 'batches_per_dset' to 1 since training with a single dataset.")
            self.num_batches_per_dset = 1

        self.batch_size = tr_loaders[0].batch_size
        self.num_epochs = num_epochs

        self.scheduler_type = scheduler_type
        self.model_type = model_type
        self.optim_type = optim_type
        self.transfer_learning = kwargs.get('transfer_learning')
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.density_mask_update_steps = set(kwargs.get('dmask_update', []))

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.epoch = None
        self.global_step = None
        self.loss_info = None
        self.train_iterators = None

        self.model = self.init_model(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

    def eval_step(self, data, dset_id) -> torch.Tensor:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data[0]
            rays_d = data[1]
            preds = []
            for b in range(math.ceil(rays_o.shape[0] / self.batch_size)):
                rays_o_b = rays_o[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rays_d_b = rays_d[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                preds.append(self.model(rays_o_b, rays_d_b, grid_id=dset_id, bg_color=1))
            preds = torch.cat(preds, 0)
        return preds

    def step(self, data, dset_id):
        rays_o = data[0].cuda()
        rays_d = data[1].cuda()
        imgs = data[2].cuda()
        self.optimizer.zero_grad(set_to_none=True)

        C = imgs.shape[-1]
        # Random bg-color
        if C == 3:
            bg_color = 1
        else:
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rgb_preds = self.model(rays_o, rays_d, grid_id=dset_id, bg_color=bg_color)
            loss = self.criterion(rgb_preds, imgs)

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        loss_val = loss.item()
        self.loss_info[dset_id]["mse"].update(loss_val)
        self.loss_info[dset_id]["psnr"].update(-10 * math.log10(loss_val))

        if self.global_step in self.density_mask_update_steps:
            logging.info(f"Updating alpha-mask for all datasets at step {self.global_step}.")
            for u_dset_id in range(self.num_dsets):
                new_aabb = self.model.update_alpha_mask(grid_id=u_dset_id)
                self.model.shrink(new_aabb, grid_id=u_dset_id)  # TODO: This doesn't actually work

        return scale <= self.gscaler.get_scale()

    def post_step(self, dset_id, progress_bar):
        self.writer.add_scalar(f"mse/D{dset_id}", self.loss_info[dset_id]["mse"].value, self.global_step)
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
            raise StopIteration(f"Finished after {self.epoch} epochs.")

    def train_epoch(self):
        self.pre_epoch()
        active_scenes = list(range(self.num_dsets))
        ascene_idx = 0
        pb = tqdm(total=self.total_batches_per_epoch(), desc=f"E{self.epoch}")
        try:
            # Whether the set of batches for one loop of num_batches_per_dset
            # for every dataset, had any successful step
            step_successful = False
            while len(active_scenes) > 0:
                try:
                    for j in range(self.num_batches_per_dset):
                        data = next(self.train_iterators[active_scenes[ascene_idx]])
                        step_successful |= self.step(data, active_scenes[ascene_idx])
                        self.post_step(dset_id=active_scenes[ascene_idx], progress_bar=pb)
                except StopIteration:
                    active_scenes.pop(ascene_idx)
                else:
                    # go to next scene
                    ascene_idx = (ascene_idx + 1) % len(active_scenes)
                    self.global_step += 1
                # If we've been through all scenes, and at least one successful step was
                # done, we can update the scheduler.
                if ascene_idx == 0 and step_successful and self.scheduler is not None:
                    self.scheduler.step()
                    step_successful = False  # reset counter
        finally:
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
                    gt = data[2]
                    out_metrics = self.evaluate_metrics(
                        gt, preds, dset_id=dset_id, dset=dataset, img_idx=img_idx, name=None,
                        save_outputs=self.save_outputs)
                    per_scene_metrics["psnr"] += out_metrics["psnr"]
                    per_scene_metrics["ssim"] += out_metrics["ssim"]
                    pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
                    pb.update(1)
                pb.close()
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"EPOCH {self.epoch}/{self.num_epochs} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_epoch{self.epoch}.csv"))

    def evaluate_metrics(self, gt, preds: torch.Tensor, dset, dset_id: int, img_idx: int,
                         name: Optional[str] = None, save_outputs: bool = True):
        gt = gt.reshape(dset.img_h, dset.img_w, -1).cpu()
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
        preds = preds.reshape(dset.img_h, dset.img_w, 3).cpu()
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

        if save_outputs:
            out_name = f"epoch{self.epoch}-D{dset_id}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
            write_png(os.path.join(self.log_dir, out_name + ".png"), (preds * 255.0).byte().numpy())

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
                logging.info(f"=> Loaded model state {key} with shape {checkpoint_data['model'][key].shape} from checkpoint")
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
        return sum(math.ceil(len(dl.dataset) / self.batch_size) for dl in self.train_data_loaders)

    def reset_data_iterators(self, dataset_idx=None):
        """Rewind the iterator for the new epoch.
        """
        if dataset_idx is None:
            self.train_iterators = [
                iter(dloader) for dloader in self.train_data_loaders
            ]
        else:
            self.train_iterators[dataset_idx] = iter(self.train_data_loaders[dataset_idx])

    def init_epoch_info(self):
        ema_weight = 0.1
        self.loss_info = [defaultdict(lambda: EMA(ema_weight)) for _ in range(self.num_dsets)]

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
            optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr']))
        else:
            raise NotImplementedError()
        return optim

    def init_model(self, **kwargs) -> torch.nn.Module:
        aabbs = [dl.dataset.scene_bbox for dl in self.train_data_loaders]
        if self.model_type == "learnable_hash":
            model = LowrankLearnableHash(
                num_scenes=self.num_dsets,
                grid_config=kwargs.pop("grid_config"),
                aabb=aabbs,
                is_ndc=self.train_data_loaders[0].dataset.is_ndc,  # TODO: This should also be per-scene
                **kwargs)
        else:
            raise ValueError(f"Model type {self.model_type} invalid")
        logging.info(f"Initialized model of type {self.model_type} with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model


def losses_to_postfix(losses: List[Dict[str, EMA]]) -> str:
    pfix_list = []
    for dset_id, loss_dict in enumerate(losses):
        for lname, lval in loss_dict.items():
            if lname != 'psnr':
                continue
            pfix_inner = f"{lname}={lval}"
        # pfix_inner = ", ".join(f"{lname}={lval}" for lname, lval in loss_dict.items())
        pfix_list.append(f"D{dset_id}({pfix_inner})")
    return '  '.join(pfix_list)


def load_data(data_downsample, data_dirs, batch_size, **kwargs):
    # TODO: multiple different dataset types are currently not supported well.
    def decide_dset_type(data_dir) -> str:
        if ("chair" in data_dir or "drums" in data_dir or "ficus" in data_dir or "hotdog" in data_dir
                or "lego" in data_dir or "materials" in data_dir or "mic" in data_dir
                or "ship" in data_dir):
            return "synthetic"
        elif ("fern" in data_dir or "flower" in data_dir or "fortress" in data_dir
              or "horns" in data_dir or "leaves" in data_dir or "orchids" in data_dir
              or "room" in data_dir or "trex" in data_dir):
            return "llff"
        else:
            raise RuntimeError(f"data_dir {data_dir} not recognized as LLFF or Synthetic dataset.")

    data_resolution = parse_optint(kwargs.get('data_resolution'))

    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for data_dir in data_dirs:
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                max_frames=max_tr_frames))
            ts_dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, resolution=800, max_frames=max_ts_frames))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(LLFFDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                hold_every=hold_every))
            ts_dsets.append(LLFFDataset(
                data_dir, split='test', downsample=1, resolution=None, hold_every=hold_every))
        else:
            raise ValueError(dset_type)
        tr_loaders.append(torch.utils.data.DataLoader(
            tr_dsets[-1], batch_size=batch_size, shuffle=True, num_workers=3, prefetch_factor=4,
            pin_memory=True))

    return {"ts_dsets": ts_dsets, "tr_loaders": tr_loaders}
