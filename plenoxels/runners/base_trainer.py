import abc
import logging
import os
from typing import Dict, Optional, MutableMapping, Union, Iterable, Tuple, Sequence

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from plenoxels.ema import EMA
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_png
from .regularization import Regularizer
from .utils import get_cosine_schedule_with_warmup, get_step_schedule_with_warmup
from ..my_tqdm import tqdm


class BaseTrainer():
    def __init__(self,
                 train_data_loader: Iterable,
                 num_steps: int,
                 scheduler_type: Optional[str],
                 optim_type: str,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.eval_batch_size = kwargs.get('eval_batch_size', 8192)
        self.extra_args = kwargs

        self.num_steps = num_steps
        self.device = device

        self.scheduler_type = scheduler_type
        self.optim_type = optim_type
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step: Optional[int] = None
        self.loss_info: Optional[Dict[str, EMA]] = None
        self.train_data_loader = train_data_loader

        self.model = self.init_model(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')  # Unsure about this here but maybe ok
        self.regularizers = self.init_regularizers(**self.extra_args)
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

        self.model.to(self.device)

    @abc.abstractmethod
    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return {}

    @abc.abstractmethod
    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs) -> bool:
        self.model.train()
        return False

    def post_step(self, progress_bar):
        if self.global_step % self.calc_metrics_every == 0:
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info, lr=self.lr), refresh=False)
            for loss_name, loss_val in self.loss_info.items():
                self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)

        progress_bar.update(1)

        if self.valid_every > -1 and self.global_step > 0 and self.global_step % self.valid_every == 0:
            print()
            self.validate()
        if self.save_every > -1 and self.global_step > 0 and self.global_step % self.save_every == 0:
            print()
            self.save_model()

    def pre_epoch(self):
        self.loss_info = self.init_epoch_info()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        logging.info(f"Starting training from step {self.global_step + 1}")
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            batch_iter = iter(self.train_data_loader)
            self.pre_epoch()
            while self.global_step < self.num_steps:
                try:
                    self.model.train()
                    data = next(batch_iter)
                    self.model.train()
                    step_successful = self.train_step(data)

                    if step_successful and self.scheduler is not None:
                        self.scheduler.step()
                    for r in self.regularizers:
                        r.step(self.global_step)
                    self.post_step(progress_bar=pb)
                    #self.model.step_cb(self.global_step, self.num_steps)

                    self.global_step += 1
                except StopIteration as e:
                    # This get thrown in `next` or `step`
                    # Still need to increment the step, otherwise we get stuck in a loop
                    self.global_step += 1
                    logging.info(str(e))
                    logging.info(f'resetting after a full pass through the data, or when the dataset changed')
                    batch_iter = iter(self.train_data_loader)
                    self.pre_epoch()
        finally:
            pb.close()
            self.writer.close()

    def _normalize_err(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        err = torch.abs(preds - gt)
        err = err.mean(-1, keepdim=True)  # mean over channels
        # normalize between 0, 1 where 1 corresponds to the 90th percentile
        # err = err.clamp_max(torch.quantile(err, 0.9))
        err = self._normalize_01(err)
        return err.repeat(1, 1, 3)

    def _normalize_01(self, t: torch.Tensor) -> torch.Tensor:
        return (t - t.min()) / t.max()

    def _normalize_depth(self, depth: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        return (
            self._normalize_01(depth)
        ).cpu().reshape(img_h, img_w)[..., None]

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])

        err = (gt - preds) ** 2
        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds, gt),
            "ms-ssim": metrics.msssim(preds, gt),
        }

    def evaluate_metrics(self,
                         gt: Optional[torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            logging.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                            f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)
        out_img = preds_rgb
        summary = dict()

        out_depth = None
        if "depth" in preds:
            out_depth = preds['depth'].cpu().reshape(img_h, img_w)[..., None]

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            summary.update(self.calc_metrics(preds_rgb, gt))
            out_img = torch.cat((out_img, gt), dim=0)
            out_img = torch.cat((out_img, self._normalize_err(preds_rgb, gt)), dim=0)

        out_img = (out_img * 255.0).byte().numpy()
        if out_depth is not None:
            out_depth = self._normalize_01(out_depth)
            out_depth = (out_depth * 255.0).repeat(1, 1, 3).byte().numpy()

        if save_outputs:
            out_name = f"step{self.global_step}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_png(os.path.join(self.log_dir, out_name + ".png"), out_img)
            if out_depth is not None:
                depth_name = out_name + "-depth"
                write_png(os.path.join(self.log_dir, depth_name + ".png"), out_depth)

        return summary, out_img, out_depth

    @abc.abstractmethod
    def validate(self):
        pass

    def get_save_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step
        }

    def save_model(self):
        model_fname = os.path.join(self.log_dir, f'model.pth')
        logging.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.get_save_dict(), model_fname)

    def load_model(self, checkpoint_data):
        self.model.load_state_dict(checkpoint_data["model"])
        logging.info("=> Loaded model state from checkpoint")

        self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        logging.info("=> Loaded optimizer state from checkpoint")

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
            logging.info("=> Loaded scheduler state from checkpoint")

        self.global_step = checkpoint_data["global_step"]
        logging.info(f"=> Loaded step {self.global_step} from checkpoints")

    @abc.abstractmethod
    def init_epoch_info(self) -> Dict[str, EMA]:
        pass

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        logging.info(f"Initializing LR Scheduler of type {self.scheduler_type} with "
                     f"{max_steps} maximum steps.")
        if self.scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif self.scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
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
        elif self.scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
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

    @abc.abstractmethod
    def init_model(self, **kwargs) -> torch.nn.Module:
        pass

    def get_regularizers(self, **kwargs) -> Sequence[Regularizer]:
        return ()

    def init_regularizers(self, **kwargs):
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in self.get_regularizers(**kwargs) if r.weight > 0]
        return regularizers

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def calc_metrics_every(self):
        return 1


def losses_to_postfix(loss_dict: Dict[str, EMA], lr: Optional[float]) -> str:
    pfix = [f"{lname}={lval}" for lname, lval in loss_dict.items()]
    if lr is not None:
        pfix.append(f"lr={lr:.2e}")
    return "  ".join(pfix)
