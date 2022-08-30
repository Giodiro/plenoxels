import abc
import logging as log
import os
from datetime import datetime
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..render import OfflineRenderer
from ..datasets import ray_default_collate
from ..tracers import PackedRFTracer


class BaseTrainer():
    """
    init()
    train():
        for every epoch:
            pre_epoch()
            train_one_epoch()
                step()
                post_step()
            post_epoch()
            |- save_model()
            validate()
    Each of these submodules can be overriden, or extended with super().
    """
    def __init__(self, tracer: PackedRFTracer, datasets: List[torch.utils.data.Dataset],
                 num_epochs: int, batch_size: int, optim_cls, lr_scheduler_type: str, lr: float,
                 weight_decay: float, grid_lr_weight: float, optim_params, log_dir: str, device,
                 num_batches_per_scene=10, exp_name=None, save_every=-1, **kwargs):
        self.extra_args = kwargs
        self.device = device
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.tracer = tracer
        self.num_scenes = len(datasets)
        if self.num_scenes == 1:
            # Reset batches-per-scene to 1 when a single scene is used, in order to get correct
            # lr-scheduling behavior.
            self.num_batches_per_scene = 1
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.tracer.parameters())))

        self.optim_cls = optim_cls
        self.lr = lr
        self.weight_decay = weight_decay
        self.grid_lr_weight = grid_lr_weight
        self.optim_params = optim_params
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_batches_per_scene = num_batches_per_scene
        self.lr_scheduler_type = lr_scheduler_type

        # Training params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.exp_name = exp_name if exp_name else "unnamed_experiment"

        self.datasets = datasets
        self.train_data_loader_iters = None
        self.epoch = None

        self.log_dir = os.path.join(
            log_dir,
            self.exp_name,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.save_every = save_every

        self.optimizer: torch.optim.Optimizer = self.init_optimizer()
        self.renderer: OfflineRenderer = self.init_renderer()
        self.scheduler = self.init_lr_scheduler()
        self.train_data_loaders = self.init_dataloaders()

    def total_batches_per_epoch(self):
        # noinspection PyTypeChecker
        return sum(len(dset) // self.batch_size for dset in self.datasets)

    def init_dataloaders(self):
        return [
            dataset.dataloader() for dataset in self.datasets
        ]

    def reset_data_iterators(self, dataset_idx=None):
        """Rewind the iterator for the new epoch.
        """
        if dataset_idx is None:
            self.train_data_loader_iters = [
                iter(dloader) for dloader in self.train_data_loaders
            ]
        else:
            self.train_data_loader_iters[dataset_idx] = iter(self.train_data_loaders[dataset_idx])

    def init_optimizer(self):
        """Default initialization for the optimizer.
        """
        decoder_params, grid_params, rest_params = [], [], []
        for name, param in self.tracer.named_parameters():
            if 'decoder' in name:
                # If "decoder" is in the name, there's a good chance it is in fact a decoder,
                # so use weight_decay
                decoder_params.append(param)
            elif 'grid' in name:
                # If "grid" is in the name, there's a good chance it is in fact a grid,
                # so use grid_lr_weight
                grid_params.append(param)
            else:
                rest_params.append(param)
        params = [
            {"params": decoder_params, "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": grid_params, "lr": self.lr * self.grid_lr_weight},
            {"params": rest_params, "lr": self.lr},
        ]
        return self.optim_cls(params, **self.optim_params)

    def init_lr_scheduler(self):
        lr_sched = None
        if self.lr_scheduler_type == "cosine":
            eta_min = 1e-4
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_batches_per_epoch() // self.num_batches_per_scene,
                eta_min=eta_min)
        return lr_sched

    def init_renderer(self):
        return OfflineRenderer(**self.extra_args)

    def train_one_epoch(self):
        self.pre_epoch()
        active_scenes = list(range(self.num_scenes))
        ascene_idx = 0
        iteration = 1
        pb = tqdm(total=self.total_batches_per_epoch(), desc=f"E{self.epoch}")
        step_successful = False
        while len(active_scenes) > 0:
            try:
                for j in range(self.num_batches_per_scene):
                    data = next(self.train_data_loader_iters[active_scenes[ascene_idx]])
                    step_successful |= self.step(iteration, data, active_scenes[ascene_idx])
                    self.post_step(n_iter=iteration, pb=pb)
            except StopIteration:
                active_scenes.pop(ascene_idx)
            else:
                # go to next scene
                ascene_idx = (ascene_idx + 1) % len(active_scenes)
                iteration += 1
            if ascene_idx == 0 and step_successful and self.scheduler is not None:  # we went through all scenes
                self.scheduler.step()
        self.post_epoch()

    def pre_epoch(self):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        self.reset_data_iterators()
        self.tracer.train()

    def post_epoch(self):
        """End epoch.
        """
        self.tracer.eval()
        # Save model
        if self.save_every > -1 and self.epoch % self.save_every == 0:
            self.save_model()
        if self.extra_args["valid_every"] > -1 and \
                self.epoch % self.extra_args["valid_every"] == 0 and \
                self.epoch != 0:
            self.validate()
        if self.epoch >= self.num_epochs:
            raise StopIteration(f"Finished after {self.num_epochs} epochs.")

    @abc.abstractmethod
    def step(self, n_iter, data, scene_idx) -> bool:
        pass

    @abc.abstractmethod
    def post_step(self, n_iter, pb):
        pass

    def save_model(self):
        """Override this function to change model saving."""
        model_fname = os.path.join(self.log_dir, f'model.pth')
        log.info(f'Saving model checkpoint to: {model_fname}')

        torch.save({
            "tracer": self.tracer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": self.epoch,
        }, model_fname)

    def load_model(self, checkpoint_data):
        self.tracer.load_state_dict(checkpoint_data["tracer"])
        log.info("=> Loaded tracer state from checkpoint")
        self.optimizer.load_state_dict(checkpoint_data["optimizer"])
        log.info("=> Loaded optimizer state from checkpoint")
        self.scheduler.load_state_dict(checkpoint_data['scheduler'])
        log.info("=> Loaded scheduler state from checkpoint")
        self.epoch = checkpoint_data["epoch"]
        log.info(f"=> Loaded epoch-state {self.epoch} from checkpoints")

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.epoch is None:
            self.epoch = 0
        log.info(f"Starting training from epoch {self.epoch + 1}")
        try:
            while True:
                self.epoch += 1
                self.train_one_epoch()
        except StopIteration as e:
            log.info(str(e))
        finally:
            self.writer.close()

    @abc.abstractmethod
    def validate(self):
        pass
