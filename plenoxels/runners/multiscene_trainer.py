import logging
from contextlib import ExitStack
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from plenoxels.ema import EMA
from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from plenoxels.models.utils import compute_tv_norm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_exr, write_png
from ..datasets import SyntheticNerfDataset, LLFFDataset
from ..datasets.patchloader import PatchLoader
from ..my_tqdm import tqdm
from ..utils import parse_optint


class Trainer():
    def __init__(self,
                 tr_loaders: List[torch.utils.data.DataLoader],
                 ts_dsets: List[torch.utils.data.TensorDataset],
                 patch_loaders: List[Optional[PatchLoader]],
                 regnerf_weight_start: float,
                 regnerf_weight_end: float,
                 regnerf_weight_max_step: int,
                 plane_tv_weight: float,
                 l1density_weight: float,
                 num_batches_per_dset: int,
                 num_epochs: int,
                 scheduler_type: Optional[str],
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
        self.patch_loaders = patch_loaders
        self.extra_args = kwargs
        self.num_dsets = len(self.train_data_loaders)
        assert len(self.test_datasets) == self.num_dsets

        self.regnerf_weight_start = regnerf_weight_start
        self.regnerf_weight_end = regnerf_weight_end
        self.regnerf_weight_max_step = regnerf_weight_max_step
        self.cur_regnerf_weight = self.regnerf_weight_start
        if self.cur_regnerf_weight > 0:
            assert all(pl is not None for pl in self.patch_loaders)
        self.plane_tv_weight = plane_tv_weight
        self.l1density_weight = l1density_weight

        self.num_batches_per_dset = num_batches_per_dset
        if self.num_dsets == 1 and self.num_batches_per_dset != 1:
            logging.warning("Changing 'batches_per_dset' to 1 since training with a single dataset.")
            self.num_batches_per_dset = 1

        self.batch_size = tr_loaders[0].batch_size
        self.num_epochs = num_epochs

        self.scheduler_type = scheduler_type
        self.optim_type = optim_type
        self.transfer_learning = kwargs.get('transfer_learning')
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.density_mask_update_steps = list(kwargs.get('dmask_update', []))
        self.upsample_steps = list(kwargs.get('upsample_steps', []))
        self.upsample_resolution_list = list(kwargs.get('upsample_resolution', []))
        assert len(self.upsample_resolution_list) == len(self.upsample_steps)

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

    def eval_step(self, data, dset_id) -> Dict[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            rays_o = data[0]
            rays_d = data[1]
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / self.batch_size)):
                rays_o_b = rays_o[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                rays_d_b = rays_d[b * self.batch_size: (b + 1) * self.batch_size].cuda()
                if self.train_data_loaders[0].dataset.is_ndc:
                    bg_color = None
                else:
                    bg_color = 1
                outputs = self.model(rays_o_b, rays_d_b, grid_id=dset_id, bg_color=bg_color, channels={"rgb", "depth"})
                for k, v in outputs.items():
                    preds[k].append(v)
            for k, v in preds.items():
                preds[k] = torch.cat(v, 0)
        return preds

    def step(self, data, dset_id):
        rays_o = data["train"][0].cuda()
        rays_d = data["train"][1].cuda()
        imgs = data["train"][2].cuda()
        patch_rays_o, patch_rays_d = None, None
        if "patches" in data:
            patch_rays_o = data["patches"][0].cuda()
            patch_rays_d = data["patches"][1].cuda()

        self.optimizer.zero_grad(set_to_none=True)

        C = imgs.shape[-1]
        if self.train_data_loaders[0].dataset.is_ndc:
            bg_color = None
        elif C == 3:
            bg_color = 1
        else:  # Random bg-color
            bg_color = torch.rand_like(imgs[..., :3])
            imgs = imgs[..., :3] * imgs[..., 3:] + bg_color * (1.0 - imgs[..., 3:])

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(rays_o, rays_d, grid_id=dset_id, bg_color=bg_color, channels={"rgb"})
            rgb_preds = fwd_out["rgb"]
            # Reconstruction loss
            recon_loss = self.criterion(rgb_preds, imgs)
            loss = recon_loss
            # Different regularizers
            l1density = None
            if self.l1density_weight > 0:
                l1density = self.model.compute_l1density(
                    max_voxels=100_000, grid_id=dset_id) * self.l1density_weight
                loss = loss + l1density
            depth_tv = None
            if self.cur_regnerf_weight > 0:
                ps = patch_rays_o.shape[1]  # patch-size
                out = self.model(
                    patch_rays_o.reshape(-1, 3), patch_rays_d.reshape(-1, 3), bg_color=None,
                    channels={"depth", "rgb"}, grid_id=dset_id)
                reshape_to_patch = lambda x, dim: x.reshape(-1, ps, ps, dim)
                depths = reshape_to_patch(out["depth"], 1)
                #if self.global_step > 2000:
                #    torch.save(depths.detach().cpu(), "depths8.pt")
                #    rgbs = reshape_to_patch(out["rgb"], 3)
                #    torch.save(rgbs.detach().cpu(), "rgbs8.pt")
                #    raise RuntimeError()
                # NOTE: the weighting below is never applied in RegNerf (the constant in front of it is set to 0)
                # with torch.autograd.no_grad():
                #     weighting = reshape_to_patch(out["alpha"], 1)[:, :-1, :-1]
                depth_tv = compute_tv_norm(depths, 'l2', None) * self.cur_regnerf_weight
                loss = loss + depth_tv
            plane_tv = None
            if self.plane_tv_weight > 0:
                plane_tv = self.model.compute_plane_tv(dset_id) * self.plane_tv_weight
                loss = loss + plane_tv

        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()

        recon_loss_val = recon_loss.item()
        self.loss_info[dset_id]["mse"].update(recon_loss_val)
        self.loss_info[dset_id]["psnr"].update(-10 * math.log10(recon_loss_val))
        if l1density is not None:
            self.loss_info[dset_id]["l1_density"].update(l1density.item())
        if depth_tv is not None:
            self.loss_info[dset_id]["depth_tv"].update(depth_tv.item())
        if plane_tv is not None:
            self.loss_info[dset_id]["plane_tv"].update(plane_tv.item())

        opt_reset_required = False
        if self.global_step in self.density_mask_update_steps:
            logging.info(f"Updating alpha-mask for all datasets at step {self.global_step}.")
            for u_dset_id in range(self.num_dsets):
                new_aabb = self.model.update_alpha_mask(grid_id=u_dset_id)
                if self.global_step == min(self.density_mask_update_steps):
                    self.model.shrink(new_aabb, grid_id=u_dset_id)
                    opt_reset_required = True
        try:
            upsample_step_idx = self.upsample_steps.index(self.global_step)
            new_num_voxels = self.upsample_resolution_list[upsample_step_idx]
            for u_dset_id in range(self.num_dsets):
                new_reso = N_to_reso(new_num_voxels, self.model.aabb(u_dset_id))
                self.model.upsample(new_reso, u_dset_id)
            opt_reset_required = True
        except ValueError:
            pass

        if opt_reset_required:
            # We reset the optimizer in case some of the parameters in model were changed.
            self.optimizer = self.init_optim(**self.extra_args)

        return scale <= self.gscaler.get_scale()

    def post_step(self, dset_id, progress_bar):
        # Anneal regularization weights
        if self.regnerf_weight_start > 0:
            w = np.clip(self.global_step / (1 if self.regnerf_weight_max_step < 1 else self.regnerf_weight_max_step), 0, 1)
            self.cur_regnerf_weight = self.regnerf_weight_start * (1 - w) + w * self.regnerf_weight_end

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
            with ExitStack() as stack:
                prof = None
                if False:  # TODO: Put this behind a flag
                    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                   schedule=schedule(wait=10, warmup=5, active=2),
                                   on_trace_ready=trace_handler,
                                   record_shapes=True,
                                   with_stack=True)
                    stack.enter_context(p)
                # Whether the set of batches for one loop of num_batches_per_dset
                # for every dataset, had any successful step
                step_successful = False
                while len(active_scenes) > 0:
                    try:
                        for j in range(self.num_batches_per_dset):
                            data = next(self.train_iterators[active_scenes[ascene_idx]])
                            step_successful |= self.step(data, active_scenes[ascene_idx])
                            self.post_step(dset_id=active_scenes[ascene_idx], progress_bar=pb)
                            if prof is not None:
                                prof.step()
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
                if False:  # Save a training image as well
                    dset = self.train_data_loaders[0].dataset
                    data = (dset.rays_o.view(-1, gt.shape[0], 3)[0],
                            dset.rays_d.view(-1, gt.shape[0], 3)[0],
                            dset.imgs.view(-1, gt.shape[0], gt.shape[1])[0])
                    preds = self.eval_step(data, dset_id=dset_id)
                    out_metrics = self.evaluate_metrics(
                        data[2], preds, dset_id=dset_id, dset=dset, img_idx=0, name="train",
                        save_outputs=True)
                    print(f"train img 0 PSNR={out_metrics['psnr']}")
                pb.close()
                per_scene_metrics["psnr"] /= len(dataset)  # noqa
                per_scene_metrics["ssim"] /= len(dataset)  # noqa
                log_text = f"EPOCH {self.epoch}/{self.num_epochs} | scene {dset_id}"
                log_text += f" | D{dset_id} PSNR: {per_scene_metrics['psnr']:.2f}"
                log_text += f" | D{dset_id} SSIM: {per_scene_metrics['ssim']:.6f}"
                logging.info(log_text)
                val_metrics.append(per_scene_metrics)
            if False:  # Render extra poses
                dset = self.train_data_loaders[0].dataset
                ro, rd = dset.extra_rays_o, dset.extra_rays_d
                pb = tqdm(total=ro.shape[0], desc="Rendering extra poses")
                for pose_idx in range(ro.shape[0]):
                    data = [ro[pose_idx].view(-1, 3), rd[pose_idx].view(-1, 3)]
                    preds = self.eval_step(data, dset_id=0)
                    self.evaluate_metrics(None, preds, dset_id=0, dset=dset, img_idx=pose_idx, name="extra_pose",
                                          save_outputs=True)
                    pb.update(1)

        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_epoch{self.epoch}.csv"))

    def evaluate_metrics(self, gt, preds: Dict[str, torch.Tensor], dset, dset_id: int, img_idx: int,
                         name: Optional[str] = None, save_outputs: bool = True):
        preds_rgb = preds["rgb"].reshape(dset.img_h, dset.img_w, 3).cpu()
        exrdict = {
            "preds": preds_rgb.numpy(),
        }
        if gt is not None:
            gt = gt.reshape(dset.img_h, dset.img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            exrdict["gt"] = gt.numpy()

            err = (gt - preds_rgb) ** 2
            exrdict["err"] = err.numpy()
        if "depth" in preds:
            # normalize depth and add to exrdict
            preds["depth"] = ((preds["depth"] - preds["depth"].min()) / (preds["depth"].max() - preds["depth"].min())) \
                                .cpu() \
                                .reshape(dset.img_h, dset.img_w) \
            exrdict["depth"] = preds["depth"][..., None].numpy()

        summary = dict()
        if gt is not None:
            summary["mse"] = torch.mean(err)
            summary["psnr"] = metrics.psnr(preds_rgb, gt)
            summary["ssim"] = metrics.ssim(preds_rgb, gt)

        if save_outputs:
            out_name = f"epoch{self.epoch}-D{dset_id}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_exr(os.path.join(self.log_dir, out_name + ".exr"), exrdict)
            write_png(os.path.join(self.log_dir, out_name + ".png"), (preds_rgb * 255.0).byte().numpy())
            if "depth" in preds:
                out_name = f"epoch{self.epoch}-D{dset_id}-{img_idx}-depth"
                depth = preds["depth"][..., None].repeat(1, 1, 3)
                write_png(os.path.join(self.log_dir, out_name + ".png"), (depth * 255.0).byte().numpy())

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

        Since we have an infinite iterable for patches and a finite iterable for the
        train-data-loader, we zip the two and give each a key ("train" or "patches")
        """

        def imerge_wkeys(patches, train):
            if patches is None:
                for i in train:
                    yield {"train": i}
            else:
                for i, j in zip(train, patches):
                    yield {"train": i, "patches": j}

        if dataset_idx is None:
            self.train_iterators = [
                imerge_wkeys(
                    patches=iter(self.patch_loaders[dset_idx]) if self.patch_loaders[dset_idx] is not None else None,
                    train=iter(self.train_data_loaders[dset_idx])
                ) for dset_idx in range(self.num_dsets)
            ]
        else:
            self.train_iterators[dataset_idx] = imerge_wkeys(
                patches=iter(self.patch_loaders[dataset_idx]) if self.patch_loaders[dataset_idx] is not None else None,
                train=iter(self.train_data_loaders[dataset_idx]))

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
        model = LowrankLearnableHash(
            num_scenes=self.num_dsets,
            grid_config=kwargs.pop("grid_config"),
            aabb=aabbs,
            is_ndc=self.train_data_loaders[0].dataset.is_ndc,  # TODO: This should also be per-scene
            **kwargs)
        logging.info(f"Initialized LowrankLearnableHash model with "
                     f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters.")
        model.cuda()
        return model


def losses_to_postfix(losses: List[Dict[str, EMA]]) -> str:
    pfix_list = []
    for dset_id, loss_dict in enumerate(losses):
        pfix_inner = []
        for lname, lval in loss_dict.items():
            pfix_inner.append(f"{lname}={lval}")
        pfix_list.append(f"D{dset_id}({', '.join(pfix_inner)})")
    return '  '.join(pfix_list)


def load_data(data_downsample, data_dirs, batch_size, **kwargs):
    # TODO: multiple different dataset types are currently not supported well.
    def decide_dset_type(dd) -> str:
        if ("chair" in dd or "drums" in dd or "ficus" in dd or "hotdog" in dd
                or "lego" in dd or "materials" in dd or "mic" in dd
                or "ship" in dd):
            return "synthetic"
        elif ("fern" in dd or "flower" in dd or "fortress" in dd
              or "horns" in dd or "leaves" in dd or "orchids" in dd
              or "room" in dd or "trex" in dd):
            return "llff"
        else:
            raise RuntimeError(f"data_dir {dd} not recognized as LLFF or Synthetic dataset.")

    data_resolution = parse_optint(kwargs.get('data_resolution'))
    regnerf_weight = float(kwargs.get('regnerf_weight_start'))
    extra_views = regnerf_weight > 0

    tr_dsets, tr_loaders, ts_dsets, patch_loaders = [], [], [], []
    for data_dir in data_dirs:
        dset_type = decide_dset_type(data_dir)
        if dset_type == "synthetic":
            max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
            max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                max_frames=max_tr_frames, extra_views=extra_views))
            ts_dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, resolution=800, max_frames=max_ts_frames))
        elif dset_type == "llff":
            hold_every = parse_optint(kwargs.get('hold_every'))
            logging.info(f"About to load LLFF data downsampled by {data_downsample} times.")
            tr_dsets.append(LLFFDataset(
                data_dir, split='train', downsample=data_downsample, hold_every=hold_every,
                extra_views=extra_views))
            # Note that LLFF has same downsampling applied to train and test datasets
            ts_dsets.append(LLFFDataset(
                data_dir, split='test', downsample=data_downsample, hold_every=hold_every))
        else:
            raise ValueError(dset_type)
        tr_loaders.append(torch.utils.data.DataLoader(
            tr_dsets[-1], batch_size=batch_size, shuffle=True, num_workers=3, prefetch_factor=4,
            pin_memory=True))
        patch_loader = None
        if extra_views:
            patch_loader = PatchLoader(
                rays_o=tr_dsets[-1].extra_rays_o, rays_d=tr_dsets[-1].extra_rays_d, len_time=None,
                batch_size=batch_size, patch_size=8)
        patch_loaders.append(patch_loader)

    return {"ts_dsets": ts_dsets, "tr_loaders": tr_loaders, "patch_loaders": patch_loaders}


def trace_handler(p):
    output = p.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=20)
    print(output)
    #output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=7)
    #print(output)
    output = p.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20)
    print(output)
    p.export_chrome_trace("./logs/trace_" + str(p.step_num) + ".json")


def N_to_reso(num_voxels, aabb):
    voxel_size = ((aabb[1] - aabb[0]).prod() / num_voxels).pow(1 / 3)
    return ((aabb[1] - aabb[0]) / voxel_size).long().cpu().tolist()

