import abc
from typing import Optional
import logging as log

import torch.optim.lr_scheduler
from torch import nn

from plenoxels.models.lowrank_learnable_hash import LowrankLearnableHash
from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.models.utils import compute_plane_tv, compute_plane_smoothness
from plenoxels.ops.losses.distortion_loss import distortion_loss
from plenoxels.ops.losses.histogram_loss import interlevel_loss

import matplotlib.pyplot as plt
import numpy as np
import os


class Regularizer():
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class L1Density(Regularizer):
    def __init__(self, initial_value, max_voxels: int = 100_000):
        super().__init__('l1-density', initial_value)
        self.max_voxels = max_voxels

    def get_points_on_grid(self, aabb, grid_size, max_voxels: Optional[int] = None):
        """
        Returns points from a regularly spaced grids of size grid_size.
        Coordinates normalized between [aabb0, aabb1]
        """
        dev = aabb.device
        pts = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0], device=dev),
            torch.linspace(0, 1, grid_size[1], device=dev),
            torch.linspace(0, 1, grid_size[2], device=dev), indexing='ij'
        ), dim=-1)  # [gs0, gs1, gs2, 3]
        pts = pts.view(-1, 3)  # [gs0*gs1*gs2, 3]
        if max_voxels is not None:
            # with replacement as it's faster?
            pts = pts[torch.randint(pts.shape[0], (max_voxels, )), :]
        # Normalize between [aabb0, aabb1]
        pts = aabb[0] * (1 - pts) + aabb[1] * pts
        return pts

    def _regularize(self, model: LowrankLearnableHash, grid_id: int = 0, **kwargs):
        aabb = model.aabb(grid_id)
        reso = model.resolution(grid_id)
        pts = self.get_points_on_grid(aabb, reso, self.max_voxels)
        density = model.query_density(pts, grid_id)
        return torch.abs(density).mean()


class L1PlaneDensityVideo(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-plane-density', initial_value)

    # TODO: make this also work for lowrankappearance
    def _regularize(self, model: LowrankVideo, grid_id: int = 0, **kwargs):
        grids: nn.ModuleList = model.grids[grid_id]
        total = 0
        for grid in grids:
            grid = grid.view(model.feature_dim, -1, grid.shape[-2], grid.shape[-1])
            # density is on last feature. Apply activation before computing loss.
            total += model.density_act(grid[-1, ...]).mean()
        return total


class L1PlaneDensity(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-plane-density', initial_value)

    def _regularize(self, model: LowrankLearnableHash, grid_id: int = 0, **kwargs):
        multi_res_grids: nn.ModuleList = model.scene_grids[grid_id]
        total = 0
        
        for grids in multi_res_grids:
            for grid_ls in grids:
                for grid in grid_ls:
                    grid = grid.view(model.feature_dim, -1, grid.shape[-2], grid.shape[-1])
                    # density is on last feature. Apply activation before computing loss.
                    total += torch.abs(grid[-1, ...]).mean()
        return total


class L1PlaneColor(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-plane-color', initial_value)

    def _regularize(self, model: LowrankLearnableHash, grid_id: int = 0, **kwargs):
        multi_res_grids: nn.ModuleList = model.scene_grids[grid_id]
        total = 0
        for grids in multi_res_grids:
            for grid_ls in grids:
                for grid in grid_ls:
                    grid = grid.view(model.feature_dim, -1, grid.shape[-2], grid.shape[-1])
                    # color is on all features apart the last
                    total += torch.abs(grid[:-1, ...]).mean()
        return total


class PlaneTV(Regularizer):
    def __init__(self, initial_value, features: str = 'all'):
        if features not in {'all', 'sigma', 'sh'}:
            raise ValueError(f'features must be one of "all", "sigma" or "sh" '
                             f'but {features} was passed.')
        name = 'plane-TV'
        if features != 'all':
            name += f"-{features}"
        super().__init__(name, initial_value)
        self.features = features

    def step(self, global_step):
        #if global_step == 23000:
        #    self.weight /= 2
        #    log.info(f"Setting PlaneTV weight to {self.weight}")
        pass

    def _regularize(self, model: LowrankModel, grid_id: int = 0, **kwargs):
        multi_res_grids: nn.ModuleList = model.field.grids
        total = 0
        # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
        for grids in multi_res_grids:
            for grid in grids:
                # grid: [1, c, h, w]
                if self.features == 'all':
                    total += compute_plane_tv(grid)
                else:
                    if self.features == 'sigma':
                        total += compute_plane_tv(grid[:, -1:, ...])
                    else:
                        total += compute_plane_tv(grid[:, :-1, ...])
        return total


class VideoPlaneTV(Regularizer):
    def __init__(self, initial_value):
        super().__init__('plane-TV', initial_value)

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in model.field.grids:
            if len(grids) == 3:
                spatial_grids = [0, 1, 2]
            else:
                spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal
            for grid_id in spatial_grids:
                total += compute_plane_tv(grids[grid_id])
        return total


class TimeSmoothness(Regularizer):
    def __init__(self, initial_value):
        super().__init__('time-smoothness', initial_value)

    def _regularize(self, model: LowrankVideo, **kwargs) -> torch.Tensor:
        time_grids = [2, 4, 5]  # These are the spatiotemporal grids; the others are only spatial
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in model.field.grids:
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total


class VolumeTV(Regularizer):
    def __init__(self, initial_value, what: str = 'Gcoords', patch_size: int = 3, batch_size: int = 100):
        self.what = what
        self.patch_size = patch_size
        self.batch_size = batch_size
        super().__init__('volume-TV', initial_value)

    def _regularize(self,
                    model: LowrankLearnableHash,
                    grid_id: int = 0,
                    **kwargs) -> torch.Tensor:
        aabb = model.aabb(grid_id)
        reso = model.resolution(grid_id)
        dev = aabb.device
        pts = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.patch_size, device=dev),
            torch.linspace(0, 1, self.patch_size, device=dev),
            torch.linspace(0, 1, self.patch_size, device=dev), indexing='ij'
        ), dim=-1)  # [gs0, gs1, gs2, 3]
        pts = pts.view(-1, 3)

        start = torch.rand(self.batch_size, 3, device=dev) * (1 - self.patch_size / reso[None, :])
        end = start + (self.patch_size / reso[None, :])

        # pts: [1, gs0, gs1, gs2, 3] * (bs, 1, 1, 1, 3) + (bs, 1, 1, 1, 3)
        pts = pts[None, ...] * (end - start)[:, None, None, None, :] + start[:, None, None, None, :]
        pts = pts.view(-1, 3)  # [bs*gs0*gs1*gs2, 3]

        # Normalize between [aabb0, aabb1]
        pts = aabb[0] * (1 - pts) + aabb[1] * pts

        if self.what == 'density':
            # Compute density on the grid
            density = model.query_density(pts, grid_id)
            patches = density.view(-1, self.patch_size, self.patch_size, self.patch_size)
        elif self.what == 'Gcoords':
            pts = model.normalize_coords(pts, grid_id)
            _, coords = model.compute_features(pts, grid_id, return_coords=True)
            patches = coords.view(-1, self.patch_size, self.patch_size, self.patch_size, coords.shape[-1])
        else:
            raise ValueError(self.what)

        d0 = patches[:, 1:, :, :, :] - patches[:, :-1, :, :, :]
        d1 = patches[:, :, 1:, :, :] - patches[:, :, :-1, :, :]
        d2 = patches[:, :, :, 1:, :] - patches[:, :, :, :-1, :]

        return (d0.square().mean() + d1.square().mean() + d2.square().mean())  # l2
        # return (d0.abs().mean() + d1.abs().mean() + d2.abs().mean())  # l1


class FloaterLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('floater-loss', initial_value)

    def _regularize(self, model: LowrankLearnableHash, model_out, grid_id: int, **kwargs) -> torch.Tensor:
        from plenoxels.ops.losses.distortion_loss_warp import distortion_loss
        midpoint = torch.cat(
            [model_out["midpoint"],
             (2*model_out["midpoint"][:, -1] - model_out["midpoint"][:, -2])[:, None]], dim=1)
        dt = torch.cat([model_out["deltas"], model_out["deltas"][:, -2:-1]], dim=1)
        weight = torch.cat([
            model_out["weight"],
            1 - model_out["weight"].sum(dim=1, keepdim=True)], dim=1)
        return distortion_loss(midpoint, weight, dt) * 1e-2


class HistogramLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('histogram-loss', initial_value)

        self.visualize = False
        self.count = 0

    def _regularize(self, model: LowrankLearnableHash, model_out, grid_id: int, **kwargs) -> torch.Tensor:

        if self.visualize:
            if self.count % 100 == 0:
                # proposal info
                weights_proposal = model_out["weights_list"][0].detach().cpu().numpy()
                spacing_starts_proposal = model_out["ray_samples_list"][0].spacing_starts
                spacing_ends_proposal = model_out["ray_samples_list"][0].spacing_ends
                sdist_proposal = torch.cat([spacing_starts_proposal[..., 0], spacing_ends_proposal[..., -1:, 0]], dim=-1).detach().cpu().numpy()

                # fine info
                weights_fine = model_out["weights_list"][1].detach().cpu().numpy()
                spacing_starts_fine = model_out["ray_samples_list"][1].spacing_starts
                spacing_ends_fine = model_out["ray_samples_list"][1].spacing_ends
                sdist_fine = torch.cat([spacing_starts_fine[..., 0], spacing_ends_fine[..., -1:, 0]], dim=-1).detach().cpu().numpy()

                for i in range(10):
                    fix, ax1 = plt.subplots()

                    delta = np.diff(sdist_proposal[i], axis=-1)
                    ax1.bar(sdist_proposal[i, :-1], weights_proposal[i].squeeze() / delta , width=delta, align="edge", label='proposal', alpha=0.7, color="b")
                    ax1.legend()
                    ax2 = ax1.twinx()

                    delta = np.diff(sdist_fine[i], axis=-1)
                    ax2.bar(sdist_fine[i, :-1], weights_fine[i].squeeze() / delta, width=delta, align="edge", label='fine', alpha=0.3, color='r')
                    ax2.legend()
                    os.makedirs(f'histogram_loss/{self.count}', exist_ok=True)
                    plt.savefig(f'./histogram_loss/{self.count}/batch_{i}.png')
                    plt.close()
                    plt.cla()
                    plt.clf()
            self.count += 1
        return interlevel_loss(model_out['weights_list'], model_out['ray_samples_list'])


class DistortionLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('distortion-loss', initial_value)

    def _regularize(self, model: LowrankLearnableHash, model_out, grid_id: int, **kwargs) -> torch.Tensor:
        return distortion_loss(model_out['weights_list'], model_out['ray_samples_list'])


class L1AppearancePlanes(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-appearance', initial_value)

    def _regularize(self, model: LowrankVideo, **kwargs) -> torch.Tensor:
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = model.field.grids
        for grids in multi_res_grids:
            if len(grids) == 3:
                return 0
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]  
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
