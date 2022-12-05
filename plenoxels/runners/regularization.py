import abc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler
from torch import nn

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.models.utils import compute_plane_tv, compute_plane_smoothness
from plenoxels.ops.losses.histogram_loss import interlevel_loss


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


class HistogramLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('histogram-loss', initial_value)

        self.visualize = False
        self.count = 0

    def _regularize(self, model: LowrankModel, model_out, grid_id: int, **kwargs) -> torch.Tensor:

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
