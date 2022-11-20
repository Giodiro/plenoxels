import logging as log
import math
from typing import Dict, List, Union, Sequence

import torch
import torch.nn as nn

from .lowrank_model import LowrankModel


class LowrankVideo(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: torch.Tensor,  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 sh: bool,
                 use_F: bool,
                 density_activation: str,
                 render_n_samples: int,
                 multiscale_res: Sequence[int] = (1, ),
                 global_translation=None,
                 global_scale=None,
                 **kwargs):
        self.len_time = len_time  # maximum timestep - used for normalization
        super().__init__(
            grid_config=grid_config,
            sh=sh,
            use_F=use_F,
            density_activation=density_activation,
            aabb=aabb,
        )
        self.multiscale_res = multiscale_res
        self.render_n_samples = render_n_samples
        self.extra_args = kwargs

        if self.use_F:
            raise NotImplementedError()

        self.grids = nn.ModuleList()
        for res in self.multiscale_res:
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config:
                    continue
                config = grid_config.copy()
                config['resolution'] = [r * res for r in config['resolution'][:3]]
                if len(grid_config['resolution']) == 4:
                    config['resolution'].append(grid_config['resolution'][3])
                gpdesc = self.init_grid_param(config, grid_level=li, is_video=True, use_F=False)
                self.set_resolution(gpdesc.reso, 0)
                self.grids.append(gpdesc.grid_coefs)
                self.feature_dim = gpdesc.grid_coefs[-1].shape[1]

        log.info(f"Initialized LowrankVideo. decoder={self.decoder}")
        log.info(f"Model grids: {self.grids}")

    def compute_features(self,
                         pts,
                         timestamps,
                         ) -> torch.Tensor:
        # space: 6 x [1, rank * F_dim, reso, reso] where the reso can be different in different grids and dimensions
        multiscale_space: torch.nn.ModuleList = self.grids
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid
        # Interpolate in space and time
        pts = torch.cat([pts, timestamps[:, None]], dim=-1)  # [batch, 4] for xyzt

        multiscale_interp = self.interpolate_ms_features(
            pts, multiscale_space, level_info, self.feature_dim)
        return multiscale_interp

    def step_size(self, n_samples: int):
        aabb = self.aabb(0)
        return (
            (aabb[1] - aabb[0]).max()
            * (math.sqrt(3) / n_samples)
        )

    def query_opacity(self, x, timestamps, dset):
        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        # TODO: This is problematic since it assumes a constant step size, which is weird when using contraction.
        step_size = self.step_size(self.render_n_samples)
        opacity = density * step_size
        return opacity

    def query_density(self, pts: torch.Tensor, timestamps: torch.Tensor, return_feat: bool = False):
        pts_norm = self.normalize_coords(pts, 0)
        timestamps_norm = (timestamps * 2 / self.len_time) - 1
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, timestamps_norm)
        density = (
            self.density_act(self.decoder.compute_density(
                features, rays_d=None, precompute_color=False)).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        if return_feat:
            return density, features
        return density

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        timestamps: torch.Tensor,
    ):
        density, embedding = self.query_density(rays_o, timestamps, return_feat=True)
        rgb = self.decoder.compute_color(embedding, rays_d=rays_d)
        return rgb, density

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
