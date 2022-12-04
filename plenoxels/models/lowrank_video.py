import logging as log
import math
from typing import Dict, List, Union, Sequence, Tuple

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
                 concat_features: bool = False,
                 **kwargs):
        self.len_time = len_time  # maximum timestep - used for normalization
        super().__init__(
            grid_config=grid_config,
            sh=sh,
            use_F=use_F,
            density_activation=density_activation,
            aabb=aabb,
            concat_features=concat_features,
        )
        self.multiscale_res = multiscale_res
        self.render_n_samples = render_n_samples
        self.extra_args = kwargs

        if self.use_F:
            raise NotImplementedError()

        self.feature_dim = 0
        self.grids = nn.ModuleList()
        for res_idx, res in enumerate(self.multiscale_res):
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config:
                    continue
                config = grid_config.copy()
                config['resolution'] = [r * res for r in config['resolution'][:3]]
                if len(grid_config['resolution']) == 4:
                    config['resolution'].append(grid_config['resolution'][3])
                gpdesc = self.init_grid_param(
                    grid_nd=config['grid_dimensions'],
                    resolution=config['resolution'],
                    out_features=config['output_coordinate_dim'],
                    input_features=4,
                    is_video=True,
                    use_F=False,
                    is_density=False,
                )
                self.set_resolution(gpdesc.reso, 0)
                if self.concat_features:
                    self.feature_dim += gpdesc.grid_coefs[-1].shape[1]
                else:
                    self.feature_dim = gpdesc.grid_coefs[-1].shape[1]
                self.grids.append(gpdesc.grid_coefs)

        self.decoder = self.init_decoder()

        log.info(f"Initialized LowrankVideo. decoder={self.decoder}, use-F: {self.use_F}, "
                 f"concat-features: {self.concat_features}")
        log.info(f"Model grids: {self.grids}")

    def compute_features(self, xyz: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        grids = self.grids
        level_info = self.config[0]
        xyzt = torch.cat([xyz, t[:, None]], dim=-1)  # [batch, 4] for xyzt
        return self.interpolate_ms_features(
            xyzt, grids, level_info, concat_features=self.concat_features, num_levels=None)

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

    def query_density(self,
                      pts: torch.Tensor,
                      timestamps: torch.Tensor,
                      return_feat: bool = False
                      ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pts_norm = self.normalize_coords(pts, 0)
        timestamps_norm = (timestamps * 2 / self.len_time) - 1
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, timestamps_norm)
        density = (
            self.density_act(self.decoder.compute_density(
                features, rays_d=None)
            ).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        if return_feat:
            return density, features
        return density

    def forward(
        self,
        pts: torch.Tensor,
        rays_d: torch.Tensor,
        timestamps: torch.Tensor,
    ):
        density, embedding = self.query_density(
                pts, timestamps, return_feat=True)
        rgb = self.decoder.compute_color(embedding, rays_d=rays_d)
        return rgb, density

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
