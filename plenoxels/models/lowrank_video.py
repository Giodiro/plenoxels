import logging as log
import math
from typing import Dict, List, Union, Sequence

import torch
import torch.nn as nn

from .decoders.mlp_decoder import RgbRenderDecoder
from .lowrank_model import LowrankModel


def contract_to_unisphere(
    x: torch.Tensor,
):
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    x = x / 2 #+ 0.5  # [-inf, inf] is at [0, 1]
    return x


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

        rgb_feature_dim = 0
        self.rgb_grids = nn.ModuleList()
        self.density_grids = nn.ModuleList()
        for res_idx, res in enumerate(self.multiscale_res):
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config:
                    continue
                config = grid_config.copy()
                config['resolution'] = [r * res for r in config['resolution'][:3]]
                if len(grid_config['resolution']) == 4:
                    config['resolution'].append(grid_config['resolution'][3])
                rgb_grid_data = self.init_grid_param(
                    grid_nd=config['grid_dimensions'],
                    resolution=config['resolution'],
                    out_features=config['rgb_features'][res_idx],
                    input_features=4,
                    is_video=True,
                    use_F=False,
                )
                density_grid_data = self.init_grid_param(
                    grid_nd=config['grid_dimensions'],
                    resolution=config['resolution'],
                    out_features=config['density_features'][res_idx],
                    input_features=4,
                    is_video=True,
                    use_F=False,
                )
                self.set_resolution(rgb_grid_data.reso, 0)
                if self.concat_features:
                    rgb_feature_dim += rgb_grid_data.grid_coefs[-1].shape[1]
                else:
                    rgb_feature_dim = rgb_grid_data.grid_coefs[-1].shape[1]
                self.rgb_grids.append(rgb_grid_data.grid_coefs)
                self.density_grids.append(density_grid_data.grid_coefs)

        self.decoder = RgbRenderDecoder(feature_dim=rgb_feature_dim)

        log.info(f"Initialized LowrankVideo. decoder={self.decoder}, use-F: {self.use_F}, "
                 f"concat-features: {self.concat_features}")
        log.info(f"Model grids: {self.grids}")

    def compute_density_features(self, xyzt) -> torch.Tensor:
        grids = self.density_grids
        level_info = self.config[0]
        density_interp = self.interpolate_ms_features(
            xyzt, grids, level_info, concat_features=self.concat_features)
        return density_interp  # [N, D]

    def compute_rgb_features(self, xyzt) -> torch.Tensor:
        # space: 6 x [1, rank * F_dim, reso, reso] where the reso can be different in different grids and dimensions
        grids = self.rgb_grids
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid
        rgb_interp = self.interpolate_ms_features(
            xyzt, grids, level_info, concat_features=self.concat_features)
        return rgb_interp  # [N, D]

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

    def query_density(self, pts: torch.Tensor, timestamps: torch.Tensor):
        pts_norm = contract_to_unisphere(self.normalize_coords(pts, 0))
        timestamps_norm = (timestamps * 2 / self.len_time) - 1
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        xyzt = torch.cat([pts_norm, timestamps_norm[:, None]], dim=-1)  # [batch, 4] for xyzt
        features = self.compute_density_features(xyzt)
        density = (
            self.density_act(self.decoder.compute_density(
                features, rays_d=None)).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        return density

    def forward(
        self,
        pts: torch.Tensor,
        rays_d: torch.Tensor,
        timestamps: torch.Tensor,
    ):
        pts_norm = contract_to_unisphere(self.normalize_coords(pts, 0))
        timestamps_norm = (timestamps * 2 / self.len_time) - 1
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        xyzt = torch.cat([pts_norm, timestamps_norm[:, None]], dim=-1)  # [batch, 4] for xyzt
        density = (
            self.density_act(self.decoder.compute_density(
                self.compute_density_features(xyzt), rays_d=None)
            ).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        rgb = (
            self.decoder.compute_color(
                self.compute_rgb_features(xyzt), rays_d=rays_d
            )
            * selector[..., None]
        )
        return rgb, density

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
