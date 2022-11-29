import logging as log
import math
from typing import Dict, List, Union, Sequence

import torch
import torch.nn as nn

from .decoders.mlp_decoder import RgbRenderDecoder
from .lowrank_model import LowrankModel


class LowrankLearnableHash(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],
                 sh: bool,
                 use_F: bool,
                 density_activation: str,
                 render_n_samples: int,
                 multiscale_res: Sequence[int] = (1, ),
                 num_scenes: int = 1,
                 concat_features: bool = False,
                 **kwargs):
        """
        :param grid_config:
        :param aabb:
        :param sh:
        :param use_F:
        :param density_activation:
        :param render_n_samples:
            number of intersections in each ray. Used for computing step-size.
        :param multiscale_res:
        :param num_scenes:
        :param concat_features:
            Whether to concatenate features from each resolution. The alternative is to sum them.
        :param kwargs:
        """
        super().__init__(
            grid_config=grid_config,
            sh=sh,
            use_F=use_F,
            density_activation=density_activation,
            aabb=aabb,
            concat_features=concat_features,
        )
        self.multiscale_res = multiscale_res
        self.cone_angle = kwargs.get('cone_angle', 0.0)
        self.render_n_samples = render_n_samples
        self.extra_args = kwargs

        if self.use_F:
            raise NotImplementedError()

        rgb_feature_dim = 0
        self.rgb_grids = nn.ModuleList()
        self.density_grids = nn.ModuleList()
        grid_config = self.config[0]
        for si in range(num_scenes):
            _rgb_grids = nn.ModuleList()
            _density_grids = nn.ModuleList()
            for res_idx, res in enumerate(self.multiscale_res):
                if "feature_dim" in grid_config:
                    raise ValueError(f"use_F is False but found 'feature_dim' key in grid-config.")
                config = grid_config.copy()
                config['resolution'] = [int(r * res) for r in config['resolution'][:3]]

                rgb_grid_data = self.init_grid_param(
                    grid_nd=config['grid_dimensions'],
                    resolution=config['resolution'],
                    out_features=config['rgb_features'][res_idx],
                    input_features=3,
                    is_video=False,
                    use_F=self.use_F,
                )
                density_grid_data = self.init_grid_param(
                    grid_nd=config['grid_dimensions'],
                    resolution=config['resolution'],
                    out_features=config['density_features'][res_idx],
                    input_features=3,
                    is_video=False,
                    use_F=self.use_F,
                )
                self.set_resolution(rgb_grid_data.reso, grid_id=si)
                if self.concat_features:
                    rgb_feature_dim += rgb_grid_data.grid_coefs[-1].shape[1]
                else:
                    rgb_feature_dim = rgb_grid_data.grid_coefs[-1].shape[1]
                _rgb_grids.append(rgb_grid_data.grid_coefs)
                _density_grids.append(density_grid_data.grid_coefs)
            self.rgb_grids.append(_rgb_grids)
            self.density_grids.append(_density_grids)

        # self.decoder = self.init_decoder()
        self.decoder = RgbRenderDecoder(feature_dim=rgb_feature_dim)

        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, "
                 f"decoder: {self.decoder}, use-F: {self.use_F}, "
                 f"concat-features: {self.concat_features}"
                 f"rgb-feature-dim: {rgb_feature_dim}")
        log.info(f"RGB grids: {self.rgb_grids}")
        log.info(f"Density grids: {self.density_grids}")

    def compute_density_features(self, xyz: torch.Tensor, grid_id: int) -> torch.Tensor:
        grids: nn.ModuleList = self.density_grids[grid_id]  # noqa
        level_info = self.config[0]
        density_interp = self.interpolate_ms_features(
            xyz, grids, level_info, concat_features=self.concat_features)
        return density_interp  # [N, D]

    def compute_rgb_features(self, xyz: torch.Tensor, grid_id: int) -> torch.Tensor:
        grids: nn.ModuleList = self.rgb_grids[grid_id]  # noqa
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid
        rgb_interp = self.interpolate_ms_features(
            xyz, grids, level_info, concat_features=self.concat_features)
        return rgb_interp  # [N, D]

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         ) -> torch.Tensor:
        grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grid_info = self.config[0]
        multiscale_interp = self.interpolate_ms_features(
            pts, grids, grid_info, concat_features=self.concat_features)
        return multiscale_interp

    def step_size(self, n_samples: int, grid_id: int):
        aabb = self.aabb(grid_id)
        return (
            (aabb[1] - aabb[0]).max()
            * (math.sqrt(3) / n_samples)
        )

    def query_opacity(self, pts: torch.Tensor, grid_id: int, dset):
        density = self.query_density(pts, grid_id)
        if self.cone_angle > 0.0:
            render_step_size = self.step_size(self.render_n_samples, grid_id)
            # randomly sample a camera for computing step size.
            camera_ids = torch.randint(
                0, len(dset), (pts.shape[0],), device=pts.device
            )
            origins = dset.camtoworlds[camera_ids, :3, -1]
            t: torch.Tensor = (origins - pts).norm(dim=-1, keepdim=True)
            # compute actual step size used in marching, based on the distance to the camera.
            step_size = torch.clamp(
                t * self.cone_angle, min=render_step_size
            )
            # filter out the points that are not in the near far plane.
            if (dset.near is not None) and (dset.far is not None):
                step_size = torch.where(
                    (t > dset.near) & (t < dset.far),
                    step_size,
                    torch.zeros_like(step_size),
                )
        else:
            step_size = self.step_size(self.render_n_samples, grid_id)

        opacity = density * step_size
        return opacity

    def query_density(self, pts: torch.Tensor, grid_id: int):
        pts_norm = self.normalize_coords(pts, grid_id)
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)
        density = (
            self.density_act(self.decoder.compute_density(
                self.compute_density_features(pts_norm, grid_id), rays_d=None)
            ).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        return density

    def forward(
        self,
        pts: torch.Tensor,
        rays_d: torch.Tensor,
        grid_id: int,
    ):
        pts_norm = self.normalize_coords(pts, grid_id)
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)
        density = (
            self.density_act(self.decoder.compute_density(
                self.compute_density_features(pts_norm, grid_id), rays_d=None)
            ).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        rgb = (
            self.decoder.compute_color(
                self.compute_rgb_features(pts_norm, grid_id), rays_d=rays_d
            )
            * selector[..., None]
        )
        return rgb, density

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
