import logging as log
import math
from typing import Dict, List, Union, Sequence, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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
        self.train_every_scale = kwargs.get('train_every_scale', False)
        assert not (self.train_every_scale and self.concat_features), \
            "concat_features not compatible with train_every_scale"
        self.extra_args = kwargs

        if self.use_F:
            raise NotImplementedError()

        rgb_feature_dim = 0
        self.grids = nn.ModuleList()
        grid_config = self.config[0]
        for si in range(num_scenes):
            _grids = nn.ModuleList()
            for res_idx, res in enumerate(self.multiscale_res):
                if "feature_dim" in grid_config:
                    raise ValueError(f"use_F is False but found 'feature_dim' key in grid-config.")
                config = grid_config.copy()
                config['resolution'] = [int(r * res) for r in config['resolution'][:3]]

                gpdesc = self.init_grid_param(
                    grid_nd=config['grid_dimensions'],
                    resolution=config['resolution'],
                    out_features=config['output_coordinate_dim'],
                    input_features=3,
                    is_video=False, use_F=False, is_density=False)
                self.set_resolution(gpdesc.reso, grid_id=si)
                if self.concat_features:
                    rgb_feature_dim += gpdesc.grid_coefs[-1].shape[1]
                else:
                    rgb_feature_dim = gpdesc.grid_coefs[-1].shape[1]
                _grids.append(gpdesc.grid_coefs)
            self.grids.append(_grids)

        self.decoder = self.init_decoder()

        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, "
                 f"decoder: {self.decoder}, use-F: {self.use_F}, "
                 f"concat-features: {self.concat_features}, "
                 f"feature-dim: {self.feature_dim}, "
                 f"aabb: {self.aabb(0)}.")
        log.info(f"Model grids: {self.scene_grids}")

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         num_levels: Optional[int] = None
                         ) -> torch.Tensor:
        grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grid_info = self.config[0]
        multiscale_interp = self.interpolate_ms_features(
            pts, grids, grid_info, concat_features=self.concat_features,
            num_levels=num_levels)
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
                0, len(dset.camtoworlds), (pts.shape[0],), device=pts.device
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

    def query_density(self,
                      pts: torch.Tensor,
                      grid_id: int,
                      return_feat: bool = False,
                      num_levels: Optional[int] = None
                      ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pts_norm = self.normalize_coords(pts, grid_id)
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, grid_id, num_levels=num_levels)
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
        grid_id: int,
    ):
        num_levels = None
        if self.train_every_scale and self.training:
            all_levels = np.arange(1, len(self.scene_grids[grid_id]) + 1)
            level_p = (all_levels ** 3).astype(float)
            level_p /= level_p.sum()
            num_levels = np.random.choice(all_levels, p=level_p)
        density, embedding = self.query_density(
                pts, grid_id, return_feat=True, num_levels=num_levels)
        rgb = self.decoder.compute_color(embedding, rays_d=rays_d)
        return rgb, density

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
