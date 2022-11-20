import itertools
import logging as log
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn

from plenoxels.models.utils import grid_sample_wrapper, init_density_activation
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel


class LowrankLearnableHash(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],
                 sh: bool,
                 use_F: bool,
                 density_activation: str,
                 render_n_samples: int,
                 multiscale_res: List[int] = [1],
                 num_scenes: int = 1,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.set_aabb(aabb)
        self.sh = sh
        self.use_F = use_F
        self.density_act = init_density_activation(density_activation)
        self.cone_angle = kwargs.get('cone_angle', 0.0)
        # render_n_samples: number of intersections in each ray. Used for computing step-size.
        self.render_n_samples = render_n_samples
        self.extra_args = kwargs

        if self.use_F:
            raise NotImplementedError()

        self.scene_grids = nn.ModuleList()
        grid_config = self.config[0]
        for si in range(num_scenes):
            grids = nn.ModuleList()
            for res in self.multiscale_res:
                if "feature_dim" in grid_config:
                    raise ValueError(f"use_F is False but found 'feature_dim' key in grid-config.")
                config = grid_config.copy()
                config['resolution'] = [r * res for r in config['resolution'][:3]]
                gpdesc = self.init_grid_param(
                    config, is_video=False, grid_level=0, use_F=False)
                self.set_resolution(gpdesc.reso, grid_id=si)
                self.feature_dim = gpdesc.grid_coefs[-1].shape[1]
                grids.append(gpdesc.grid_coefs)
            self.scene_grids.append(grids)

        if self.sh:
            self.decoder = SHDecoder(feature_dim=self.feature_dim)
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)

        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, "
                 f"decoder: {self.decoder}, use-F: {self.use_F}")
        log.info(f"Model grids: {self.scene_grids}")

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         ) -> torch.Tensor:
        grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grid_info = self.config[0]

        grid: nn.ParameterList
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]),
            grid_info.get("grid_dimensions", grid_info["input_coordinate_dim"]))
        )
        multi_scale_interp = 0#torch.zeros_like(pts[0, 0])
        for scale_id, grid in enumerate(grids):
            interp = 1
            for ci, coo_comb in enumerate(coo_combs):
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                    .view(-1, self.feature_dim)
                )
                # compute product
                interp = interp * interp_out_plane
            # sum over scales
            multi_scale_interp = multi_scale_interp + interp
        return multi_scale_interp

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

    def query_density(self, pts: torch.Tensor, grid_id: int, return_feat: bool = False):
        pts_norm = self.normalize_coords(pts, grid_id)
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, grid_id)
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
        grid_id: int,
    ):
        density, embedding = self.query_density(rays_o, grid_id, return_feat=True)
        rgb = self.decoder.compute_color(embedding, rays_d=rays_d)
        return rgb, density

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
