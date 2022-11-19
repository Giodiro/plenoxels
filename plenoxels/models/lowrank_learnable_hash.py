import itertools
import logging as log
import math
from typing import Dict, List, Union, Tuple

import torch
import torch.nn as nn

from plenoxels.models.utils import grid_sample_wrapper
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel
from ..ops.activations import trunc_exp


class LowrankLearnableHash(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],
                 sh: bool,
                 render_n_samples: int,
                 num_scenes: int = 1,
                 use_F: bool = True,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.set_aabb(aabb)
        self.extra_args = kwargs
        self.sh = sh
        self.use_F = use_F
        self.cone_angle = kwargs.get('cone_angle', 0.0)
        # render_n_samples: number of intersections in each ray. Used for computing step-size.
        self.render_n_samples = render_n_samples
        self.transfer_learning = self.extra_args["transfer_learning"]

        self.density_act = lambda x: trunc_exp(x - 1)
        self.pt_min, self.pt_max = None, None
        if self.use_F:
            self.pt_min = torch.nn.Parameter(torch.tensor(-1.0))
            self.pt_max = torch.nn.Parameter(torch.tensor(1.0))

        self.scene_grids = nn.ModuleList()
        for si in range(num_scenes):
            grids = nn.ModuleList()
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config and si == 0 and use_F:
                    self.features = self.init_features_param(grid_config, self.sh)
                    self.feature_dim = self.features.shape[0]
                else:
                    gpdesc = self.init_grid_param(
                        grid_config, is_video=False, grid_level=li, use_F=use_F)
                    if li == 0:
                        self.set_resolution(gpdesc.reso, grid_id=si)
                    if not self.use_F:
                        # shape[1] is out_dim * rank
                        self.feature_dim = gpdesc.grid_coefs[-1].shape[1] // grid_config['rank'][0]
                    grids.append(gpdesc.grid_coefs)
            self.scene_grids.append(grids)
        if self.sh:
            self.decoder = SHDecoder(feature_dim=self.feature_dim)
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, "
                 f"decoder: {self.decoder}, use-F: {self.use_F}")

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grids_info = self.config

        interp = pts
        grid: nn.ParameterList
        for level_info, grid in zip(grids_info, grids):
            if "feature_dim" in level_info:
                continue
            coo_combs = list(itertools.combinations(
                range(interp.shape[-1]),
                level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))
            interp_out = None
            for ci, coo_comb in enumerate(coo_combs):
                if interp_out is None:
                    interp_out = (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
                else:
                    interp_out = interp_out * (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
            interp = interp_out.mean(dim=-1)

        # Learned normalization
        if self.use_F:
            if interp.numel() > 0:
                interp = (interp - self.pt_min) / (self.pt_max - self.pt_min)
                interp = interp * 2 - 1
            out = grid_sample_wrapper(self.features.to(dtype=interp.dtype), interp).view(-1, self.feature_dim)
        else:
            out = interp

        if return_coords:
            return out, interp
        return out

    def step_size(self, n_samples: int, grid_id: int):
        aabb = self.aabb(grid_id)
        return (
            (aabb[1] - aabb[0]).max()
            * math.sqrt(3)
            / n_samples
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
        params = [
            {"params": self.scene_grids.parameters(), "lr": lr},
        ]
        if self.use_F:
            params.append(
                {"params": [self.pt_min, self.pt_max], "lr": lr}
            )
        if not self.transfer_learning:
            params.append(
                {"params": self.decoder.parameters(), "lr": lr}
            )
            if self.use_F:
                params.append({"params": self.features, "lr": lr})
        return params
