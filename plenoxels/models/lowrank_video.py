import itertools
import logging as log
import math
from typing import Dict, List, Union, Tuple

import torch
import torch.nn.functional as F

from plenoxels.models.utils import grid_sample_wrapper, compute_plane_tv, compute_line_tv
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel
from ..ops.activations import trunc_exp


class LowrankVideo(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: torch.Tensor,  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 is_ndc: bool,
                 sh: bool,
                 render_n_samples: int,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.set_aabb(aabb, 0)
        self.len_time = len_time
        self.extra_args = kwargs
        self.is_ndc = is_ndc
        self.sh = sh
        self.render_n_samples = render_n_samples
        self.density_act = lambda x: trunc_exp(x - 1)
        self.pt_min = torch.nn.Parameter(torch.tensor(-1.0))
        self.pt_max = torch.nn.Parameter(torch.tensor(1.0))

        # For now, only allow a single index grid and a single feature grid, not multiple layers
        assert len(self.config) == 2
        for li, grid_config in enumerate(self.config):
            if "feature_dim" in grid_config:
                self.features = self.init_features_param(grid_config, self.sh)
                self.feature_dim = self.features.shape[0]
            else:
                gpdesc = self.init_grid_param(grid_config, is_video=True, grid_level=li)
                self.set_resolution(gpdesc.reso, 0)
                self.grids = gpdesc.grid_coefs
                self.time_coef = gpdesc.time_coef  # [out_dim * rank, time_reso]
        if self.sh:
            self.decoder = SHDecoder(feature_dim=self.feature_dim)
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        log.info(f"Initialized LowrankVideo. "
                 f"time-reso={self.time_coef.shape[1]} - decoder={self.decoder}")

    @torch.no_grad()
    def upsample_time(self, new_reso):
        old_reso = self.time_coef.shape[-1]
        self.time_coef = torch.nn.Parameter(
            F.interpolate(
                self.time_coef.data[:, None, :],
                size=(new_reso),
                mode='linear',
                align_corners=True)
             .squeeze()
        )
        log.info(f"Upsampled time resolution from {old_reso} to {new_reso}")

    def compute_features(self,
                         pts,
                         timestamps,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        grid_space = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
        grid_time = self.time_coef  # time: [rank * F_dim, time_reso]
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

        # Interpolate in time
        interp_time = grid_sample_wrapper(grid_time.unsqueeze(0), timestamps[:, None])  # [n, F_dim * rank]
        interp_time = interp_time.view(-1, level_info["output_coordinate_dim"], level_info["rank"][0])  # [n, F_dim, rank]
        # Interpolate in space
        interp = pts
        coo_combs = list(itertools.combinations(
            range(interp.shape[-1]),
            level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))
        interp_space = None  # [n, F_dim, rank]
        for ci, coo_comb in enumerate(coo_combs):
            if interp_space is None:
                interp_space = (
                    grid_sample_wrapper(grid_space[ci], interp[..., coo_comb]).view(
                        -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
            else:
                interp_space = interp_space * (
                    grid_sample_wrapper(grid_space[ci], interp[..., coo_comb]).view(
                        -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
        # Combine space and time over rank
        interp = (interp_space * interp_time).mean(dim=-1)  # [n, F_dim]
        # Learned normalization
        if interp.numel() > 0:
            interp = (interp - self.pt_min) / (self.pt_max - self.pt_min)
            interp = interp * 2 - 1
        out = grid_sample_wrapper(self.features, interp).view(-1, self.feature_dim)
        if return_coords:
            return out, interp
        return out

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
        step_size = self.step_size(self.render_n_samples)
        opacity = density * step_size
        return opacity

    def query_density(self, pts: torch.Tensor, timestamps: torch.Tensor, return_feat: bool = False):
        pts_norm = self.normalize_coords(pts, 0)
        timestamps_norm = (timestamps * 2 / self.len_time) - 1
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, timestamps_norm, return_coords=False)
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
        params = [
            {"params": self.decoder.parameters(), "lr": lr},
            {"params": self.grids.parameters(), "lr": lr},
            {"params": [self.features, self.pt_min, self.pt_max], "lr": lr},
        ]
        return params

    def compute_plane_tv(self):
        grid_space = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
        grid_time = self.time_coef  # time: [rank * F_dim, time_reso]
        total = 0
        for grid in grid_space:
            total += compute_plane_tv(grid)
        total += compute_line_tv(grid_time)
        return total
