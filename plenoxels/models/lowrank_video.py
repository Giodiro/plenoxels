import itertools
import logging as log
from typing import Dict, List, Union, Sequence, Tuple

import torch
import torch.nn.functional as F

from plenoxels.models.utils import grid_sample_wrapper, compute_plane_tv, compute_line_tv, raw2alpha
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel
from ..raymarching.raymarching import RayMarcher
from ..ops.activations import trunc_exp


class LowrankVideo(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: torch.Tensor,  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 is_ndc: bool,
                 is_contracted: bool,
                 sh: bool,
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
        self.is_contracted = is_contracted
        self.raymarcher = RayMarcher(**self.extra_args)
        self.sh = sh
        self.density_act = trunc_exp
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
        grid_data = self.time_coef.data
        self.time_coef = torch.nn.Parameter(
            F.interpolate(self.time_coef.data[:,None,:], size=(new_reso), mode='linear', align_corners=True).squeeze())
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

    def forward(self, rays_o, rays_d, timestamps, bg_color, channels: Sequence[str] = ("rgb", "depth")):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        """
        
        rm_out = self.raymarcher.get_intersections2(
            rays_o, rays_d, self.aabb(0), self.resolution(0), perturb=self.training,
            is_ndc=self.is_ndc, is_contracted=self.is_contracted)
        rays_d = rm_out["rays_d"]                   # [n_rays, 3]
        intersection_pts = rm_out["intersections"]  # [n_rays, n_intrs, 3]
        mask = rm_out["mask"]                       # [n_rays, n_intrs]
        z_vals = rm_out["z_vals"]                   # [n_rays, n_intrs]
        deltas = rm_out["deltas"]                   # [n_rays, n_intrs]
        n_rays, n_intrs = intersection_pts.shape[:2]
        
        times = timestamps[:, None].repeat(1, n_intrs)[mask]  # [n_rays, n_intrs]

        # Normalization (between [-1, 1])
        intersection_pts = self.normalize_coords(intersection_pts, 0)
        times = (times * 2 / self.len_time) - 1

        # compute features and render
        features = self.compute_features(intersection_pts[mask], times)

        rays_d_rep = rays_d.view(-1, 1, 3).expand(intersection_pts.shape)
        masked_rays_d_rep = rays_d_rep[mask]

        density_masked = self.density_act(self.decoder.compute_density(features, rays_d=masked_rays_d_rep) - 1)
        density = torch.zeros(n_rays, n_intrs, device=intersection_pts.device, dtype=density_masked.dtype)
        density[mask] = density_masked.view(-1)

        alpha, weight, transmission = raw2alpha(density, deltas)  # Each is shape [batch_size, n_samples]

        rgb_masked = self.decoder.compute_color(features, rays_d=masked_rays_d_rep)
        rgb = torch.zeros(n_rays, n_intrs, 3, device=intersection_pts.device, dtype=rgb_masked.dtype)
        rgb[mask] = rgb_masked
        rgb = torch.sigmoid(rgb)

        # Confirmed that torch.sum(weight, -1) matches 1-transmission[:, -1]
        acc_map = 1 - transmission[:, -1]

        outputs = {}
        if "rgb" in channels:
            rgb_map = torch.sum(weight[..., None] * rgb, -2)
            if bg_color is None:
                pass
            else:
                rgb_map = rgb_map + (1.0 - acc_map[..., None]) * bg_color
            outputs["rgb"] = rgb_map
        if "depth" in channels:
            depth_map = torch.sum(weight * z_vals, -1)  # [batch_size]
            depth_map = depth_map + (1.0 - acc_map) * rays_d[..., -1]  # Maybe the rays_d is to transform ray depth to absolute depth?
            outputs["depth"] = depth_map

        return outputs

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