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


class LowrankAppearance(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: torch.Tensor,  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 is_ndc: bool,
                 is_contracted: bool,
                 lookup_time: bool,
                 sh: bool,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.set_aabb(aabb, 0)
        self.len_time = len_time  # maximum timestep - used for normalization
        self.extra_args = kwargs
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.lookup_time = lookup_time
        self.raymarcher = RayMarcher(**self.extra_args)
        self.sh = sh
        self.density_act = trunc_exp
        self.pt_min = torch.nn.Parameter(torch.tensor(-1.0))
        self.pt_max = torch.nn.Parameter(torch.tensor(1.0))
        self.use_F = self.extra_args["use_F"]
        self.appearance_code = True

        # For now, only allow a single index grid and a single feature grid, not multiple layers
        assert len(self.config) == 2
        for li, grid_config in enumerate(self.config):
            if "feature_dim" in grid_config:
                self.features = None
                self.feature_dim = grid_config["feature_dim"]
                if self.use_F:
                    self.features = self.init_features_param(grid_config, self.sh)
                    self.feature_dim = self.features.shape[0]
            else:
                gpdesc = self.init_grid_param(grid_config, is_video=False, is_appearance=True, grid_level=li, use_F=self.use_F)
                self.set_resolution(gpdesc.reso, 0)
                self.register_buffer(
                    'time_resolution', torch.tensor(gpdesc.time_coef.shape[-1], dtype=torch.int32))
                self.grids = gpdesc.grid_coefs
                self.time_coef = gpdesc.time_coef  # [out_dim * rank, time_reso]
        if self.sh:
            self.decoder = SHDecoder(feature_dim=self.feature_dim)
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        log.info(f"Initialized LowrankAppearance. "
                 f"time-reso={self.time_coef.shape[1]} - decoder={self.decoder}")

    def compute_features(self,
                         pts,
                         timestamps,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        grid_space = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
        grid_time = self.time_coef  # time: [rank * F_dim, time_reso]
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

        dim = level_info["output_coordinate_dim"] - 1 if level_info["output_coordinate_dim"] == 28 else level_info["output_coordinate_dim"]

        interp_time = grid_time[:, timestamps.long()].unsqueeze(0).repeat(pts.shape[0], 1)  # [n, F_dim * rank]
        interp_time = interp_time.view(-1, dim, level_info["rank"][0])  # [n, F_dim, rank]
        
        # add density one to appearance code
        if level_info["output_coordinate_dim"] == 28:
             interp_time = torch.cat([interp_time, torch.ones_like(interp_time[:, 0:1, :])], dim=1)
        
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

        if self.use_F:
            # Learned normalization
            if interp.numel() > 0:
                interp = (interp - self.pt_min) / (self.pt_max - self.pt_min)
                interp = interp * 2 - 1

            out = grid_sample_wrapper(self.features, interp).view(-1, self.feature_dim)
        else:
            out = interp
        if return_coords:
            return out, interp
        return out

    def forward(self, rays_o, rays_d, timestamps, bg_color, channels: Sequence[str] = ("rgb", "depth"), near_far=None, global_translation=torch.tensor([0, 0, 0]), global_scale=torch.tensor([1, 1, 1])):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """

        rm_out = self.raymarcher.get_intersections2(
            rays_o, rays_d, self.aabb(0), self.resolution(0), perturb=self.training,
            is_ndc=self.is_ndc, is_contracted=self.is_contracted, near_far=near_far,
            global_translation=global_translation, global_scale=global_scale)
        rays_d = rm_out["rays_d"]                   # [n_rays, 3]
        intersection_pts = rm_out["intersections"]  # [n_rays, n_intrs, 3]
        mask = rm_out["mask"]                       # [n_rays, n_intrs]
        z_vals = rm_out["z_vals"]                   # [n_rays, n_intrs]
        deltas = rm_out["deltas"]                   # [n_rays, n_intrs]
        n_rays, n_intrs = intersection_pts.shape[:2]

        # assumes all rays are sampled at the same time
        # speed up look up a quite a bit
        times = timestamps[0]

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
        outputs["deltas"] = deltas
        outputs["weight"] = weight
        outputs["midpoint"] = rm_out["z_mids"]

        return outputs

    def get_params(self, lr):
        if self.use_F:
            params = [
                {"params": self.decoder.parameters(), "lr": lr},
                {"params": self.grids.parameters(), "lr": lr},
                {"params": [self.features, self.pt_min, self.pt_max], "lr": lr},
                {"params": self.time_coef, "lr": lr},
            ]
        else:
            params = [
                {"params": self.decoder.parameters(), "lr": lr},
                {"params": self.grids.parameters(), "lr": lr},
                {"params": self.time_coef, "lr": lr},
                # {"params": [self.features, self.pt_min, self.pt_max], "lr": lr},
            ]
        return params

    def compute_plane_tv(self):
        grid_space = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
        #grid_time = self.time_coef  # time: [rank * F_dim, time_reso]
        
        if len(grid_space) == 6:
            # only use tv on spatial planes
            grid_ids = [0,1,3]
        else:
            grid_ids = list(range(len(grid_space)))
        
        total = 0
        for grid_id in grid_ids:
            grid = grid_space[grid_id]
            total += compute_plane_tv(grid)
        #total += compute_line_tv(grid_time)
        return total
