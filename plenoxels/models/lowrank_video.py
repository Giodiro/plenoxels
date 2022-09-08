import itertools
import math
from typing import Dict, List, Union
import logging as log

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render

from plenoxels.models.utils import grid_sample_wrapper
from .decoders import NNDecoder, SHDecoder
from ..ops.activations import trunc_exp
from ..raymarching.raymarching import RayMarcher


class LowrankVideo(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.aabb = aabb
        self.len_time = len_time
        self.extra_args = kwargs
        self.raymarcher = RayMarcher(**kwargs)

        self.features = None
        feature_dim = None
        self.grids = nn.ParameterList()
        for li, grid_config in enumerate(self.config):
            if "feature_dim" in grid_config:
                in_dim = grid_config["input_coordinate_dim"]
                reso = grid_config["resolution"]
                self.features = nn.Parameter(nn.init.normal_(
                        torch.empty([grid_config["feature_dim"]] + [reso] * in_dim),
                        mean=0.0, std=grid_config["init_std"]))
                feature_dim = grid_config["feature_dim"]
            else:
                in_dim = grid_config["input_coordinate_dim"]
                out_dim = grid_config["output_coordinate_dim"]
                grid_nd = grid_config["grid_dimensions"]
                reso = grid_config["resolution"]
                rank = grid_config["rank"]
                num_comp = math.comb(in_dim, grid_nd)
                # Configuration correctness checks
                if li == 0:
                    assert in_dim == 3
                assert out_dim in {1, 2, 3, 4}
                assert grid_nd <= in_dim
                if grid_nd == in_dim:
                    assert rank == 1
                time_reso = grid_config["time_reso"]
                time_rank = grid_config["time_rank"]
                self.grids.append(
                    nn.Parameter(nn.init.normal_(
                        torch.empty([num_comp, rank * out_dim * time_rank] + [reso] * grid_nd),
                        mean=0.0, std=grid_config["init_std"]))
                )
                self.grids.append(
                    nn.Parameter(nn.init.normal_(
                        torch.empty([out_dim * time_rank, time_reso]),
                        mean=0.0, std=grid_config["init_std"]))
                )
        assert len(self.grids) == 2 # For now, only allow a single index grid and a single feature grid, not multiple layers
        self.decoder = NNDecoder(feature_dim=feature_dim, sigma_net_width=64, sigma_net_layers=1)
        log.info(f"Initialized LowrankVideo.")

    @staticmethod
    def get_coo_plane(coords, dim):
        """
        :param coords:
            torch tensor [n, input d]
        :param dim:
        :return:
            torch tensor [num_comp, n, dim]
        """
        coo_combs = list(itertools.combinations(range(coords.shape[-1]), dim))
        return coords[..., coo_combs].transpose(0, 1)

    def compute_features(self, pts, timestamps):
        [grid_space, grid_time] = self.grids  # space: [3, rank * F_dim * time_rank, reso, reso], time: [time_rank*F_dim, time_reso]
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

        interp = torch.cat([pts, timestamps[:,None]], dim=-1)  # [n, 4] xyzt
        # Interpolate in time
        interp_time = grid_sample_wrapper(grid_time.unsqueeze(0), interp[:,-1].unsqueeze(0).unsqueeze(-1))  # [n, F_dim * time_rank]
        interp_time = interp_time.view(len(interp), -1, level_info['time_rank'])  # [n, F_dim, time_rank]
        # Interpolate in space
        coo_plane = self.get_coo_plane(
            interp[:,0:3],
            level_info.get("grid_dimensions", level_info["input_coordinate_dim"])
        )  # [3, n, grid_dim]
        interp_space = grid_sample_wrapper(grid_space, coo_plane).view(
            grid_space.shape[0], -1, level_info["output_coordinate_dim"], level_info["time_rank"], level_info["rank"])  # [3, n, F_dim, time_rank, rank]
        interp_space = interp_space.prod(dim=0).sum(dim=-1)  # [n, F_dim, time_rank]
        # Combine space and time
        interp = (interp_space * interp_time).sum(dim=-1)  # [n, F_dim]
        return grid_sample_wrapper(self.features, interp)

    def normalize_coord(self, pts):
        """
        break-down of the normalization steps
        1. pts - aabb[0] => [0, a1-a0]
        2. / (a1 - a0) => [0, 1]
        3. * 2 => [0, 2]
        4. - 1 => [-1, 1]
        """
        return (pts - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1

    def forward(self, rays_o, rays_d, timestamps, bg_color):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        """

        intersection_pts, ridx, boundary, deltas, times = self.raymarcher.get_intersections(
            rays_o, rays_d, self.aabb, perturb=self.training, timestamps=timestamps)
        n_rays = rays_o.shape[0]
        dev = rays_o.device

        # mask has shape [batch, n_intrs]
        # intersection_pts has shape [n_valid_intrs, 3]

        # Normalization (between [-1, 1])
        intersection_pts = self.normalize_coord(intersection_pts)
        times = (times * 2 / self.len_time) - 1
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
        # rays_d in the packed format (essentially repeated a number of times)
        rays_d_rep = rays_d.index_select(0, ridx)

        # compute features and render
        features = self.compute_features(intersection_pts, times)
        density_masked = trunc_exp(self.decoder.compute_density(features, rays_d=rays_d_rep))
        rgb_masked = torch.sigmoid(self.decoder.compute_color(features, rays_d=rays_d_rep))

        # Compute optical thickness
        tau = density_masked.reshape(-1, 1) * deltas.reshape(-1, 1)
        # ridx_hit are the ray-IDs at the first intersection (the boundary).
        ridx_hit = ridx[boundary]
        # Perform volumetric integration
        ray_colors, transmittance = spc_render.exponential_integration(
            rgb_masked.reshape(-1, 3), tau, boundary, exclusive=True)
        alpha = spc_render.sum_reduce(transmittance, boundary)
        # Blend output color with background
        if isinstance(bg_color, torch.Tensor) and bg_color.shape == (n_rays, 3):
            rgb = bg_color
            color = ray_colors + (1.0 - alpha) * bg_color[ridx_hit.long(), :]
        else:
            rgb = torch.full((n_rays, 3), bg_color, dtype=ray_colors.dtype, device=dev)
            color = ray_colors + (1.0 - alpha) * bg_color
        rgb[ridx_hit.long(), :] = color

        return rgb

    def get_params(self, lr):
        params = [
            {"params": self.decoder.parameters(), "lr": lr},
            {"params": self.grids.parameters(), "lr": lr},
            {"params": self.features, "lr": lr},
        ]
        return params