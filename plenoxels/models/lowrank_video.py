import collections.abc
import itertools
import math
from typing import Dict, List, Union, Sequence, Optional
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
                 aabb: torch.Tensor,  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 is_ndc: bool,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.register_buffer("aabb", aabb)
        self.len_time = len_time
        self.extra_args = kwargs
        self.is_ndc = is_ndc
        self.raymarcher = RayMarcher(**self.extra_args)

        self.features = None
        feature_dim = None
        self.grids = nn.ModuleList()
        for li, grid_config in enumerate(self.config):
            if "feature_dim" in grid_config:
                reso: List[int] = grid_config["resolution"]
                try:
                    in_dim = len(reso)
                except AttributeError:
                    raise ValueError("Configuration incorrect: resolution must be a list.")
                assert in_dim == grid_config["input_coordinate_dim"]
                self.features = nn.Parameter(nn.init.normal_(
                        torch.empty([grid_config["feature_dim"]] + reso),
                        mean=0.0, std=grid_config["init_std"]))
                feature_dim = grid_config["feature_dim"]
            else:
                out_dim = grid_config["output_coordinate_dim"]
                grid_nd = grid_config["grid_dimensions"]
                reso: List[int] = grid_config["resolution"]
                try:
                    in_dim = len(reso)
                except AttributeError:
                    raise ValueError("Configuration incorrect: resolution must be a list.")
                num_comp = math.comb(in_dim, grid_nd)
                rank: Sequence[int] = to_list(grid_config["rank"], num_comp, "rank")
                grid_config["rank"] = rank
                # Configuration correctness checks
                assert in_dim == grid_config["input_coordinate_dim"]
                if li == 0:
                    assert in_dim == 3
                    self.register_buffer("resolution", torch.tensor(reso, dtype=torch.long))
                assert out_dim in {1, 2, 3, 4}
                assert grid_nd <= in_dim
                if grid_nd == in_dim:
                    assert all(r == 1 for r in rank)
                time_reso = grid_config["time_reso"]
                time_rank = grid_config["time_rank"]
                coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
                grid_coefs = nn.ParameterList()
                for ci, coo_comb in enumerate(coo_combs):
                    grid_coefs.append(
                        torch.nn.Parameter(nn.init.normal_(torch.empty(
                            [1, out_dim * rank[ci] * time_rank] + [reso[cc] for cc in coo_comb]
                        ), mean=0.0, std=grid_config["init_std"])))
                time_coef = nn.Parameter(nn.init.normal_(
                                torch.empty([out_dim * time_rank, time_reso]),
                                mean=0.0, std=grid_config["init_std"]))
                self.grids.append(grid_coefs)
                self.grids.append(nn.ParameterList([time_coef]))
        assert len(self.grids) == 2  # For now, only allow a single index grid and a single feature grid, not multiple layers
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
        grid_space, grid_time = self.grids  # space: [3, rank * F_dim * time_rank, reso, reso], time: [time_rank*F_dim, time_reso]
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

        # Interpolate in time
        grid_time = grid_time[0]  # we need it to be a length-1 ParameterList
        interp_time = grid_sample_wrapper(grid_time.unsqueeze(0), timestamps[:, None])  # [n, F_dim * time_rank]
        interp_time = interp_time.view(-1, level_info["output_coordinate_dim"], level_info['time_rank'])  # [n, F_dim, time_rank]
        # Interpolate in space
        interp = pts
        coo_combs = list(itertools.combinations(
            range(interp.shape[-1]),
            level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))
        interp_space = None
        for ci, coo_comb in enumerate(coo_combs):
            if interp_space is None:
                interp_space = (
                    grid_sample_wrapper(grid_space[ci], interp[..., coo_comb]).view(
                        -1, level_info["output_coordinate_dim"], level_info["time_rank"], level_info["rank"][ci]))
            else:
                interp_space = interp_space * (
                    grid_sample_wrapper(grid_space[ci], interp[..., coo_comb]).view(
                        -1, level_info["output_coordinate_dim"], level_info["time_rank"], level_info["rank"][ci]))
        interp_space = interp_space.sum(dim=-1)  # [N, F_dim, time_rank]
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

        rm_out = self.raymarcher.get_intersections2(
            rays_o, rays_d, self.aabb, self.resolution, perturb=self.training,
            is_ndc=self.is_ndc)
        rays_d = rm_out["rays_d"]
        deltas = rm_out["deltas"]
        intersection_pts = rm_out["intersections"]
        ridx = rm_out["ridx"]
        boundary = rm_out["boundary"]

        times = timestamps[:, None].repeat(1, rm_out["mask"].shape[1])[rm_out["mask"]]

        n_rays = rays_o.shape[0]
        dev = rays_o.device

        # mask has shape [batch, n_intrs]
        # intersection_pts has shape [n_valid_intrs, 3]

        # Normalization (between [-1, 1])
        intersection_pts = self.normalize_coord(intersection_pts)
        times = (times * 2 / self.len_time) - 1

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
            rgb_masked.reshape(-1, 3), tau, boundary, exclusive=True)  # transmittance is [n_intersections, 1]
        alpha = spc_render.sum_reduce(transmittance, boundary)  # [n_valid_rays, 1]
        # Compute depth as a weighted sum over z values according to absorption/transmission
        z_vals = spc_render.cumsum(deltas[:, None], boundary)
        real_depth = spc_render.sum_reduce(transmittance * z_vals, boundary)  # [n_valid_rays, 1]
        depth = torch.full((n_rays, 1), 100, dtype=ray_colors.dtype, device=dev)
        depth[ridx_hit.long()] = real_depth  # [n_rays, 1]

        # Blend output color with background
        if isinstance(bg_color, torch.Tensor) and bg_color.shape == (n_rays, 3):
            rgb = bg_color
            color = ray_colors + (1.0 - alpha) * bg_color[ridx_hit.long(), :]
        else:
            rgb = torch.full((n_rays, 3), bg_color, dtype=ray_colors.dtype, device=dev)
            color = ray_colors + (1.0 - alpha) * bg_color
        rgb[ridx_hit.long(), :] = color  # [n_rays, 3]

        return rgb, depth

    def get_params(self, lr):
        params = [
            {"params": self.decoder.parameters(), "lr": lr},
            {"params": self.grids.parameters(), "lr": lr},
            {"params": self.features, "lr": lr},
        ]
        return params


def to_list(el, list_len, name: Optional[str] = None) -> Sequence:
    if not isinstance(el, collections.abc.Sequence):
        return [el] * list_len
    if len(el) != list_len:
        raise ValueError(f"Length of {name} is incorrect. Expected {list_len} but found {len(el)}")
    return el
