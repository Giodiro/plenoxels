import itertools
import math
from typing import Dict, List, Union
import logging as log

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render

from plenoxels.models.utils import get_intersections, grid_sample_wrapper
from .decoders import NNDecoder, SHDecoder
from ..ops.activations import trunc_exp


class LowrankLearnableHash(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 radi: List[float],
                 n_intersections: int,
                 num_scenes: int = 1):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.radi = radi
        self.n_intersections = n_intersections

        self.scene_grids = nn.ModuleList()
        self.features = None
        feature_dim = None
        for si in range(num_scenes):
            grids = nn.ParameterList()
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config and si == 0:
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
                    grids.append(
                        nn.Parameter(nn.init.normal_(
                            torch.empty([num_comp, out_dim * rank] + [reso] * grid_nd),
                            mean=0.0, std=grid_config["init_std"]))
                    )
            self.scene_grids.append(grids)
        self.decoder = NNDecoder(feature_dim=feature_dim, sigma_net_width=64, sigma_net_layers=1)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes. "
                 f"Ray-marching will use {n_intersections} samples.")

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

    def compute_features(self, pts, grid_id):
        grids = self.scene_grids[grid_id]
        grids_info = self.config

        interp = pts
        for level_info, grid in zip(grids_info, grids):
            if "feature_dim" in level_info:
                continue
            coo_plane = self.get_coo_plane(
                interp,
                level_info.get("grid_dimensions", level_info["input_coordinate_dim"])
            )
            interp = grid_sample_wrapper(grid, coo_plane).view(
                grid.shape[0], -1, level_info["output_coordinate_dim"], level_info["rank"])
            interp = interp.prod(dim=0).sum(dim=-1)
        return grid_sample_wrapper(self.features, interp)

    def forward(self, rays_o, rays_d, bg_color, grid_id=0):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        intersection_pts, ridx, boundary, deltas = get_intersections(
            rays_o, rays_d, self.radi[grid_id], self.n_intersections, perturb=self.training)
        n_rays = rays_o.shape[0]
        dev = rays_o.device

        # mask has shape [batch, n_intrs]
        # intersection_pts has shape [n_valid_intrs, 3]

        # Normalization (between [-1, 1])
        intersection_pts = intersection_pts / self.radi[grid_id]
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
        # rays_d in the packed format (essentially repeated a number of times)
        rays_d_rep = rays_d.index_select(0, ridx)

        # compute features and render
        features = self.compute_features(intersection_pts, grid_id)
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
            {"params": self.scene_grids.parameters(), "lr": lr},
            {"params": self.features, "lr": lr},
        ]
        return params
