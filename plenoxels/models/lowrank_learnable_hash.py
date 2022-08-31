import itertools
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from plenoxels.models.utils import get_intersections, grid_sample_wrapper
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha
from .decoders import NNDecoder, SHDecoder


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LowrankLearnableHash(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 radius: float,
                 n_intersections: int,
                 num_scenes: int = 1):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.radius = radius
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
                    assert out_dim in {2, 3, 4}
                    assert grid_nd <= in_dim
                    if grid_nd == in_dim:
                        assert rank == 1
                    grids.append(
                        nn.Parameter(nn.init.normal_(
                            torch.empty([num_comp, out_dim * rank] + [reso] * grid_nd),
                            mean=0.0, std=grid_config["init_std"]))
                    )
            self.scene_grids.append(grids)
        self.renderer = NNDecoder(feature_dim=feature_dim, sigma_net_width=64, sigma_net_layers=1)

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

    def forward(self, rays_o, rays_d, grid_id=0):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        intersection_pts, intersections, mask = get_intersections(
            rays_o, rays_d, self.radius, self.n_intersections, perturb=self.training)

        ridx = torch.arange(0, rays_d.shape[0], device=rays_o.device)
        ridx = ridx[..., None].repeat(1, mask.shape[1])[mask]

        # mask has shape [batch, n_intrs]
        intersection_pts = intersection_pts[mask]  # [n_valid_intrs, 3] puts together the valid intrs from all rays
        # Normalization (between [-1, 1])
        intersection_pts = intersection_pts / self.radius
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
        rays_d_rep = rays_d.index_select(0, ridx)

        # compute features and render
        features = self.compute_features(intersection_pts, grid_id)

        density_masked = torch.relu(self.renderer.compute_density(features, rays_d=rays_d_rep))
        density = torch.zeros(mask.shape[0], mask.shape[1], dtype=density_masked.dtype, device=density_masked.device)
        density.masked_scatter_(mask, density_masked)
        alpha, abs_light = sigma2alpha(density, intersections, rays_d)

        rgb_masked = self.renderer.compute_color(features, rays_d_rep)
        rgb = torch.zeros(mask.shape[0], mask.shape[1], 3, dtype=rgb_masked.dtype, device=rgb_masked.device)
        rgb.masked_scatter_(mask.unsqueeze(-1), rgb_masked)
        rgb = shrgb2rgb(rgb, abs_light, True)
        return rgb

    def get_params(self, lr):
        params = [
            {"params": self.renderer.parameters(), "lr": lr},
            {"params": self.scene_grids.parameters(), "lr": lr},
            {"params": self.features, "lr": lr},
        ]
        return params
