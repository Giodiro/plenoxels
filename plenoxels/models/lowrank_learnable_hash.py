import collections.abc
import itertools
import math
from typing import Dict, List, Union, Optional, Sequence
import logging as log

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render

from plenoxels.models.utils import grid_sample_wrapper
from .decoders import NNDecoder, SHDecoder
from ..ops.activations import trunc_exp
from ..raymarching.raymarching import RayMarcher


class LowrankLearnableHash(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],
                 is_ndc: bool,
                 num_scenes: int = 1,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        # aabb needs to be BufferList (but BufferList doesn't exist so we emulate it)
        for i, p in enumerate(aabb):
            self.register_buffer(f'aabb{i}', p)
        self.extra_args = kwargs
        self.is_ndc = is_ndc

        self.transfer_learning = self.extra_args["transfer_learning"]
        self.scene_grids = nn.ModuleList()
        self.features = None
        feature_dim = None
        for si in range(num_scenes):
            grids = nn.ModuleList()
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config:
                    if si == 0:  # Only make one set of features
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
                    out_dim: int = grid_config["output_coordinate_dim"]
                    grid_nd: int = grid_config["grid_dimensions"]
                    reso: List[int] = grid_config["resolution"]
                    try:
                        in_dim = len(reso)
                    except AttributeError:
                        raise ValueError("Configuration incorrect: resolution must be a list.")
                    num_comp = math.comb(in_dim, grid_nd)
                    rank: Sequence[int] = to_list(grid_config["rank"], num_comp, "rank")
                    # Configuration correctness checks
                    assert in_dim == grid_config["input_coordinate_dim"]
                    if li == 0:
                        assert in_dim == 3
                    assert out_dim in {1, 2, 3, 4, 5, 6, 7}
                    assert grid_nd <= in_dim
                    if grid_nd == in_dim:
                        assert all(r == 1 for r in rank)
                    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
                    grid_coefs = nn.ParameterList()
                    for ci, coo_comb in enumerate(coo_combs):
                        grid_coefs.append(
                            torch.nn.Parameter(nn.init.normal_(torch.empty(
                                [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb]
                            ), mean=0.0, std=grid_config["init_std"])))
                    grids.append(grid_coefs)
            self.scene_grids.append(grids)
        self.decoder = NNDecoder(feature_dim=feature_dim, sigma_net_width=64, sigma_net_layers=1)
        self.raymarcher = RayMarcher(**self.extra_args)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes.")

    def aabb(self, i: int) -> torch.Tensor:
        return getattr(self, f'aabb{i}')

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

    def compute_features(self, pts: torch.Tensor, grid_id: int) -> torch.Tensor:
        grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grids_info = self.config

        interp = pts
        grid: nn.ParameterList
        for level_info, grid in zip(grids_info, grids):
            if "feature_dim" in level_info:
                continue
            coo_plane = self.get_coo_plane(
                interp,
                level_info.get("grid_dimensions", level_info["input_coordinate_dim"])
            )
            coo_combs = list(itertools.combinations(range(interp.shape[-1]), level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))
            interp_out = None
            for ci, coo_comb in enumerate(coo_combs):
                if interp_out is None:
                    interp_out = (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"]))
                else:
                    interp_out = interp_out * (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"]))
            interp = interp_out.sum(dim=-1)
        return grid_sample_wrapper(self.features, interp)

    @torch.autograd.no_grad()
    def normalize_coord(self, pts: torch.Tensor, grid_id: int) -> torch.Tensor:
        """
        break-down of the normalization steps. pts starts from [a0, a1]
        1. pts - a0 => [0, a1-a0]
        2. / (a1 - a0) => [0, 1]
        3. * 2 => [0, 2]
        4. - 1 => [-1, 1]
        """
        return (pts - self.aabb(grid_id)[0]) * (2.0 / (self.aabb(grid_id)[1] - self.aabb(grid_id)[0])) - 1

    def forward(self, rays_o, rays_d, bg_color, grid_id=0):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        intersection_pts, ridx, boundary, deltas = self.raymarcher.get_intersections(
            rays_o, rays_d, self.aabb(grid_id), perturb=self.training, is_ndc=self.is_ndc)
        n_rays = rays_o.shape[0]
        dev = rays_o.device

        # mask has shape [batch, n_intrs]
        # intersection_pts has shape [n_valid_intrs, 3]

        # Normalization (between [-1, 1])
        intersection_pts = self.normalize_coord(intersection_pts, grid_id)
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
        if self.transfer_learning:
            params = [
                {"params": self.scene_grids.parameters(), "lr": lr},
            ]
        else:
            params = [
                {"params": self.scene_grids.parameters(), "lr": lr},
                {"params": self.decoder.parameters(), "lr": lr},
                {"params": self.features, "lr": lr},
            ]
        return params


def to_list(el, list_len, name: Optional[str] = None) -> Sequence:
    if not isinstance(el, collections.abc.Sequence):
        return [el] * list_len
    if len(el) != list_len:
        raise ValueError(f"Length of {name} is incorrect. Expected {list_len} but found {len(el)}")
    return el
