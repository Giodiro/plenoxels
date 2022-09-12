import collections.abc
import itertools
import math
from typing import Dict, List, Union, Optional, Sequence
import logging as log

import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin.render.spc as spc_render

from plenoxels.models.utils import grid_sample_wrapper
from .decoders import NNDecoder, SHDecoder
from ..ops.activations import trunc_exp
from ..raymarching.raymarching import RayMarcher


class DensityMask(nn.Module):
    def __init__(self, density_volume: torch.Tensor):
        super().__init__()
        self.register_buffer('density_volume', density_volume)

    def sample_density(self, pts: torch.Tensor) -> torch.Tensor:
        density_vals = grid_sample_wrapper(self.density_volume, pts, align_corners=True).view(-1)
        return density_vals


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
        self.set_aabb(aabb)
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
        self.density_mask = nn.ModuleList([None] * num_scenes)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes.")

    def set_aabb(self, aabb: Union[torch.Tensor, List[torch.Tensor]], grid_id: Optional[int] = None):
        if grid_id is None:
            # aabb needs to be BufferList (but BufferList doesn't exist so we emulate it)
            for i, p in enumerate(aabb):
                if hasattr(self, f'aabb{i}'):
                    setattr(self, f'aabb{i}', p)
                else:
                    self.register_buffer(f'aabb{i}', p)
        else:
            assert isinstance(aabb, torch.Tensor)
            if hasattr(self, f'aabb{grid_id}'):
                setattr(self, f'aabb{grid_id}', aabb)
            else:
                self.register_buffer(f'aabb{grid_id}', aabb)

    def aabb(self, i: int) -> torch.Tensor:
        return getattr(self, f'aabb{i}')

    def compute_features(self, pts: torch.Tensor, grid_id: int) -> torch.Tensor:
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
        aabb = self.aabb(grid_id)
        return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1

    def update_alpha_mask(self, step_size: float, grid_id: int = 0):
        assert len(self.config) == 2, "Alpha-masking not supported for multiple layers of indirection."
        grid_size = self.config[0]["resolution"]

        # Generate points in regularly spaced grid (already normalized)
        pts = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size[0]),
            torch.linspace(-1, 1, grid_size[1]),
            torch.linspace(-1, 1, grid_size[2]), indexing='ij'
        ), dim=-1).to(self.device)  # [gs0, gs1, gs2, 3]  TODO: self.device doesn't exist
        pts = pts.view(-1, 3)  # [gs0*gs1*gs2, 3]

        # Compute density on the grid at the regularly spaced points
        if self.density_mask[grid_id] is not None:
            alpha_mask = self.density_mask[grid_id].sample_alpha(pts) > 0.0
        else:
            alpha_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)

        density = torch.zeros(pts.shape[0], dtype=torch.float32, device=pts.device)
        if alpha_mask.any():
            features = self.compute_features(pts[alpha_mask], grid_id)
            density_masked = trunc_exp(
                self.decoder.compute_density(features, rays_d=None, precompute_color=False))
            density[alpha_mask] = density_masked

        alpha = 1.0 - torch.exp(-density * step_size)
        alpha = alpha.view(grid_size)  # [gs0, gs1, gs2]
        pts = pts.view(grid_size[0], grid_size[1], grid_size[2], 3)

        # Compute the mask (max-pooling) and the new aabb
        pts = pts.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()

        alpha = F.max_pool3d(alpha[None, None, ...], kernel_size=3, padding=1, stride=1).view(grid_size[::-1])
        alpha = F.threshold(alpha, self.alpha_mask_threshold, 0.0)  # set to 0 if <= threshold

        valid_pts = pts[alpha > 0]
        pts_min = valid_pts.amin(0)
        pts_max = valid_pts.amax(0)

        new_aabb = torch.stack((pts_min, pts_max), 0)
        self.density_mask[grid_id] = DensityMask(alpha)
        self.set_aabb(new_aabb, grid_id=grid_id)
        return alpha

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

        # Filter intersections which have a low density according to the density mask
        if self.density_mask[grid_id] is not None:
            alpha_mask = self.density_mask[grid_id].sample_density(intersection_pts) > 0
            intersection_pts = intersection_pts[alpha_mask]
            deltas = deltas[alpha_mask]
            ridx = ridx[alpha_mask]
            boundary = spc_render.mark_pack_boundaries(ridx)

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
