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
        pts = pts.to(dtype=self.density_volume.dtype)
        density_vals = grid_sample_wrapper(self.density_volume[None, ...], pts, align_corners=True).view(-1)
        return density_vals

    @property
    def grid_size(self) -> torch.Tensor:
        return torch.tensor((self.density_volume.shape[-1], self.density_volume.shape[-2], self.density_volume.shape[-3]), dtype=torch.long)


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
        self.alpha_mask_threshold = self.extra_args["density_threshold"]
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
                    grid_config["rank"] = rank
                    # Configuration correctness checks
                    assert in_dim == grid_config["input_coordinate_dim"]
                    if li == 0:
                        assert in_dim == 3
                        self.set_resolution(torch.tensor(reso, dtype=torch.long), grid_id=si)
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

    def set_resolution(self, resolution: Union[torch.Tensor, List[torch.Tensor]], grid_id: Optional[int] = None):
        if grid_id is None:
            # resolution needs to be BufferList (but BufferList doesn't exist so we emulate it)
            for i, p in enumerate(resolution):
                if hasattr(self, f'resolution{i}'):
                    setattr(self, f'resolution{i}', p)
                else:
                    self.register_buffer(f'resolution{i}', p)
        else:
            assert isinstance(resolution, torch.Tensor)
            if hasattr(self, f'resolution{grid_id}'):
                setattr(self, f'resolution{grid_id}', resolution)
            else:
                self.register_buffer(f'resolution{grid_id}', resolution)

    def resolution(self, i: int) -> torch.Tensor:
        return getattr(self, f'resolution{i}')

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
                            -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
                else:
                    interp_out = interp_out * (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
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

    @torch.no_grad()
    def update_alpha_mask(self, grid_id: int = 0):
        assert len(self.config) == 2, "Alpha-masking not supported for multiple layers of indirection."
        aabb = self.aabb(grid_id)
        grid_size = self.resolution(grid_id)
        grid_size_l = self.config[0]["resolution"]
        dev = aabb.device

        step_size = self.raymarcher.get_step_size(aabb, grid_size)

        # Generate points in regularly spaced grid (already normalized)
        pts = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size_l[0]),
            torch.linspace(-1, 1, grid_size_l[1]),
            torch.linspace(-1, 1, grid_size_l[2]), indexing='ij'
        ), dim=-1).to(dev)  # [gs0, gs1, gs2, 3]
        pts = pts.view(-1, 3)  # [gs0*gs1*gs2, 3]

        # Compute density on the grid at the regularly spaced points
        if self.density_mask[grid_id] is not None:
            alpha_mask = self.density_mask[grid_id].sample_density(pts) > 0.0
        else:
            alpha_mask = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)

        if alpha_mask.any():
            features = self.compute_features(pts[alpha_mask], grid_id)
            density_masked = trunc_exp(
                self.decoder.compute_density(features, rays_d=None, precompute_color=False))
            density = torch.zeros(pts.shape[0], dtype=density_masked.dtype, device=density_masked.device)
            density[alpha_mask] = density_masked.view(-1)
        else:
            density = torch.zeros(pts.shape[0], dtype=torch.float32, device=dev)

        alpha = 1.0 - torch.exp(-density * step_size).view(grid_size_l)  # [gs0, gs1, gs2]
        pts = pts.view(grid_size_l + [3])  # [gs0, gs1, gs2, 3]

        # Compute the mask (max-pooling) and the new aabb
        pts = pts.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()

        alpha = F.max_pool3d(alpha[None, None, ...], kernel_size=3, padding=1, stride=1)
        alpha = alpha.view(grid_size_l[::-1])
        alpha = F.threshold(alpha, self.alpha_mask_threshold, 0.0)  # set to 0 if <= threshold

        alpha_mask = alpha > 0
        valid_pts = pts[alpha_mask]
        pts_min = valid_pts.amin(0)
        pts_max = valid_pts.amax(0)
        # pts_min, pts_max are normalized between -1, 1 so we need to denormalize them
        # +1 => [0, 2]; / 2 => [0, 1]; * (a1 - a0) => [0, a1-a0]; + a0 => [a0, a1]
        pts_min = ((pts_min + 1) / 2 * (aabb[1] - aabb[0])) + aabb[0]
        pts_max = ((pts_max + 1) / 2 * (aabb[1] - aabb[0])) + aabb[0]

        new_aabb = torch.stack((pts_min, pts_max), 0)
        log.info(f"Updated alpha mask for grid {grid_id}. "
                 f"Bounding box from {aabb} to {new_aabb}. "
                 f"Remaining {alpha_mask.sum() / grid_size_l[0] / grid_size_l[1] / grid_size_l[2] * 100:.2f}% voxels.")
        self.density_mask[grid_id] = DensityMask(alpha)
        #self.set_aabb(new_aabb, grid_id=grid_id)
        return new_aabb

    @torch.no_grad()
    def shrink(self, new_aabb, grid_id: int):
        log.info(f"Shrinking grid {grid_id}...")

        cur_aabb = self.aabb(grid_id)
        cur_grid_size = self.resolution(grid_id)
        dev = cur_aabb.device

        cur_units = (cur_aabb[1] - cur_aabb[0]) / (cur_grid_size - 1)
        t_l, b_r = (new_aabb[0] - cur_aabb[0]) / cur_units, (new_aabb[1] - cur_aabb[0]) / cur_units
        t_l = torch.round(t_l).long()
        b_r = torch.round(b_r).long() + 1
        b_r = torch.minimum(b_r, cur_grid_size)  # don't exceed current grid dimensions

        # Truncate the parameter grid to the new grid-size
        # IMPORTANT: This will only work if input-dim is 3!
        grid_info = self.config[0]
        coo_combs = list(itertools.combinations(
            range(grid_info["input_coordinate_dim"]),
            grid_info.get("grid_dimensions", grid_info["input_coordinate_dim"])))
        for ci, coo_comb in enumerate(coo_combs):
            slices = [slice(None), slice(None)] + [slice(t_l[cc], b_r[cc]) for cc in coo_comb]
            self.scene_grids[grid_id][0][ci] = torch.nn.Parameter(
                self.scene_grids[grid_id][0][ci].data[slices]
            )

        # TODO: Why the correction? Check if this ever occurs
        if not torch.all(self.density_mask[grid_id].grid_size.to(device=dev) == cur_grid_size):
            t_l_r, b_r_r = t_l / (cur_grid_size - 1), (b_r - 1) / (cur_grid_size - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * cur_aabb[0] + t_l_r * cur_aabb[1]
            correct_aabb[1] = (1 - b_r_r) * cur_aabb[0] + b_r_r * cur_aabb[1]
            log.info(f"Corrected new AABB from {new_aabb} to {correct_aabb}")
            new_aabb = correct_aabb

        new_size = b_r - t_l
        self.set_aabb(new_aabb, grid_id)
        self.set_resolution(new_size, grid_id)

    def forward(self, rays_o, rays_d, bg_color, grid_id=0, channels: Sequence[str] = ("rgb", "depth")):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        rm_out = self.raymarcher.get_intersections2(
            rays_o, rays_d, self.aabb(grid_id), self.resolution(grid_id), perturb=self.training,
            is_ndc=self.is_ndc)
        rays_d = rm_out["rays_d"]
        deltas = rm_out["deltas"]
        intersection_pts = rm_out["intersections"]
        ridx = rm_out["ridx"]
        boundary = rm_out["boundary"]
        z_vals = rm_out["z_vals"]
        n_rays = rays_o.shape[0]
        dev = rays_o.device

        # mask has shape [batch, n_intrs]
        # intersection_pts has shape [n_valid_intrs, 3]

        # Normalization (between [-1, 1])
        intersection_pts = self.normalize_coord(intersection_pts, grid_id)

        # Filter intersections which have a low density according to the density mask
        if self.density_mask[grid_id] is not None:
            alpha_mask = self.density_mask[grid_id].sample_density(intersection_pts) > 0
            intersection_pts = intersection_pts[alpha_mask]
            deltas = deltas[alpha_mask]
            ridx = ridx[alpha_mask]
            z_vals = z_vals[alpha_mask]
            boundary = spc_render.mark_pack_boundaries(ridx)

        # rays_d in the packed format (essentially repeated a number of times)
        rays_d_rep = rays_d.index_select(0, ridx)

        # compute features and render
        features = self.compute_features(intersection_pts, grid_id)
        density_masked = trunc_exp(self.decoder.compute_density(features, rays_d=rays_d_rep))
        rgb_masked = torch.sigmoid(self.decoder.compute_color(features, rays_d=rays_d_rep))

        # Compute optical thickness
        tau = density_masked.reshape(-1, 1) * deltas.reshape(-1, 1) * 25
        # ridx_hit are the ray-IDs at the first intersection (the boundary).
        ridx_hit = ridx[boundary]
        # Perform volumetric integration
        ray_colors, transmittance = spc_render.exponential_integration(
            rgb_masked.reshape(-1, 3), tau, boundary, exclusive=True)

        outputs = []
        if "rgb" in channels:
            alpha = spc_render.sum_reduce(transmittance, boundary)
            # Blend output color with background
            if isinstance(bg_color, torch.Tensor) and bg_color.shape == (n_rays, 3):
                rgb = bg_color
                color = ray_colors + (1.0 - alpha) * bg_color[ridx_hit.long(), :]
            else:
                rgb = torch.full((n_rays, 3), bg_color, dtype=ray_colors.dtype, device=dev)
                color = ray_colors + (1.0 - alpha) * bg_color
            rgb[ridx_hit.long(), :] = color
            outputs.append(rgb)

        if "depth" in channels:
            # Compute depth
            depth_map = spc_render.sum_reduce(z_vals.view(-1, 1) * transmittance, boundary)
            depth = torch.zeros(n_rays, 1, device=depth_map.device)
            depth[ridx_hit.long(), :] = depth_map
            outputs.append(depth)

        return outputs

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
