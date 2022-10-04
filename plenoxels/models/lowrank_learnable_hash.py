import collections.abc
import itertools
import math
from typing import Dict, List, Union, Optional, Sequence, Tuple
import logging as log

import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin.render.spc as spc_render

from plenoxels.models.utils import grid_sample_wrapper, compute_plane_tv
from .decoders import NNDecoder, SHDecoder
from ..ops.activations import trunc_exp
from ..raymarching.raymarching import RayMarcher


class DensityMask(nn.Module):
    def __init__(self, density_volume: torch.Tensor, aabb: torch.Tensor):
        super().__init__()
        self.register_buffer('density_volume', density_volume)
        self.register_buffer('aabb', aabb)

    def sample_density(self, pts: torch.Tensor) -> torch.Tensor:
        # Normalize pts
        pts = (pts - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1
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
                 sh: bool = False,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.set_aabb(aabb)
        self.extra_args = kwargs
        self.is_ndc = is_ndc
        self.sh = sh
        self.density_multiplier = self.extra_args.get("density_multiplier")
        self.transfer_learning = self.extra_args["transfer_learning"]
        self.alpha_mask_threshold = self.extra_args["density_threshold"]

        self.density_act = F.relu#trunc_exp

        self.scene_grids = nn.ModuleList()
        self.features = None
        self.feature_dim = None
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
                        self.features = nn.Parameter(
                            torch.empty([grid_config["feature_dim"]] + reso[::-1]))
                        if self.sh:
                            nn.init.zeros_(self.features)
                            self.features[-1].data.fill_(grid_config["init_std"])  # here init_std is repurposed as the sigma initialization
                        else:
                            nn.init.normal_(self.features, mean=0.0, std=grid_config["init_std"])
                        self.feature_dim = grid_config["feature_dim"]
                else:
                    out_dim: int = grid_config["output_coordinate_dim"]
                    grid_nd: int = grid_config["grid_dimensions"]
                    self.reso: List[int] = grid_config["resolution"]
                    try:
                        in_dim = len(self.reso)
                    except AttributeError:
                        raise ValueError("Configuration incorrect: resolution must be a list.")
                    num_comp = math.comb(in_dim, grid_nd)
                    rank: Sequence[int] = to_list(grid_config["rank"], num_comp, "rank")
                    grid_config["rank"] = rank
                    # Configuration correctness checks
                    assert in_dim == grid_config["input_coordinate_dim"]
                    if li == 0:
                        assert in_dim == 3
                        self.set_resolution(torch.tensor(self.reso, dtype=torch.long), grid_id=si)
                    assert out_dim in {1, 2, 3, 4, 5, 6, 7}
                    assert grid_nd <= in_dim
                    if grid_nd == in_dim:
                        assert all(r == 1 for r in rank)
                    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
                    grid_coefs = nn.ParameterList()
                    for ci, coo_comb in enumerate(coo_combs):
                        grid_coefs.append(
                            torch.nn.Parameter(nn.init.normal_(torch.empty(
                                [1, out_dim * rank[ci]] + [self.reso[cc] for cc in coo_comb[::-1]]
                            ), mean=0.0, std=grid_config["init_std"])))
                    grids.append(grid_coefs)
            self.scene_grids.append(grids)
        if self.sh:
            self.decoder = SHDecoder(feature_dim=self.feature_dim)
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        self.raymarcher = RayMarcher(**self.extra_args)
        self.density_mask = nn.ModuleList([None] * num_scenes)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, decoder: {self.decoder}.")

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

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param pts:
            Coordinates normalized between -1, 1
        :param grid_id:
        :param return_coords:
        :return:
        """
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
        out = grid_sample_wrapper(self.features.to(dtype=interp.dtype), interp).view(-1, self.feature_dim)
        if return_coords:
            return out, interp
        return out

    @torch.autograd.no_grad()
    def normalize_coords(self, pts: torch.Tensor, grid_id: int) -> torch.Tensor:
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
        grid_size_l = grid_size.cpu().tolist()
        step_size = self.raymarcher.get_step_size(aabb, grid_size)

        # Compute density on a regular grid (of shape grid_size)
        pts = self.get_points_on_grid(aabb, grid_size, max_voxels=None)
        density = self.compute_density(pts, grid_id, use_mask=True)

        alpha = 1.0 - torch.exp(-density * step_size * self.density_multiplier).view(grid_size_l)  # [gs0, gs1, gs2]
        pts = pts.view(grid_size_l + [3])  # [gs0, gs1, gs2, 3]

        # Transpose to get correct Depth, Height, Width format
        pts = pts.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()

        # Compute the mask (max-pooling) and the new aabb.
        alpha = F.max_pool3d(alpha[None, None, ...], kernel_size=3, padding=1, stride=1)
        alpha = alpha.view(grid_size_l[::-1])
        alpha = F.threshold(alpha, self.alpha_mask_threshold, 0.0)  # set to 0 if <= threshold

        alpha_mask = alpha > 0
        valid_pts = pts[alpha_mask]
        pts_min = valid_pts.amin(0)
        pts_max = valid_pts.amax(0)

        new_aabb = torch.stack((pts_min, pts_max), 0)
        log.info(f"Updated alpha mask for scene {grid_id}. "
                 f"New bounding box={new_aabb.view(-1)}. "
                 f"Remaining {alpha_mask.sum() / grid_size_l[0] / grid_size_l[1] / grid_size_l[2] * 100:.2f}% voxels.")
        self.density_mask[grid_id] = DensityMask(alpha, aabb=aabb)
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
            slices = [slice(None), slice(None)] + [slice(t_l[cc].item(), b_r[cc].item()) for cc in coo_comb[::-1]]
            self.scene_grids[grid_id][0][ci] = torch.nn.Parameter(
                self.scene_grids[grid_id][0][ci].data[slices]
            )

        # TODO: Why the correction? Check if this ever occurs
        if not torch.all(self.density_mask[grid_id].grid_size.to(device=dev) == cur_grid_size):
            t_l_r, b_r_r = t_l / (cur_grid_size - 1), (b_r - 1) / (cur_grid_size - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * cur_aabb[0] + t_l_r * cur_aabb[1]
            correct_aabb[1] = (1 - b_r_r) * cur_aabb[0] + b_r_r * cur_aabb[1]
            log.info(f"Corrected new AABB from {new_aabb.view(-1)} to {correct_aabb.view(-1)}")
            new_aabb = correct_aabb

        new_size = b_r - t_l
        self.set_aabb(new_aabb, grid_id)
        self.set_resolution(new_size, grid_id)
        log.info(f"Shrunk scene {grid_id}. New AABB={new_aabb.view(-1)} New resolution={new_size.view(-1)}")

    @torch.no_grad()
    def upsample(self, new_reso, grid_id: int):
        grid_info = self.config[0]
        coo_combs = list(itertools.combinations(
            range(grid_info["input_coordinate_dim"]),
            grid_info.get("grid_dimensions", grid_info["input_coordinate_dim"])))
        for ci, coo_comb in enumerate(coo_combs):
            new_size = [new_reso[cc] for cc in coo_comb]
            if len(coo_comb) == 3:
                mode = 'trilinear'
            elif len(coo_comb) == 2:
                mode = 'bilinear'
            elif len(coo_comb) == 1:
                mode = 'linear'
            else:
                raise RuntimeError()
            grid_data = self.scene_grids[grid_id][0][ci].data
            self.scene_grids[grid_id][0][ci] = torch.nn.Parameter(
                F.interpolate(grid_data, size=new_size[::-1], mode=mode, align_corners=True))
        self.set_resolution(
            torch.tensor(new_reso, dtype=torch.long, device=grid_data.device), grid_id)
        log.info(f"Upsampled scene {grid_id} to resolution={new_reso}")

    def get_points_on_grid(self, aabb, grid_size, max_voxels: Optional[int] = None):
        """
        Returns points from a regularly spaced grids of size grid_size.
        Coordinates normalized between [aabb0, aabb1]

        :param aabb:
        :param grid_size:
        :param max_voxels:
        :return:
        """
        dev = self.features.device
        pts = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, grid_size[0], device=dev),
            torch.linspace(0, 1, grid_size[1], device=dev),
            torch.linspace(0, 1, grid_size[2], device=dev), indexing='ij'
        ), dim=-1)  # [gs0, gs1, gs2, 3]
        pts = pts.view(-1, 3)  # [gs0*gs1*gs2, 3]
        if max_voxels is not None:
            # with replacement as it's faster?
            pts = pts[torch.randint(pts.shape[0], (max_voxels, )), :]
        # Normalize between [aabb0, aabb1]
        pts = aabb[0] * (1 - pts) + aabb[1] * pts
        return pts

    def compute_density(self, pts: torch.Tensor, grid_id: int, use_mask: bool):
        dev = pts.device
        alpha_mask = None
        if self.density_mask[grid_id] is not None and use_mask:
            alpha_mask = self.density_mask[grid_id].sample_density(pts).gt(0.0)

        if alpha_mask is not None and alpha_mask.any():
            pts_sampled = self.normalize_coords(pts[alpha_mask], grid_id)
            features = self.compute_features(pts_sampled, grid_id)
            density_masked = self.density_act(
                self.decoder.compute_density(features, rays_d=None, precompute_color=False))
            density = torch.zeros(pts.shape[0], dtype=density_masked.dtype, device=dev)
            density[alpha_mask] = density_masked.view(-1)
        elif alpha_mask is None:
            features = self.compute_features(self.normalize_coords(pts, grid_id), grid_id)
            density = self.density_act(
                self.decoder.compute_density(features, rays_d=None, precompute_color=False)).view(-1)
        else:
            density = torch.zeros(pts.shape[0], dtype=torch.float32, device=dev)
        return density

    def forward(self, rays_o, rays_d, bg_color, grid_id=0, channels: Sequence[str] = ("rgb", "depth")):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        rm_out = self.raymarcher.get_intersections2(
            rays_o, rays_d, self.aabb(grid_id), self.resolution(grid_id), perturb=self.training,
            is_ndc=self.is_ndc)
        rays_d = rm_out["rays_d"]                   # [n_rays, 3]
        intersection_pts = rm_out["intersections"]  # [n_rays, n_intrs, 3]
        mask = rm_out["mask"]                       # [n_rays, n_intrs]
        n_rays, n_intrs = intersection_pts.shape[:2]
        dev = rays_o.device

        version = 2

        # Filter intersections which have a low density according to the density mask
        if self.density_mask[grid_id] is not None:
            # density_mask needs unnormalized coordinates: normalization happens internally
            # and can be with a different aabb than the current one.
            alpha_mask = self.density_mask[grid_id].sample_density(intersection_pts[mask]) > 0
            invalid_mask = ~mask
            invalid_mask[mask] |= (~alpha_mask)
            mask = ~invalid_mask

        # Normalization (between [-1, 1])
        intersection_pts = self.normalize_coords(intersection_pts, grid_id)

        if len(intersection_pts) == 0:
            outputs = {}
            if "rgb" in channels:
                if bg_color is None:
                    outputs["rgb"] = torch.zeros((n_rays, 3), dtype=rays_o.dtype, device=dev)
                elif isinstance(bg_color, torch.Tensor) and bg_color.shape == (n_rays, 3):
                    outputs["rgb"] = bg_color
                else:
                    outputs["rgb"] = torch.full((n_rays, 3), bg_color, dtype=rays_o.dtype, device=dev)
            if "depth" in channels:
                outputs["depth"] = torch.zeros(n_rays, 1, device=dev, dtype=rays_o.dtype)
            if "alpha" in channels:
                outputs["alpha"] = torch.zeros(n_rays, 1, device=dev, dtype=rays_o.dtype)
            return outputs

        # compute features and render
        features = self.compute_features(intersection_pts[mask], grid_id)

        outputs = {}
        if version == 2:
            z_vals = rm_out["z_vals"]  # [batch_size, n_samples]
            deltas = rm_out["deltas"]  # [batch_size, n_samples]
            rays_d_rep = rays_d.view(-1, 1, 3).expand(intersection_pts.shape)
            masked_rays_d_rep = rays_d_rep[mask]
            density_masked = self.density_act(self.decoder.compute_density(features, rays_d=masked_rays_d_rep))
            density = torch.zeros(n_rays, n_intrs, device=intersection_pts.device, dtype=density_masked.dtype)
            density[mask] = density_masked.view(-1)

            alpha, weight, transmission = raw2alpha(density, deltas * self.density_multiplier)  # Each is shape [batch_size, n_samples]

            rgb_masked = self.decoder.compute_color(features, rays_d=masked_rays_d_rep)
            rgb = torch.zeros(n_rays, n_intrs, 3, device=intersection_pts.device, dtype=rgb_masked.dtype)
            rgb[mask] = rgb_masked
            rgb = torch.sigmoid(rgb)

            # Confirmed that torch.sum(weight, -1) matches 1-transmission[:,-1]
            acc_map = 1 - transmission[:,-1]

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
        elif version == 1:  # This uses kaolin. The depth could be wrong in mysterious ways.
            ridx = rm_out["ridx"][mask]
            deltas = rm_out["deltas"][mask]
            boundary = spc_render.mark_pack_boundaries(ridx)

            # rays_d in the packed format (essentially repeated a number of times)
            rays_d_rep = rays_d.index_select(0, ridx)

            density_masked = self.density_act(self.decoder.compute_density(features, rays_d=rays_d_rep))
            rgb_masked = torch.sigmoid(self.decoder.compute_color(features, rays_d=rays_d_rep))

            # Compute optical thickness
            tau = density_masked.reshape(-1, 1) * deltas.reshape(-1, 1) * self.density_multiplier
            # ridx_hit are the ray-IDs at the first intersection (the boundary).
            ridx_hit = ridx[boundary]
            # Perform volumetric integration
            ray_colors, transmittance = spc_render.exponential_integration(
                rgb_masked.reshape(-1, 3), tau, boundary, exclusive=True)

            alpha = spc_render.sum_reduce(transmittance, boundary)

            if "rgb" in channels:
                # Blend output color with background
                if bg_color is None:
                    rgb = torch.zeros((n_rays, 3), dtype=ray_colors.dtype, device=dev)
                    color = ray_colors
                elif isinstance(bg_color, torch.Tensor) and bg_color.shape == (n_rays, 3):
                    rgb = bg_color
                    color = ray_colors + (1.0 - alpha) * bg_color[ridx_hit.long(), :]
                else:
                    rgb = torch.full((n_rays, 3), bg_color, dtype=ray_colors.dtype, device=dev)
                    color = ray_colors + (1.0 - alpha) * bg_color
                rgb[ridx_hit.long(), :] = color
                outputs["rgb"] = rgb
            if "depth" in channels:
                z_mids = rm_out["z_mids"][mask]
                # Compute depth
                depth_map = spc_render.sum_reduce(z_mids.view(-1, 1) * transmittance, boundary) / torch.clip(alpha, 1e-5)
                depth = torch.zeros(n_rays, 1, device=dev, dtype=depth_map.dtype)
                depth[ridx_hit.long(), :] = depth_map
                outputs["depth"] = depth
            if "alpha" in channels:
                alpha_out = torch.zeros(n_rays, 1, device=alpha.device, dtype=alpha.dtype)
                alpha_out[ridx_hit.long(), :] = alpha
                outputs["alpha"] = alpha_out

        return outputs

    def compute_plane_tv(self, grid_id, what='Gcoords'):
        grids: nn.ModuleList = self.scene_grids[grid_id]
        total = 0
        for grid_ls in grids:
            for grid in grid_ls:
                if what == 'Gcoords':
                    total += compute_plane_tv(grid)
                elif what == 'features':
                    # Look up the features so we do tv on features rather than coordinates
                    coords = grid.view(-1, len(self.features.shape)-1)
                    features = grid_sample_wrapper(self.features, coords).reshape(-1, self.feature_dim, grid.shape[-2], grid.shape[-1])
                    total += compute_plane_tv(features)
        return total

    def compute_l1density(self, max_voxels, grid_id):
        pts = self.get_points_on_grid(
            self.aabb(grid_id), self.resolution(grid_id), max_voxels=max_voxels)
        # Compute density on the grid
        density = self.compute_density(pts, grid_id, use_mask=True)
        return torch.mean(torch.abs(density))

    def compute_3d_tv(self, grid_id, what='Gcoords', batch_size=100, patch_size=3):
        aabb = self.aabb(grid_id)
        grid_size_l = self.resolution(grid_id)
        dev = aabb.device
        pts = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, patch_size, device=dev),
            torch.linspace(0, 1, patch_size, device=dev),
            torch.linspace(0, 1, patch_size, device=dev), indexing='ij'
        ), dim=-1)  # [gs0, gs1, gs2, 3]
        pts = pts.view(-1, 3)

        start = torch.rand(batch_size, 3, device=dev) * (1 - patch_size / grid_size_l[None, :])
        end = start + (patch_size / grid_size_l[None, :])

        # pts: [1, gs0, gs1, gs2, 3] * (bs, 1, 1, 1, 3) + (bs, 1, 1, 1, 3)
        pts = pts[None, ...] * (end - start)[:, None, None, None, :] + start[:, None, None, None, :]
        pts = pts.view(-1, 3)  # [bs*gs0*gs1*gs2, 3]

        # Normalize between [aabb0, aabb1]
        pts = aabb[0] * (1 - pts) + aabb[1] * pts

        if what == 'density':
            # Compute density on the grid
            density = self.compute_density(pts, grid_id, use_mask=False)
            patches = density.view(-1, patch_size, patch_size, patch_size)
        elif what == 'Gcoords':
            pts = self.normalize_coords(pts, grid_id)
            _, coords = self.compute_features(pts, grid_id, return_coords=True)
            patches = coords.view(-1, patch_size, patch_size, patch_size, coords.shape[-1])
        else:
            raise ValueError(what)

        d0 = patches[:, 1:, :, :, :] - patches[:, :-1, :, :, :]
        d1 = patches[:, :, 1:, :, :] - patches[:, :, :-1, :, :]
        d2 = patches[:, :, :, 1:, :] - patches[:, :, :, :-1, :]

        return (d0.square().mean() + d1.square().mean() + d2.square().mean())  # l2
        # return (d0.abs().mean() + d1.abs().mean() + d2.abs().mean())  # l1

    def get_params(self, lr):
        params = [
            {"params": self.scene_grids.parameters(), "lr": lr},
        ]
        if not self.transfer_learning:
            params += [
                {"params": self.decoder.parameters(), "lr": lr},
                {"params": self.features, "lr": lr},
            ]
        return params


def raw2alpha(sigma, dist):
    alpha = 1 - torch.exp(-sigma * dist)
    T = torch.cat((torch.ones(alpha.shape[0], 1, device=alpha.device),
                   torch.cumprod(1.0 - alpha, dim=-1)), dim=-1)
    # T = torch.cat((torch.ones(alpha.shape[0], 1, device=alpha.device),
    #                torch.cumprod(1.0 - alpha[:, :-1] + 1e-10, dim=-1)), dim=-1)
    #T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, device=alpha.device), 1 - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]
    return alpha, weights, T#[:, -1:]  # Return full-length T so we can use the last one for background


def to_list(el, list_len, name: Optional[str] = None) -> Sequence:
    if not isinstance(el, collections.abc.Sequence):
        return [el] * list_len
    if len(el) != list_len:
        raise ValueError(f"Length of {name} is incorrect. Expected {list_len} but found {len(el)}")
    return el
