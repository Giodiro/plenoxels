import itertools
import logging as log
from typing import Dict, List, Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.models.utils import (
    grid_sample_wrapper, raw2alpha, init_grid_param,
    init_features_param
)
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel
from ..ops.bbox_colliders import intersect_with_aabb
from ..raymarching.ray_samplers import (
    RayBundle
)


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


class LowrankLearnableHash(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],
                 is_ndc: bool,
                 is_contracted: bool,
                 sh: bool,
                 use_F: bool,
                 density_activation: str,
                 proposal_sampling: bool,
                 n_intersections: int,
                 single_jitter: bool,
                 raymarch_type: str,
                 num_scenes: int = 1,
                 multiscale_res: List[int] = [1],
                 global_translation=None,
                 global_scale=None,
                 **kwargs):
        super().__init__(grid_config=grid_config,
                         is_ndc=is_ndc,
                         is_contracted=is_contracted,
                         sh=sh,
                         use_F=use_F,
                         num_scenes=num_scenes,
                         global_scale=global_scale,
                         global_translation=global_translation,
                         density_activation=density_activation,
                         use_proposal_sampling=proposal_sampling,
                         density_field_resolution=kwargs.get('density_field_resolution', None),
                         density_field_rank=kwargs.get('density_field_rank', None),
                         num_proposal_samples=kwargs.get('num_proposal_samples', None),
                         n_intersections=n_intersections,
                         single_jitter=single_jitter,
                         raymarch_type=raymarch_type,
                         spacing_fn=kwargs.get('spacing_fn', None),
                         num_samples_multiplier=kwargs.get('num_samples_multiplier', None),
                         aabb=aabb)
        self.extra_args = kwargs
        self.density_multiplier = self.extra_args.get("density_multiplier")
        self.transfer_learning = self.extra_args["transfer_learning"]
        self.alpha_mask_threshold = self.extra_args["density_threshold"]

        self.multiscale_res = multiscale_res
        self.active_scales = [self.multiscale_res[0]]
        self.scene_grids = nn.ModuleList()
        for si in range(num_scenes):
            multi_scale_grids = nn.ModuleList()
            for res in self.multiscale_res:
                grids = nn.ModuleList()
                for li, grid_config in enumerate(self.config):
                    # initialize feature grid
                    if "feature_dim" in grid_config and si == 0:  # Only make one set of features
                        if self.use_F:
                            self.features = init_features_param(grid_config, self.sh)
                            self.feature_dim = self.features.shape[0]
                    # initialize coordinate grid
                    else:
                        config = grid_config.copy()
                        config["resolution"] = [r * res for r in config["resolution"]]

                        gpdesc = init_grid_param(config, is_video=False, grid_level=li, use_F=self.use_F, is_appearance=False)
                        if li == 0:
                            self.set_resolution(gpdesc.reso, grid_id=si)
                        grids.append(gpdesc.grid_coefs)
                        for gc in gpdesc.grid_coefs:
                            log.info(f"Initialized grid with shape {gc.shape}")
                        if not self.use_F:
                            # shape[1] is out-dim * rank
                            self.feature_dim = gpdesc.grid_coefs[-1].shape[1] // config["rank"][0]
                multi_scale_grids.append(grids)
            self.scene_grids.append(multi_scale_grids)
        if self.sh:
            self.decoder = SHDecoder(
                feature_dim=self.feature_dim,
                decoder_type=self.extra_args.get('sh_decoder_type', 'manual'))
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        self.density_mask = nn.ModuleList([None] * num_scenes)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, "
                 f"decoder: {self.decoder}. Raymarcher: {self.raymarcher}")

    def step_cb(self, step, max_steps):
        if len(self.active_scales) != len(self.multiscale_res):
            if step == 1_000 or step == 2_000 or step == 3_000:
                new_scale = self.multiscale_res[len(self.active_scales)]
                self.active_scales.append(new_scale)
                log.info(f"Adding new scale to the set of active scales. "
                         f"New scales: {self.active_scales}")

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mulitres_grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grids_info = self.config

        multi_scale_interp = 0
        for scale_id, res in enumerate(self.active_scales):  # noqa
            grids: nn.ParameterList = mulitres_grids[scale_id]
            for level_info, grid in zip(grids_info, grids):
                if "feature_dim" in level_info:
                    continue

                # create plane combinations
                coo_combs = list(itertools.combinations(
                    range(pts.shape[-1]),
                    level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))

                interp_out = None
                for ci, coo_comb in enumerate(coo_combs):
                    # interpolate in plane
                    interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(
                                -1, level_info["output_coordinate_dim"], level_info["rank"])
                    # compute product
                    interp_out = interp_out_plane if interp_out is None else interp_out * interp_out_plane
            # average over rank
            interp = interp_out.mean(dim=-1)
            # sum over scales
            multi_scale_interp += interp

        if self.use_F:
            if multi_scale_interp.numel() > 0:
                multi_scale_interp = (multi_scale_interp - self.pt_min) / (self.pt_max - self.pt_min)
                multi_scale_interp = multi_scale_interp * 2 - 1
            out = grid_sample_wrapper(self.features.to(dtype=interp.dtype), multi_scale_interp).view(-1, self.feature_dim)
        else:
            out = multi_scale_interp

        if return_coords:
            return out, multi_scale_interp
        return out

    @torch.no_grad()
    def update_alpha_mask(self, grid_id: int = 0):
        assert len(self.config) <= 2, "Alpha-masking not supported for multiple layers of indirection."
        aabb = self.aabb(grid_id)
        grid_size = self.resolution(grid_id)
        grid_size_l = grid_size.cpu().tolist()
        step_size = self.raymarcher.get_step_size(aabb, grid_size)

        # Compute density on a regular grid (of shape grid_size)
        pts = self.get_points_on_grid(aabb, grid_size, max_voxels=None)
        density = self.query_density(pts, grid_id).view(-1)

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

    @torch.no_grad()
    def upsample_F(self, new_reso):
        grid_info = self.config[1]
        new_size = [self.features.shape[0]] + [new_reso] * grid_info["input_coordinate_dim"]
        if grid_info["input_coordinate_dim"] == 1:
            coords = torch.tensor(np.mgrid[-1:1:new_reso*1j], device=self.features.device, dtype=torch.float32)
        elif grid_info["input_coordinate_dim"] == 2:
            coords = torch.tensor(np.mgrid[-1:1:new_reso*1j, -1:1:new_reso*1j], device=self.features.device, dtype=torch.float32)
        elif grid_info["input_coordinate_dim"] == 3:
            coords = torch.tensor(np.mgrid[-1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j], device=self.features.device, dtype=torch.float32)
        elif grid_info["input_coordinate_dim"] == 4:
            coords = torch.tensor(np.mgrid[-1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j], device=self.features.device, dtype=torch.float32)
        elif grid_info["input_coordinate_dim"] == 5:
            coords = torch.tensor(np.mgrid[-1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j], device=self.features.device, dtype=torch.float32)
        elif grid_info["input_coordinate_dim"] == 6:
            coords = torch.tensor(np.mgrid[-1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j, -1:1:new_reso*1j], device=self.features.device, dtype=torch.float32)
        else:
            assert False, "feature upsampling not supported above 6 dimensions"
        # coords shape is [dim] + [new_reso]*dim, e.g. [5, 8, 8, 8, 8, 8]
        # self.features shape is [feature_dim] + [old_reso]*dim, e.g. [28, 4, 4, 4, 4, 4]
        # Reshape coords into [batch, 1, ..., n, grid_dim] as expected by grid_sample_wrapper
        coords = coords.view(coords.shape[0], -1)  # [dim, new_reso**dim]
        coords = torch.permute(coords, (1, 0))[None, :, :]  # [1, new_reso**dim, dim]
        self.features.data = grid_sample_wrapper(self.features[None, ...], coords).permute(1, 0).view(new_size)
        log.info(f'upsampled feature grid to shape {new_size}')

    def get_points_on_grid(self, aabb, grid_size, max_voxels: Optional[int] = None):
        """
        Returns points from a regularly spaced grids of size grid_size.
        Coordinates normalized between [aabb0, aabb1]

        :param aabb:
        :param grid_size:
        :param max_voxels:
        :return:
        """
        dev = aabb.device
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

    def query_density(self, pts: torch.Tensor, grid_id: int, return_feat: bool = False):
        pts_norm = self.normalize_coords(pts, grid_id)
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, grid_id)
        density = (
            self.density_act(self.decoder.compute_density(
                features, rays_d=None, precompute_color=False)).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        if return_feat:
            return density, features
        return density

    def forward(self, rays_o, rays_d, bg_color, grid_id=0, channels: Sequence[str] = ("rgb", "depth"), near_far=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        near_far : [batch, 2]
        """
        self.timer.reset()
        outputs = {}

        if self.use_proposal_sampling:
            # TODO: determining near-far should be done in a separate function this is super cluttered
            if near_far is None:
                nears, fars = intersect_with_aabb(
                    near_plane=2.0,  # TODO: This is hard-coded from synthetic nerf.
                    rays_o=rays_o,
                    rays_d=rays_d,
                    aabb=self.aabb(grid_id),
                    training=self.training)
                nears = nears[..., None]
                fars = fars[..., None]
            else:
                nears, fars = torch.split(near_far, [1, 1], dim=-1)
                if nears.shape[0] != rays_o.shape[0]:
                    ones = torch.ones_like(rays_o[..., 0:1])
                    nears = ones * nears
                    fars = ones * fars

            ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
            ray_samples, weights_list, ray_samples_list = self.raymarcher.generate_ray_samples(
                ray_bundle, density_fns=self.density_fns)
            outputs['weights_list'] = weights_list
            outputs['ray_samples_list'] = ray_samples_list
            outputs['ray_samples_list'].append(ray_samples)
            rays_d = ray_bundle.directions
            pts = ray_samples.get_positions()
            if self.spatial_distortion is not None:
                pts = self.spatial_distortion(pts)  # cube of side 2
            aabb = self.aabb(grid_id)
            mask = ((aabb[0] <= pts) & (pts <= aabb[1])).all(dim=-1)  # noqa
            deltas = ray_samples.deltas.squeeze()
            mask[deltas <= 0] = False
            z_vals = ((ray_samples.starts + ray_samples.ends) / 2).squeeze()
        else:
            rm_out = self.raymarcher.get_intersections2(
                rays_o, rays_d, self.aabb(grid_id), self.resolution(grid_id), perturb=self.training,
                is_ndc=self.is_ndc, is_contracted=self.is_contracted, near_far=near_far)
            rays_d = rm_out["rays_d"]      # [n_rays, 3]
            pts = rm_out["intersections"]  # [n_rays, n_intrs, 3]
            mask = rm_out["mask"]          # [n_rays, n_intrs]
            z_vals = rm_out["z_vals"]      # [n_rays, n_samples]
            deltas = rm_out["deltas"]      # [n_rays, n_samples]

        n_rays, n_intrs = pts.shape[:2]
        dev = rays_o.device

        # Filter intersections which have a low density according to the density mask
        # Contraction does not currently support density masking!
        if self.density_mask[grid_id] is not None and not self.is_contracted:
            # density_mask needs unnormalized coordinates: normalization happens internally
            # and can be with a different aabb than the current one.
            alpha_mask = self.density_mask[grid_id].sample_density(pts[mask]) > 0
            invalid_mask = ~mask
            invalid_mask[mask] |= (~alpha_mask)
            mask = ~invalid_mask

        if len(pts) == 0:
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

        self.timer.check("raymarcher")
        # compute features and render
        rays_d_rep = rays_d.view(-1, 1, 3).expand(pts.shape)
        masked_rays_d_rep = rays_d_rep[mask]
        density_masked, features = self.query_density(pts=pts[mask], grid_id=grid_id, return_feat=True)
        density = torch.zeros(n_rays, n_intrs, device=dev, dtype=density_masked.dtype)
        density[mask] = density_masked.view(-1)

        alpha, weight, transmission = raw2alpha(density, deltas * self.density_multiplier)  # Each is shape [batch_size, n_samples]
        if self.use_proposal_sampling:
            outputs['weights_list'].append(weight[..., None])
        self.timer.check("density")

        rgb_masked = self.decoder.compute_color(features, rays_d=masked_rays_d_rep)
        rgb = torch.zeros(n_rays, n_intrs, 3, device=dev, dtype=rgb_masked.dtype)
        rgb[mask] = rgb_masked
        rgb = torch.sigmoid(rgb)

        # Confirmed that torch.sum(weight, -1) matches 1-transmission[:,-1]
        acc_map = 1 - transmission[:, -1]
        self.timer.check("color")

        if "rgb" in channels:
            rgb_map = torch.sum(weight[..., None] * rgb, -2)
            if bg_color is not None:
                rgb_map = rgb_map + (1.0 - acc_map[..., None]) * bg_color#.to(rgb_map.device)
            outputs["rgb"] = rgb_map
        if "depth" in channels:
            depth_map = torch.sum(weight * z_vals, -1)  # [batch_size]
            depth_map = depth_map + (1.0 - acc_map) * rays_d[..., -1]  # Maybe the rays_d is to transform ray depth to absolute depth?
            outputs["depth"] = depth_map
        outputs["deltas"] = deltas
        outputs["weight"] = weight
        self.timer.check("render")
        return outputs

    def get_params(self, lr):
        return [
            {"params": self.parameters(), "lr": lr},
        ]
        # params = [
        #     {"params": self.scene_grids.parameters(), "lr": lr},
        # ]
        # if self.density_field is not None:
        #     params.append({"params": self.density_field.parameters(), "lr": lr})
        # if self.use_F:
        #     params.append({"params": [self.pt_min, self.pt_max], "lr": lr})
        # if not self.transfer_learning:
        #     params.append({"params": self.decoder.parameters(), "lr": lr})
        #     if self.use_F:
        #         params.append({"params": self.features, "lr": lr})
        # return params
