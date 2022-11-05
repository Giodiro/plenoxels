import itertools
import logging as log
from typing import Dict, List, Union, Sequence, Tuple

import torch
import torch.nn as nn

from plenoxels.models.utils import (
    grid_sample_wrapper, compute_plane_tv, raw2alpha
)
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel
from ..ops.bbox_colliders import intersect_with_aabb
from ..raymarching.ray_samplers import RayBundle


class LowrankAppearance(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: torch.Tensor,  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                 len_time: int,
                 is_ndc: bool,
                 is_contracted: bool,
                 sh: bool,
                 use_F: bool,
                 lookup_time: bool,
                 density_activation: str,
                 proposal_sampling: bool,
                 n_intersections: int,
                 single_jitter: bool,
                 raymarch_type: str,
                 multiscale_res: List[int] = [1],
                 global_translation=None,
                 global_scale=None,
                 **kwargs):
        super().__init__(grid_config=grid_config,
                         is_ndc=is_ndc,
                         is_contracted=is_contracted,
                         sh=sh,
                         use_F=use_F,
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
                         num_samples_multiplier=kwargs.get('num_samples_multiplier', None))
        self.set_aabb(aabb, 0)
        self.len_time = len_time  # maximum timestep - used for normalization
        self.extra_args = kwargs
        self.lookup_time = lookup_time
        self.multiscale_res = multiscale_res
        self.trainable_rank = None

        # For now, only allow a single index grid and a single feature grid, not multiple layers
        assert len(self.config) == 2
        self.grids = nn.ModuleList()

        for res in self.multiscale_res:
            for li, grid_config in enumerate(self.config):
                # initialize feature grid
                if "feature_dim" in grid_config:
                    self.features = None
                    self.feature_dim = grid_config["feature_dim"]
                    if self.use_F:
                        self.features = self.init_features_param(grid_config, self.sh)
                        self.feature_dim = self.features.shape[0]
                # initialize coordinate grid
                else:
                    config = grid_config.copy()
                    config["resolution"] = [r * res for r in config["resolution"][:3]]
                    # do not have multi resolution on time.
                    if len(grid_config["resolution"]) == 4:
                        config["resolution"] += [grid_config["resolution"][-1]]

                    gpdesc = self.init_grid_param(config, is_video=False, is_appearance=True, grid_level=li, use_F=self.use_F)
                    self.set_resolution(gpdesc.reso, 0)
                    self.grids.append(gpdesc.grid_coefs)

        # if sh + density in grid, then we do not want appearance code to influence density
        self.appearance_coef = nn.Parameter(nn.init.ones_(torch.empty([self.feature_dim - 1, len_time])))  # no time dependence

        if self.sh:
            self.decoder = SHDecoder(
                feature_dim=self.feature_dim,
                decoder_type=self.extra_args.get('sh_decoder_type', 'manual'))
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)

        self.density_mask = None
        log.info(f"Initialized LowrankAppearance. "
                 f"time-reso={self.appearance_coef.shape[1]} - decoder={self.decoder}")

    def compute_features(self,
                         pts,
                         timestamps,
                         appearance_idx,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        multiscale_space : nn.ModuleList = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
        appearance_coef = self.appearance_coef  # time: [F_dim, time_reso]
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

        # if there are 6 planes then
        # interpolate in space and time
        if len(multiscale_space[0]) == 6:
            pts = torch.cat([pts, timestamps[:,None]], dim=-1)  # [batch, 4] for xyzt

        # Interpolate in space
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]),
            level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))

        multi_scale_interp = 0
        for scale_id, grid_space in enumerate(multiscale_space):
            interp_space = None  # [n, F_dim, rank]
            for ci, coo_comb in enumerate(coo_combs):

                # interpolate in plane
                interp_out_plane = grid_sample_wrapper(grid_space[ci], pts[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"])

                # compute product
                interp_space = interp_out_plane if interp_space is None else interp_space * interp_out_plane

            # Combine space over rank
            interp = interp_space.mean(dim=-1)  # [n, F_dim]

            # sum over scales
            multi_scale_interp += interp

        # combine with appearance code
        appearance_code = appearance_coef[:, appearance_idx.long()].unsqueeze(0).repeat(pts.shape[0], 1)  # [n, F_dim]
        appearance_code = appearance_code.view(-1, self.feature_dim-1)  # [n, F_dim]

        # add density one to appearance code
        appearance_code = torch.cat([appearance_code, torch.ones_like(appearance_code[:, 0:1])], dim=1)
        multi_scale_interp = multi_scale_interp * appearance_code

        if self.use_F:
            # Learned normalization
            if interp.numel() > 0:
                multi_scale_interp = (multi_scale_interp - self.pt_min) / (self.pt_max - self.pt_min)
                multi_scale_interp = multi_scale_interp * 2 - 1

            out = grid_sample_wrapper(self.features, multi_scale_interp).view(-1, self.feature_dim)
        else:
            out = multi_scale_interp

        if return_coords:
            return out, multi_scale_interp
        return out

    def forward(self, rays_o, rays_d, timestamps, bg_color,
                channels: Sequence[str] = ("rgb", "depth"), near_far=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        outputs = {}
        # Normalize rays_d
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

        if self.use_proposal_sampling:
            # TODO: determining near-far should be done in a separate function this is super cluttered
            if near_far is None:
                nears, fars = intersect_with_aabb(
                    near_plane=2.0,  # TODO: This is hard-coded from synthetic nerf.
                    rays_o=rays_o,
                    rays_d=rays_d,
                    aabb=self.aabb(0),
                    training=self.training)
                nears = nears[..., None]
                fars = fars[..., None]
            else:
                nears, fars = torch.split(near_far, [1, 1], dim=-1)
                if nears.shape[0] != rays_o.shape[0]:
                    ones = torch.ones_like(rays_o[..., 0:1])
                    nears = ones * nears
                    fars = ones * fars

            aabb=self.aabb(0)
            ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
            ray_samples, weights_list, ray_samples_list, density_list = self.raymarcher.generate_ray_samples(
                ray_bundle, density_fns=self.density_fns)
            outputs['weights_list'] = weights_list
            outputs['ray_samples_list'] = ray_samples_list
            outputs['ray_samples_list'].append(ray_samples)
            rays_d = ray_bundle.directions
            pts = ray_samples.get_positions()
            if self.spatial_distortion is not None:
                pts = self.spatial_distortion(pts)  # cube of side 2
            mask = ((aabb[0] <= pts) & (pts <= aabb[1])).all(dim=-1)  # noqa
            deltas = ray_samples.deltas.squeeze()
            z_vals = ((ray_samples.starts + ray_samples.ends) / 2).squeeze()
            
            if "proposal_depth" in channels:
                n_rays, n_intrs = pts.shape[:2]
                dev = rays_o.device
                for proposal_id in range(len(density_list) - 1):
                    density = density_list[0].squeeze()
                    deltas = ray_samples_list[0].deltas.squeeze()
                    alpha, weight, transmission = raw2alpha(density, deltas)  # Each is shape [batch_size, n_samples]
                    acc_map = 1 - transmission[:, -1]
                    outputs[f"proposal_depth_{proposal_id}"] = torch.zeros(n_rays, 1, device=dev, dtype=rays_o.dtype)
                    depth_map = torch.sum(weight * z_vals, -1)  # [batch_size]
                    depth_map = depth_map + (1.0 - acc_map) * rays_d[..., -1]  # Maybe the rays_d is to transform ray depth to absolute depth?
                    outputs[f"proposal_depth{proposal_id}"] = depth_map
        else:
            rm_out = self.raymarcher.get_intersections2(
                rays_o, rays_d, self.aabb(0), self.resolution(0), perturb=self.training,
                is_ndc=self.is_ndc, is_contracted=self.is_contracted, near_far=near_far)
            rays_d = rm_out["rays_d"]                   # [n_rays, 3]
            pts = rm_out["intersections"]  # [n_rays, n_intrs, 3]
            mask = rm_out["mask"]                       # [n_rays, n_intrs]
            z_vals = rm_out["z_vals"]                   # [n_rays, n_intrs]
            deltas = rm_out["deltas"]                   # [n_rays, n_intrs]

        n_rays, n_intrs = pts.shape[:2]
        dev = rays_o.device

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

        times = timestamps[:, None].repeat(1, n_intrs)[mask]  # [n_rays * n_intrs]

        # Normalization (between [-1, 1])
        pts = self.normalize_coords(pts, 0)
        times = (times * 2 / self.len_time) - 1

        # assumes all rays are sampled at the same time
        # speed up look up a quite a bit
        appearance_idx = timestamps[0]

        # compute features and render
        rays_d_rep = rays_d.view(-1, 1, 3).expand(pts.shape)
        masked_rays_d_rep = rays_d_rep[mask]
        features = self.compute_features(pts[mask], times, appearance_idx)
        density_masked = self.density_act(self.decoder.compute_density(features, rays_d=masked_rays_d_rep))
        density = torch.zeros(n_rays, n_intrs, device=pts.device, dtype=density_masked.dtype)
        density[mask] = density_masked.view(-1)

        alpha, weight, transmission = raw2alpha(density, deltas)  # Each is shape [batch_size, n_samples]
        if self.use_proposal_sampling:
            outputs['weights_list'].append(weight[..., None])

        rgb_masked = self.decoder.compute_color(features, rays_d=masked_rays_d_rep)
        rgb = torch.zeros(n_rays, n_intrs, 3, device=pts.device, dtype=rgb_masked.dtype)
        rgb[mask] = rgb_masked
        rgb = torch.sigmoid(rgb)

        # Confirmed that torch.sum(weight, -1) matches 1-transmission[:, -1]
        acc_map = 1 - transmission[:, -1]

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

        return outputs

    def get_params(self, lr):
        params = [
            {"params": self.grids.parameters(), "lr": lr},
            {"params": self.appearance_coef, "lr": lr},
            {"params": self.decoder.parameters(), "lr": lr},
        ]

        if len(self.density_fields) > 0:
            params.append({"params": self.density_fields.parameters(), "lr": lr})
        if self.use_F:
            params.append({"params": [self.pt_min, self.pt_max], "lr": lr})
            params.append({"params": self.features, "lr": lr})
        return params


    def compute_plane_tv(self):
        multiscale_grids = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]

        total = 0
        for grid_space in multiscale_grids:

            if len(grid_space) == 6:
                # only use tv on spatial planes
                grid_ids = [0,1,3]
            else:
                grid_ids = list(range(len(grid_space)))

            for grid_id in grid_ids:
                grid = grid_space[grid_id]
                total += compute_plane_tv(grid)

        #total += compute_line_tv(grid_time)
        return total
