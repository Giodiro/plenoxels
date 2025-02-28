import itertools
import logging as log
from typing import Dict, List, Union, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np

from plenoxels.models.utils import (
    grid_sample_wrapper, compute_plane_tv, raw2alpha, init_features_param, init_grid_param
)
from .decoders import NNDecoder, SHDecoder, LearnedBasisDecoder
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
                 learnedbasis: bool,
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
        self.len_time = len_time  # maximum timestep - used for normalization
        super().__init__(grid_config=grid_config,
                         is_ndc=is_ndc,
                         is_contracted=is_contracted,
                         sh=sh,
                         learnedbasis=learnedbasis,
                         use_F=use_F,
                         global_scale=global_scale,
                         global_translation=global_translation,
                         density_activation=density_activation,
                         use_proposal_sampling=proposal_sampling,
                         density_field_resolution=kwargs.get('density_field_resolution', None),
                         density_field_rank=kwargs.get('density_field_rank', None),
                         num_proposal_samples=kwargs.get('num_proposal_samples', None),
                         proposal_feature_dim=kwargs.get('proposal_feature_dim', None),
                         proposal_decoder_type=kwargs.get('proposal_decoder_type', None),
                         n_intersections=n_intersections,
                         single_jitter=single_jitter,
                         raymarch_type=raymarch_type,
                         spacing_fn=kwargs.get('spacing_fn', None),
                         num_sample_multiplier=kwargs.get('num_sample_multiplier', None),
                         density_model=kwargs.get('density_model', None),
                         aabb=aabb,
                         multiscale_res=multiscale_res,
                         feature_len=kwargs.get('feature_len', None),
                         intrinsics=kwargs.get('intrinsics', None))
        self.extra_args = kwargs
        self.lookup_time = lookup_time
        self.trainable_rank = None
        self.concat_features = False
        if self.feature_len is not None:
            self.concat_features = True
        else:
            self.feature_len = [self.config[0]["output_coordinate_dim"]] * len(self.multiscale_res)
        
        appearance_code_size = kwargs.get('appearance_code_size', 32),
        color_net = kwargs.get('color_net', 2),
        if isinstance(appearance_code_size, tuple):
            appearance_code_size = appearance_code_size[0]
        if isinstance(color_net, tuple):
            color_net = color_net[0]
        print("\n\n\n")
        print("==> appearance_code_size", appearance_code_size)
        print("==> color_net", color_net)
        print("\n\n\n")
        self.grids = nn.ModuleList()
        # appearance_code_size = 48 # same as in nerf-w
        # appearance_code_size = 32 # seems to be a good size
        # appearance_code_size = 16 # seems to be a bit too small
        self.appearance_coef = nn.Parameter(nn.init.normal_(torch.empty([appearance_code_size, len_time])))
        
        for res, featlen in zip(self.multiscale_res, self.feature_len):
            for li, grid_config in enumerate(self.config):
                # initialize feature grid
                if "feature_dim" in grid_config:
                    self.features = None
                    self.feature_dim = grid_config["feature_dim"]
                    if self.use_F:
                        self.features = init_features_param(grid_config, self.sh)
                        self.feature_dim = self.features.shape[0]
                # initialize coordinate grid
                else:
                    config = grid_config.copy()
                    config["resolution"] = [r * res for r in config["resolution"][:3]]
                    # do not have multi resolution on time.
                    if len(grid_config["resolution"]) == 4:
                        config["resolution"] += [grid_config["resolution"][-1]]

                    gpdesc = init_grid_param(config, feature_len=featlen, is_video=False, is_appearance=True, grid_level=li, use_F=self.use_F)
                    self.set_resolution(gpdesc.reso, 0)
                    self.grids.append(gpdesc.grid_coefs)
                    #self.appearance_coef.append(gpdesc.appearance_coef)
        if not self.use_F:
            # Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim = sum(self.feature_len)
            else:
                self.feature_dim = gpdesc.grid_coefs[-1].shape[1] // config["rank"][0]
        if self.sh:
            self.decoder = SHDecoder(
                feature_dim=self.feature_dim,
                decoder_type=self.extra_args.get('sh_decoder_type', 'manual'))
        elif self.learnedbasis:
            self.decoder = LearnedBasisDecoder(
                feature_dim=self.feature_dim, net_width=64, net_layers=1
            )
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1, color_net=color_net, appearance_code_size=appearance_code_size)

        self.density_mask = None
        log.info(f"Initialized LowrankAppearance. "
                 f"time-reso={self.appearance_coef.shape[1]} - decoder={self.decoder}")

    def compute_features(self,
                         pts,
                         timestamps,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        multiscale_space : nn.ModuleList = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
        #appearance_coef : nn.ModuleList = self.appearance_coef  # time: [F_dim, time_reso]
        level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

        # if there are 6 planes then
        # interpolate in space and time
        if len(multiscale_space[0]) == 6:
            pts = torch.cat([pts, timestamps[:, None]], dim=-1)  # [batch, 4] for xyzt

        # Interpolate in space
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]),
            level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))

        multi_scale_interp = [] if self.concat_features else 0
        for scale_id, (grid_space, featlen) in enumerate(zip(multiscale_space, self.feature_len)):
            interp_space = 1  # [n, F_dim, rank]
            for ci, coo_comb in enumerate(coo_combs):

                # interpolate in plane
                interp_out_plane = grid_sample_wrapper(grid_space[ci], pts[..., coo_comb]).view(
                            -1, featlen, level_info["rank"])

                # compute product
                interp_space *= interp_out_plane

            # Combine space over rank
            interp = interp_space.mean(dim=-1)  # [n, F_dim]

            # Merge features across scales
            if self.concat_features:  # Concatenate over scale
                multi_scale_interp.append(interp)
            else:  # Sum over scales
                multi_scale_interp += interp

        if self.concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

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
            # ray_samples, weights_list, ray_samples_list, density_list = self.raymarcher.generate_ray_samples(
                # ray_bundle, timestamps=timestamps, density_fns=self.density_fns, return_density=False)
            ray_samples, weights_list, ray_samples_list = self.raymarcher.generate_ray_samples(
                ray_bundle, timestamps=timestamps, density_fns=self.density_fns, return_density=False)
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

            # if "proposal_depth" in channels:
            #     n_rays, n_intrs = pts.shape[:2]
            #     dev = rays_o.device

            #     for proposal_id in range(len(density_list)):
            #         density_proposal = density_list[proposal_id].squeeze()
            #         deltas_proposal = ray_samples_list[proposal_id].deltas.squeeze()
            #         z_vals_proposal = ((ray_samples_list[proposal_id].starts + ray_samples_list[proposal_id].ends) / 2).squeeze()
            #         _, weight_proposal, transmission_proposal = raw2alpha(density_proposal, deltas_proposal)  # Each is shape [batch_size, n_samples]
            #         acc_map_proposal = 1 - transmission_proposal[:, -1]
            #         #outputs[f"proposal_depth_{proposal_id}"] = torch.zeros(n_rays, 1, device=dev, dtype=rays_o.dtype)
            #         depth_map_proposal = torch.sum(weight_proposal * z_vals_proposal, -1)  # [batch_size]
            #         depth_map_proposal = depth_map_proposal + (1.0 - acc_map_proposal) * rays_d[..., -1]  # Maybe the rays_d is to transform ray depth to absolute depth?
            #         outputs[f"proposal_depth_{proposal_id}"] = depth_map_proposal
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

        # [batch_size]
        # [batch_size, n_intrs]
        times = timestamps[:, None].repeat(1, n_intrs)[mask]  # [n_rays * n_intrs]

        # Normalization (between [-1, 1])
        pts = self.normalize_coords(pts, 0)
        times = (times * 2 / self.len_time) - 1

        # assumes all rays are sampled at the same time
        # speed up look up a quite a bit
        if torch.unique(times).shape[0] == 1:
            appearance_idx = timestamps[0]
        else:
            appearance_idx = timestamps[:, None].repeat(1, n_intrs)[mask]
        
        # compute features and render
        rays_d_rep = rays_d.view(-1, 1, 3).expand(pts.shape)
        masked_rays_d_rep = rays_d_rep[mask]
        features = self.compute_features(pts[mask], times)
        density_masked = self.density_act(self.decoder.compute_density(features, rays_d=masked_rays_d_rep))
        density = torch.zeros(n_rays, n_intrs, device=pts.device, dtype=density_masked.dtype)
        density[mask] = density_masked.view(-1)

        alpha, weight, transmission = raw2alpha(density, deltas)  # Each is shape [batch_size, n_samples]
        if self.use_proposal_sampling:
            outputs['weights_list'].append(weight[..., None])

        # concatenate appearance code
        appearance_code = self.appearance_coef
        # project codes onto unit sphere or inide unit sphere. 
        # (keeps thinggs compact such that interpolation should be easier)
        # div = torch.norm(appearance_code, dim=-1) + 1e-6
        # appearance_code = appearance_code / torch.max(div, torch.ones_like(div))[:, None]
        
        if appearance_idx.shape == torch.Size([]):
            # Interpolate instead of rounding (used for video generation)
            code0 = appearance_code[:, torch.floor(appearance_idx).long()]
            code1 = appearance_code[:, torch.ceil(appearance_idx).long()]
            interpweight = appearance_idx - torch.floor(appearance_idx)
            # print(f'weight {interpweight} between {torch.floor(appearance_idx).long()} and {torch.ceil(appearance_idx).long()}')
            appearance_code = interpweight * code1 + (1.0 - interpweight) * code0
            # print(f'appearance_code is {appearance_code}')
            appearance_code = appearance_code.unsqueeze(0).repeat(pts[mask].shape[0], 1)  # [n, 16]
            # Rounding (used in training)
            # appearance_code = appearance_code[:, appearance_idx.long()].unsqueeze(0).repeat(pts[mask].shape[0], 1)  # [n, 16]
        else:
            appearance_code = appearance_code[:, appearance_idx.long()].permute(1, 0)  # [n, 16]

        self.decoder.density_rgb = torch.cat([self.decoder.density_rgb, appearance_code], dim=1)

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
