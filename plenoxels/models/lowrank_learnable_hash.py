import itertools
import logging as log
from typing import Dict, List, Union, Optional, Sequence, Tuple
from collections import Iterable

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
                 train_scale_steps: List[int] = [],
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
                         proposal_feature_dim=kwargs.get('proposal_feature_dim', None),
                         proposal_decoder_type=kwargs.get('proposal_decoder_type', None),
                         n_intersections=n_intersections,
                         single_jitter=single_jitter,
                         raymarch_type=raymarch_type,
                         spacing_fn=kwargs.get('spacing_fn', None),
                         num_sample_multiplier=kwargs.get('num_sample_multiplier', None),
                         aabb=aabb,
                         multiscale_res=multiscale_res,
                         feature_len=kwargs.get('feature_len', None))
        self.extra_args = kwargs
        self.density_multiplier = self.extra_args.get("density_multiplier")
        self.transfer_learning = self.extra_args["transfer_learning"]
        self.alpha_mask_threshold = self.extra_args["density_threshold"]
        self.concat_features = False
        if self.feature_len is not None:
            self.concat_features = True
        else:
            self.feature_len = [self.config[0]["output_coordinate_dim"]] * len(self.multiscale_res)

        self.scene_grids = nn.ModuleList()
        self.features = nn.ParameterList()
        for si in range(num_scenes):
            multi_scale_grids = nn.ModuleList()
            for res, featlen in zip(self.multiscale_res, self.feature_len):
                grids = nn.ModuleList()
                for li, grid_config in enumerate(self.config):
                    # initialize feature grid
                    if "feature_dim" in grid_config and si == 0:  # Only make one set of features
                        if self.use_F:
                            self.features.append(init_features_param(grid_config, self.sh))
                            self.feature_dim = self.features[-1].shape[0]
                    # initialize coordinate grid
                    else:
                        gpdesc = init_grid_param(
                            grid_nd=grid_config["grid_dimensions"],
                            in_dim=grid_config["input_coordinate_dim"],
                            out_dim=featlen, reso=[r * res for r in grid_config["resolution"]],
                            is_video=False, is_appearance=False, use_F=self.use_F
                        )
                        if li == 0:
                            self.set_resolution(gpdesc.reso, grid_id=si)
                        grids.append(gpdesc.grid_coefs)
                        for gc in gpdesc.grid_coefs:
                            log.info(f"Initialized grid with shape {gc.shape}")
                        if not self.use_F:
                            # shape[1] is out-dim * rank - Concatenate over feature len for each scale
                            if self.concat_features:
                                self.feature_dim = sum(self.feature_len)
                            else:
                                self.feature_dim = gpdesc.grid_coefs[-1].shape[1]
                multi_scale_grids.append(grids)
            self.scene_grids.append(multi_scale_grids)
        if self.sh:
            self.decoder = SHDecoder(
                feature_dim=self.feature_dim,
                decoder_type=self.extra_args.get('sh_decoder_type', 'manual'))
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        self.density_mask = nn.ModuleList([None] * num_scenes)
        self.hooks = None
        self.train_scale_steps = train_scale_steps
        if len(self.train_scale_steps) > 0:
            assert len(self.train_scale_steps) == len(multiscale_res) - 1, "Specify which steps to use to increment the set of trainable scales"
            self.trainable_scale = 1
            self.update_trainable_scale()
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, "
                 f"decoder: {self.decoder}. Raymarcher: {self.raymarcher}")

    def step_cb(self, step, max_steps):
        pass

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         ) -> torch.Tensor:
        mulitres_grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grids_info = self.config

        multi_scale_interp = [] if self.concat_features else 0
        for scale_id, (res, featlen) in enumerate(zip(self.multiscale_res, self.feature_len)):  # noqa
            grids: nn.ParameterList = mulitres_grids[scale_id]
            for level_info, grid in zip(grids_info, grids):
                if "feature_dim" in level_info:
                    continue

                # create plane combinations
                coo_combs = list(itertools.combinations(
                    range(pts.shape[-1]),
                    level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))

                interp_out = 1
                for ci, coo_comb in enumerate(coo_combs):
                    # interpolate in plane
                    interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(
                                -1, featlen, level_info["rank"])
                    # compute product
                    interp_out = interp_out_plane if interp_out is None else interp_out * interp_out_plane
                    # interp_out = interp_out_plane if interp_out is None else interp_out + interp_out_plane # Addition ablation study
            # average over rank
            interp = interp_out.mean(dim=-1)

            if self.use_F:
                if interp.numel() > 0:
                    interp = (interp - self.pt_min[scale_id]) / (self.pt_max[scale_id] - self.pt_min[scale_id])
                    interp = interp * 2 - 1
                interp = grid_sample_wrapper(
                    self.features[scale_id].to(dtype=interp.dtype), interp
                ).view(-1, self.feature_dim)

            if self.concat_features:  # Concatenate over scale
                multi_scale_interp.append(interp)
            else:  # Sum over scales
                multi_scale_interp += interp
        if self.concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp  # noqa

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

    def do_proposal_sampling(self, raybundle: RayBundle):
        ray_samples, weights_list, ray_samples_list, density_list = self.raymarcher.generate_ray_samples(
            raybundle, density_fns=self.density_fns, return_density=True)
        outputs['weights_list'] = weights_list
        outputs['ray_samples_list'] = ray_samples_list
        outputs['ray_samples_list'].append(ray_samples)
        rays_d = raybundle.directions

    def do_volumetric_sampling(self, raybundle: RayBundle):
        pass


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
            ray_samples, weights_list, ray_samples_list, density_list = self.raymarcher.generate_ray_samples(
                ray_bundle, density_fns=self.density_fns, return_density=True)
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
            # Output depth of the proposal samples for visualization purposes.
            if "proposal_depth" in channels:
                for proposal_id in range(len(density_list)):
                    density_proposal = density_list[proposal_id].squeeze()
                    deltas_proposal = ray_samples_list[proposal_id].deltas.squeeze()
                    z_vals_proposal = ((ray_samples_list[proposal_id].starts + ray_samples_list[proposal_id].ends) / 2).squeeze()
                    _, weight_proposal, transmission_proposal = raw2alpha(density_proposal, deltas_proposal)  # Each is shape [batch_size, n_samples]
                    acc_map_proposal = 1 - transmission_proposal[:, -1]
                    depth_map_proposal = torch.sum(weight_proposal * z_vals_proposal, -1)  # [batch_size]
                    depth_map_proposal = depth_map_proposal + (1.0 - acc_map_proposal) * rays_d[..., -1]  # Maybe the rays_d is to transform ray depth to absolute depth?
                    outputs[f"proposal_depth_{proposal_id}"] = depth_map_proposal

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

        alpha, weight, transmission = raw2alpha(density, deltas)  # Each is shape [batch_size, n_samples]
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
