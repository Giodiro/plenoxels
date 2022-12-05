"""
Density proposal field
"""
import itertools
from typing import List, Optional, Callable
import logging as log

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.models.decoders.mlp_decoder import SigmaNNDecoder
from plenoxels.models.kplane_field import interpolate_ms_features, normalize_aabb
from plenoxels.models.utils import grid_sample_wrapper, init_grid_param
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


class KPlaneDensityField(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 num_input_coords,
                 num_output_coords,
                 density_activation: Callable,
                 spatial_distortion: Optional[SpatialDistortion] = None):
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.hexplane = num_input_coords == 4
        self.feature_dim = num_output_coords
        self.density_activation = density_activation

        self.grids = init_grid_param(
            grid_nd=2, in_dim=num_input_coords, out_dim=num_output_coords, reso=resolution)
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        log.info(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={resolution}")
        log.info(f"KPlaneDensityField grids: \n{self.grids}")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None and self.hexplane:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
        ).view(n_rays, n_samples, 1)
        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)


class TriplaneDensityField(nn.Module):
    def __init__(self,
                 aabb: torch.Tensor,
                 resolution: List[int],
                 num_input_coords: int,
                 num_output_coords: int,
                 rank: int,
                 spatial_distortion: Optional[SpatialDistortion],
                 density_act: Callable,
                 decoder_type: str,
                 len_time: Optional[int] = None):
        super(TriplaneDensityField, self).__init__()

        self.hexplane = num_input_coords == 4
        if self.hexplane:
            assert len_time is not None
        self.feature_dim = num_output_coords
        if decoder_type == 'sh':
            assert self.feature_dim == 1, 'SH decoder for density field requires 1 output coordinate'
        config = {
            "input_coordinate_dim": num_input_coords,
            "grid_dimensions": 2,
            "resolution": resolution,
            "rank": rank
        }
        gpdesc = init_grid_param(
            config,
            feature_len=self.feature_dim,
            is_video=self.hexplane,
            grid_level=0,
            use_F=False,
            is_appearance=False  # Not entirely sure about this
        )
        self.resolution = nn.Parameter(gpdesc.reso, requires_grad=False)
        # TODO: Enforce that when spatial_distribution is contraction, aabb must be [[-2, -2, -2], [2, 2, 2]]
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grids = gpdesc.grid_coefs
        self.spatial_distortion = spatial_distortion
        self.rank = rank
        self.density_act = density_act
        self.len_time = len_time
        if decoder_type == 'sh':
            self.decoder = lambda x, rays_d: x
        elif decoder_type == 'nn':
            self.decoder = SigmaNNDecoder(feature_dim=self.feature_dim)
        else:
            raise ValueError(f'invalid decoder type {decoder_type}.')
        log.info(f"Initialized TriplaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={self.resolution} - time-length={self.len_time}")
        log.info(f"TriplaneDensityField grids: \n{self.grids}")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """
        :param pts:
            tensor of xyz, in world coordinates. Shape [n_rays, n_samples, 3]
        :param timestamps:
            tensor of times (unnormalized). Shape [n_rays]
        :return:
            tensor of densities [n_rays, n_samples, 1]
        """
        if self.hexplane:
            assert timestamps is not None
            timestamps = (timestamps * 2 / self.len_time) - 1  # Normalize timestamps
        n_rays, n_samples = pts.shape[:2]
        # 1. Contract
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)  # cube of side 2
        # 2. Normalize
        pts = (pts - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1
        # 3. Combine xyz with time
        if self.hexplane:
            pts = torch.cat([pts, timestamps[:, None].expand(-1, n_samples)[..., None]], dim=-1)
            pts = pts.view(-1, 4)  # TODO: Masking!
        else:
            pts = pts.view(-1, 3)  # TODO: Masking!
        # 4. Interpolate over all plane combinations
        coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))
        interp_out = None
        for ci, coo_comb in enumerate(coo_combs):
            interp_out_plane = grid_sample_wrapper(
                self.grids[ci], pts[..., coo_comb]).view(-1, self.feature_dim, self.rank)
            interp_out = interp_out_plane if interp_out is None else interp_out * interp_out_plane
        # 5. Average over rank
        interp = interp_out.mean(dim=-1)
        # 6. Decode and activate
        interp = self.density_act(
            self.decoder(
                interp, rays_d=None,
            ).view(n_rays, n_samples, 1)
        )
        return interp

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)
