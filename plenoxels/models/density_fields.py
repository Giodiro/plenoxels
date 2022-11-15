"""
Density proposal field
"""
import itertools
from typing import List, Optional, Callable
import logging as log

import torch
import torch.nn as nn

from plenoxels.models.decoders.mlp_decoder import SigmaNNDecoder
from plenoxels.models.utils import grid_sample_wrapper, init_grid_param
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


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
