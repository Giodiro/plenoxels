"""
Density proposal field
"""
import itertools
from typing import List, Optional, Callable

import torch

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.models.utils import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


class TriplaneDensityField(LowrankModel):
    def __init__(self,
                 aabb: torch.Tensor,
                 resolution: List[int],
                 num_input_coords: int,
                 rank: int,
                 spatial_distortion: Optional[SpatialDistortion],
                 density_act: Callable):
        super(TriplaneDensityField, self).__init__()

        is_video = num_input_coords == 4
        config = {
            "input_coordinate_dim": num_input_coords,
            "output_coordinate_dim": 1,
            "grid_dimensions": 2,
            "resolution": resolution,
            "rank": rank
        }
        gpdesc = self.init_grid_param(
            config,
            is_video=is_video,
            grid_level=0,
            use_F=False,
            is_appearance=False  # Not entirely sure about this
        )
        self.set_resolution(gpdesc.reso, grid_id=0)
        # TODO: Enforce that when spatial_distribution is contraction, aabb must be [[-2, -2, -2], [2, 2, 2]]
        self.set_aabb(aabb, grid_id=0)
        self.grids = gpdesc.grid_coefs
        self.feature_dim = 1
        self.spatial_distortion = spatial_distortion
        self.rank = rank
        self.density_act = density_act

    def get_density(self, pts: torch.Tensor):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)  # cube of side 2
        pts = self.normalize_coords(pts)
        n_rays, n_samples = pts.shape[:2]
        pts = pts.view(-1, 3)  # TODO: Masking!

        # create plane combinations
        coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))
        interp_out = None
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            interp_out_plane = grid_sample_wrapper(self.grids[ci], pts[..., coo_comb]).view(-1, 1, self.rank)
            # compute product
            interp_out = interp_out_plane if interp_out is None else interp_out * interp_out_plane
        # average over rank
        interp = interp_out.mean(dim=-1)
        interp = self.density_act(interp.view(n_rays, n_samples, self.feature_dim))

        return interp

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)
