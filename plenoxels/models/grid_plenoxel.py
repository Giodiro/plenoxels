from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha
from plenoxels.models.utils import interp_regular, get_intersections


class RegularGrid(nn.Module):
    def __init__(self,
                 resolution: int,
                 radius: float,
                 sh_deg: int,
                 sh_encoder):
        super().__init__()
        self.data_dim = ((sh_deg + 1) ** 2) * 3 + 1
        self.resolution = resolution
        self.radius = radius
        self.step_size, self.n_intersections = self.calc_step_size()
        self.white_bkgd = True
        self.sh_encoder = sh_encoder

        self.data = nn.Parameter(torch.empty(
            1, self.data_dim, resolution, resolution, resolution))
        with torch.no_grad():
            self.data[0, :-1, ...].fill_(0.1)
            self.data[0, -1, ...].fill_(0.01)

    def __repr__(self):
        return (f"RegularGrid(data_dim={self.data_dim}, step_size={self.step_size}, "
                f"n_intersections={self.n_intersections}, sh_encoder={self.sh_encoder}, "
                f"radius={self.radius}, resolution={self.resolution})")

    def calc_step_size(self) -> Tuple[float, int]:
        voxel_size = (self.radius * 2) / self.resolution
        step_size = voxel_size / 2
        n_intersections = np.sqrt(3.) * 2 * self.resolution
        return step_size, n_intersections

    def normalize_coord(self, intersections: torch.Tensor) -> torch.Tensor:
        """Returns coordinates normalized between -1 and +1"""
        return intersections / self.radius

    def forward(self, rays_o, rays_d, use_fp16: bool = False):
        intrs_pts, intersections, intrs_pts_mask = get_intersections(
            rays_o, rays_d, self.radius, self.n_intersections, self.step_size)
        batch = intersections.shape[0]
        nintrs = intersections.shape[1] - 1
        intrs_pts = self.normalize_coord(intrs_pts)

        with torch.autocast(device_type="cuda", enabled=False):
            data_interp = interp_regular(
                self.data, intrs_pts.view(1, -1, 1, 1, 3))  # [ch, mask_pts]
            
            if data_interp.dim() == 1:  # happens if mask_pts == 1
                data_interp = data_interp.unsqueeze(1)
            data_interp = data_interp.T  # [mask_pts, ch]

            # 1. Process density: Un-masked sigma (batch, n_intrs-1), and compute.
            sigma_masked = data_interp[:, -1]
            sigma = torch.zeros(batch, nintrs, dtype=sigma_masked.dtype, device=sigma_masked.device)
            sigma.masked_scatter_(intrs_pts_mask, sigma_masked)
            sigma = F.relu(sigma)
            alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)  # both [batch, n_intrs-1]

            # 3. Create SH coefficients and mask them
            sh_mult = self.sh_encoder(rays_d).unsqueeze(1).expand(batch, nintrs, -1)  # [batch, nintrs, ch/3]
            sh_mult = sh_mult[intrs_pts_mask].unsqueeze(1)  # [mask_pts, 1, ch/3]

            # 4. Interpolate rgbdata, use SH coefficients to get to RGB
            sh_masked = data_interp[:, :-1]
            sh_masked = sh_masked.view(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
            rgb_masked = torch.sum(sh_mult * sh_masked, dim=-1)   # [mask_pts, 3]

            # 5. Post-process RGB
            rgb = torch.zeros(batch, nintrs, 3, dtype=rgb_masked.dtype, device=rgb_masked.device)
            rgb.masked_scatter_(intrs_pts_mask.unsqueeze(-1), rgb_masked)
            rgb = shrgb2rgb(rgb, abs_light, self.white_bkgd)

        return rgb
