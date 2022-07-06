from typing import Union, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.models.utils import interp_regular, ensure_list, get_intersections
from plenoxels.nerf_rendering import shrgb2rgb, depth_map, sigma2alpha


class PixelShuffle3d(nn.Module):
    """
    This class is a 3d version of pixelshuffle.
    """
    def __init__(self, scale):
        """
        :param scale: upsample scale
        """
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        x_view = x.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = x_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
                m.append(PixelShuffle3d(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, (3, 3, 3), (1, 1, 1), (1, 1, 1)))
            m.append(PixelShuffle3d(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SuperResoPlenoxel(nn.Module):
    def __init__(self,
                 coarse_reso: int,
                 reso_multiplier: int,
                 param_dim: int,
                 radius: Union[float, List[float]],
                 num_scenes: int,
                 sh_deg: int,
                 sh_encoder):
        super().__init__()
        self.radius: List[float] = ensure_list(radius, expand_size=num_scenes)
        self.coarse_reso = coarse_reso
        self.reso_multiplier = reso_multiplier
        self.fine_reso = self.coarse_reso * self.reso_multiplier
        self.data_dim = ((sh_deg + 1) ** 2) * 3 + 1
        self.sh_encoder = sh_encoder
        self.param_dim = param_dim
        self.num_scenes = num_scenes
        assert self.reso_multiplier == 4, "reso_multiplier = 4 is only one supported"
        assert len(self.radius) == self.num_scenes, "Radii != number of scenes"

        self.step_size, self.n_intersections = self.calc_step_size()
        print("Ray-marching with step-size = %.4e  -  %d intersections" %
              (self.step_size, self.n_intersections))

        self.grids = nn.ParameterList()
        for scene in range(self.num_scenes):
            self.grids.append(nn.Parameter(
                torch.empty(1, self.coarse_reso, self.coarse_reso, self.coarse_reso, self.param_dim)))

        # io for super-resolution network:
        # input: 1, param_dim, rc, rc, rc - output: 1, data_dim, rf, rf, rf
        self.sr_net_main_1 = nn.Sequential(
            nn.Conv3d(self.param_dim, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.sr_net_resi_1 = nn.Sequential(
            nn.Conv3d(self.param_dim, 32, (1, 1, 1), stride=(1, 1, 1)),
        )
        self.upsample_1 = Upsample(scale=self.reso_multiplier, num_feat=32)
        self.sr_net_main_2 = nn.Sequential(
            nn.Conv3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, self.data_dim, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.sr_net_resi_2 = nn.Sequential(
            nn.Conv3d(32, self.data_dim, (1, 1, 1), stride=(1, 1, 1))
        )

        self.init_params()

    def init_params(self):
        if self.grids is not None:
            for grid in self.grids:
                nn.init.constant_(grid[:-1, ...], 0.1)
                nn.init.constant_(grid[-1, ...], 0.01)

    def get_radius(self, dset_id: int) -> float:
        return self.radius[dset_id]

    def calc_step_size(self) -> Tuple[float, int]:
        # Smallest radius, largest fine-resolution
        smallest_radius = np.min(self.radius)
        units = (smallest_radius * 2) / (self.fine_reso - 1)
        step_size = units / 2
        grid_diag = math.sqrt(3) * np.max(self.radius) * 2
        n_intersections = int(grid_diag / step_size) - 1
        return step_size, n_intersections

    def normalize01(self, pts: torch.Tensor, dset_id: int) -> torch.Tensor:
        """Normalize from world coordinates to 0-1"""
        radius = self.get_radius(dset_id)
        return (pts + radius) / (radius * 2)

    def forward(self, rays_o, rays_d, grid_id, consistency_coef=0, level=None, run_fp16=False, verbose=False):
        # Upsample the coarse-grid
        big_grid = self.upsample_1(self.sr_net_main_1(self.grids[grid_id]) + self.sr_net_resi1(self.grids[grid_id]))
        big_grid = self.sr_net_main_2(big_grid) + self.sr_net_resi_2(big_grid)

        # Get intersections with the upsampled grid
        intrs_pts, intersections, intrs_pts_mask = get_intersections(
            rays_o, rays_d, self.radius[grid_id], self.n_intersections, self.step_size)
        batch = intersections.shape[0]
        nintrs = intersections.shape[1] - 1
        intrs_pts = intrs_pts[intrs_pts_mask]

        # Normalize pts in [-1, 1]
        intrs_pts = self.normalize01(intrs_pts, grid_id) * 2 - 1

        # Interpolate intersecting points on the upsampled grid
        intrs_pts = intrs_pts.view(1, -1, 1, 1, 3)  # 1, batch * n_intrs, 1, 1, 3
        big_grid = big_grid.permute(0, 4, 1, 2, 3)  # 1, data_dim, reso, reso, reso
        data_interp = interp_regular(big_grid, intrs_pts).T  # batch * n_intrs, data_dim

        # 1. Process density: Un-masked sigma (batch, n_intrs-1), and compute.
        sigma_masked = data_interp[:, -1]
        sigma_masked = F.relu(sigma_masked)
        sigma = torch.zeros(batch, nintrs, dtype=sigma_masked.dtype, device=sigma_masked.device)
        sigma.masked_scatter_(intrs_pts_mask, sigma_masked)
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
        rgb = shrgb2rgb(rgb, abs_light, True)

        # 6. Depth map (optional)
        depth = depth_map(abs_light, intersections)  # [batch]

        return rgb, depth, alpha, torch.tensor(0.0)

    def __repr__(self):
        return (f"SuperResoPlenoxel(coarse_reso={self.coarse_reso}, "
                f"reso_multiplier={self.reso_multiplier}, data_dim={self.data_dim}, "
                f"param_dim={self.param_dim})")
