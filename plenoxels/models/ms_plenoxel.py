from importlib.machinery import PathFinder
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.nerf_rendering import shrgb2rgb, depth_map, sigma2alpha

try:
    spec = PathFinder().find_spec("c_ext", [str(Path(__file__).resolve().parents[1])])
    torch.ops.load_library(spec.origin)
except:
    print("Failed to load C-extension necessary for DictPlenoxels model")
    raise


class DictPlenoxels(nn.Module):
    def __init__(self,
                 sh_deg: int,
                 sh_encoder,
                 fine_reso: int,
                 coarse_reso: int,
                 radius: float,
                 num_atoms: int,
                 num_scenes: int,
                 efficient_dict=True):
        super().__init__()
        assert not efficient_dict
        sh_dim = (sh_deg + 1) ** 2
        total_data_channels = sh_dim * 3 + 1
        self.radius = radius
        self.fine_reso = fine_reso
        self.coarse_reso = coarse_reso
        self.num_atoms = num_atoms
        self.data_dim = total_data_channels
        self.efficient_dict = efficient_dict
        self.sh_encoder = sh_encoder
        self.coarse_voxel_len = radius * 2 / coarse_reso
        self.fine_voxel_len = self.coarse_voxel_len / fine_reso
        self.step_size = self.fine_voxel_len / 2.
        self.n_intersections = coarse_reso * 3 * 2 * fine_reso

        self.grids = nn.ParameterList([nn.Parameter(
            torch.empty(coarse_reso ** 3, self.num_atoms)) for i in range(num_scenes)])
        self.atoms = nn.Parameter(torch.empty(fine_reso ** 3, self.num_atoms, self.data_dim))
        self.init_params()

    def init_params(self):
        for grid in self.grids:
            nn.init.normal_(grid, std=0.01)
        nn.init.uniform_(self.atoms[..., :-1])
        nn.init.constant_(self.atoms[..., -1], 0.01)

    def __repr__(self):
        return (f"DictPlenoxels(grids={self.grids}, num_atoms={self.num_atoms}, data_dim={self.data_dim}, "
                f"fine_reso={self.fine_reso}, coarse_reso={self.coarse_reso})")

    @torch.no_grad()
    def sample_proposal(self, rays_o, rays_d):
        dev, dt = rays_o.device, rays_o.dtype
        offsets_pos = (self.radius - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (-self.radius - rays_o) / rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]

        steps = torch.arange(self.n_intersections, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
        steps = steps.repeat(rays_d.shape[0], 1)   # [batch, n_intrs]
        intersections = start + steps * self.step_size  # [batch, n_intrs]
        return intersections

    def forward(self, rays_o, rays_d, grid_id, verbose=False):
        grid = self.grids[grid_id]
        with torch.autograd.no_grad():
            intersections = self.sample_proposal(rays_o, rays_d)
            intersections_trunc = intersections[:, :-1]  # [batch, n_intrs - 1]
            batch, nintrs = intersections_trunc.shape
            intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intrs - 1, 3]
            # noinspection PyTypeChecker
            intrs_pts_mask = torch.all((-self.radius < intrs_pts) & (intrs_pts < self.radius), dim=-1)  # [batch, n_intrs-1]
            intrs_pts = intrs_pts[intrs_pts_mask]  # masked points
            # Normalize pts in [0, 1]
            intrs_pts = (intrs_pts + self.radius) / (self.radius * 2)

        data_interp = torch.ops.plenoxels.l2_interp_v2(
            grid, self.atoms, intrs_pts, self.fine_reso, self.coarse_reso,
            1 / self.fine_reso, 1)

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
        rgb = shrgb2rgb(rgb, abs_light, True)

        # 6. Depth map (optional)
        depth = depth_map(abs_light, intersections)

        return rgb, alpha, depth
