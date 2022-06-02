from typing import Union, List
from importlib.machinery import PathFinder
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.nerf_rendering import shrgb2rgb, depth_map, sigma2alpha

try:
    spec = PathFinder().find_spec("c_ext", [str(Path(__file__).resolve().parents[1])])
    torch.ops.load_library(spec.origin)
except:
    print("Failed to load C-extension necessary for DictPlenoxels model")


class DictPlenoxels(nn.Module):
    def __init__(self,
                 sh_deg: int,
                 sh_encoder,
                 fine_reso: Union[int, List[int]],
                 coarse_reso: int,
                 radius: float,
                 num_atoms: Union[int, List[int]],
                 num_scenes: int,
                 noise_std: float,
                 use_csrc: bool,
                 efficient_dict=True):
        super().__init__()
        assert not efficient_dict
        sh_dim = (sh_deg + 1) ** 2
        total_data_channels = sh_dim * 3 + 1
        self.radius = radius
        if isinstance(fine_reso, int):
            fine_reso = [fine_reso]
        self.fine_reso: List[int] = fine_reso
        self.coarse_reso = coarse_reso
        if isinstance(num_atoms, int):
            num_atoms = [num_atoms]
        self.num_atoms: List[int] = num_atoms
        self.data_dim = total_data_channels
        self.efficient_dict = efficient_dict
        self.noise_std = noise_std
        self.sh_encoder = sh_encoder
        self.coarse_voxel_len = radius * 2 / coarse_reso
        self.fine_voxel_len = [self.coarse_voxel_len / reso for reso in self.fine_reso]
        self.step_size = self.fine_voxel_len[-1] / 2.
        self.n_intersections = coarse_reso * 3 * 2 * fine_reso[-1]
        self.use_csrc = use_csrc
        assert len(self.num_atoms) == len(self.fine_reso), "Number of atoms != number of fine-reso items"

        self.grids = nn.ParameterList([nn.Parameter(
            torch.empty(coarse_reso ** 3, sum(num_atoms))) for _ in range(num_scenes)])
        atoms = []
        for reso, n_atoms in zip(self.fine_reso, self.num_atoms):
            atoms.append(nn.Parameter(torch.empty(reso ** 3, n_atoms, self.data_dim)))
        self.atoms = nn.ParameterList(atoms)
        self.init_params()

    def init_params(self):
        for grid in self.grids:
            nn.init.normal_(grid, std=0.01)
        for atoms in self.atoms:
            nn.init.uniform_(atoms[..., :-1])
            nn.init.constant_(atoms[..., -1], 0.01)

    def __repr__(self):
        return (f"DictPlenoxels(grids={self.grids}, num_atoms={self.num_atoms}, data_dim={self.data_dim}, "
                f"fine_reso={self.fine_reso}, coarse_reso={self.coarse_reso}, noise_std={self.noise_std})")

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

    def tv_loss(self, grid_id):
        grid = self.grids[grid_id]
        grid = grid.view(self.coarse_reso, self.coarse_reso, self.coarse_reso, sum(self.num_atoms))

        pixel_dif1 = grid[1:, :, :, ...] - grid[:-1, :, :, ...]
        pixel_dif2 = grid[:, 1:, :, ...] - grid[:, :-1, :, ...]
        pixel_dif3 = grid[:, :, 1:, ...] - grid[:, :, :-1, ...]

        res1 = pixel_dif1.square().sum()
        res2 = pixel_dif2.square().sum()
        res3 = pixel_dif3.square().sum()

        return (res1 + res2 + res3) / np.prod(grid.shape)

    def get_neighbors(self, pts, fine_reso):
        # pts should be in grid coordinates, ranging from 0 to coarse_reso * fine_reso
        offsets_3d = torch.tensor(
            [[-1, -1, -1],
             [-1, -1, 1],
             [-1, 1, -1],
             [-1, 1, 1],
             [1, -1, -1],
             [1, -1, 1],
             [1, 1, -1],
             [1, 1, 1]], dtype=pts.dtype, device=pts.device)
        pre_floor = pts[:, None, :] + offsets_3d[None, ...] / 2.
        post_floor = torch.clamp(torch.floor(pre_floor), min=0., max=self.coarse_reso * fine_reso - 1)
        return torch.abs(pts[:,None,:] - (post_floor + 0.5)), post_floor.long()

    def encode_patches(self, patches, dict_id: int):
        # Compute the inverse dictionary
        atoms = self.atoms[dict_id]
        atoms = atoms.permute(0,1,2,4,3).reshape(-1, self.num_atoms[dict_id])  # [patch_size, num_atoms]
        pinv = torch.linalg.pinv(atoms) # [num_atoms, patch_size]
        # Apply to the patches
        vectorized_patches = patches.view(patches.size(0), -1) # [batch_size, patch_size]
        return vectorized_patches @ pinv.T  # [batch_size, num_atoms]

    def patch_consistency_loss(self, weights, dict_id: int):
        # Convert weights to patches
        # weights has shape [batch_size, num_atoms]
        atoms = self.atoms[dict_id]
        atoms = atoms.permute(0,1,2,4,3).reshape(-1, self.num_atoms[dict_id])  # [patch_size, num_atoms]
        patches = weights @ atoms.T # [batch_size, patch_size]
        # Add noise to patches
        with torch.autograd.no_grad():
            noise = torch.randn_like(patches) * self.noise_std * atoms.abs().max()
        patches = patches + noise
        # Encode patches
        new_weights = self.encode_patches(patches, dict_id)
        # Compute loss
        return F.mse_loss(new_weights, weights)

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

        consistency_loss = torch.tensor(0.0, device=rays_d.device)
        if not self.use_csrc:
            atom_idx = 0
            data_interp = torch.zeros(len(intrs_pts), self.data_dim, device=rays_o.device)
            consistency_loss = 0
            for i, (reso, n_atoms) in enumerate(zip(self.fine_reso, self.num_atoms)):
                fine_offsets, fine_neighbors = self.get_neighbors(
                    (intrs_pts + self.radius) / self.fine_voxel_len[i], fine_reso=reso)  # [n_pts, 8, 3]
                # Get corresponding coarse grid indices for each fine neighbor
                coarse_neighbors = torch.div(fine_neighbors, reso, rounding_mode='floor')  # [n_pts, 8, 3]
                fine_neighbors = fine_neighbors % reso  # [n_pts, 8, 3]
                for n in range(8):
                    coarse_neighbor_vals = grid[
                       coarse_neighbors[:,n,0], coarse_neighbors[:,n,1], coarse_neighbors[:,n,2],
                       atom_idx: atom_idx + n_atoms]  # [n_pts, n_atoms]
                    fine_neighbor_vals = self.atoms[i][
                        fine_neighbors[:,n,0], fine_neighbors[:,n,1], fine_neighbors[:,n,2], ...]  # [n_pts, n_atoms, data_dim]
                    result = torch.sum(coarse_neighbor_vals[:,:,None] * fine_neighbor_vals, dim=1)  # [n_pts, data_dim]
                    if n == 0 and self.noise_std > 0:
                        consistency_loss = consistency_loss + self.patch_consistency_loss(
                            coarse_neighbor_vals[::10,:], dict_id=i)
                    weights = torch.prod(1. - fine_offsets[:, n, :], dim=-1, keepdim=True)  # [n_pts, 1]
                    data_interp = data_interp + weights * result
                atom_idx += n_atoms
        else:
            # Normalize pts in [0, 1]
            intrs_pts = (intrs_pts + self.radius) / (self.radius * 2)
            atom_idx = 0
            for i, (reso, n_atoms) in enumerate(zip(self.fine_reso, self.num_atoms)):
                if i == 0:
                    data_interp = torch.ops.plenoxels.l2_interp_v2(
                            grid[..., atom_idx: atom_idx + n_atoms], self.atoms[i], intrs_pts, reso, self.coarse_reso, 1, 1)
                else:
                    data_interp = data_interp + torch.ops.plenoxels.l2_interp_v2(
                            grid[..., atom_idx: atom_idx + n_atoms], self.atoms[i], intrs_pts, reso, self.coarse_reso, 1, 1)
                atom_idx += n_atoms

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
        depth = depth_map(abs_light, intersections)

        return rgb, alpha, depth, consistency_loss
