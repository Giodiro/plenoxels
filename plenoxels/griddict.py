from typing import Tuple, List
from importlib.machinery import PathFinder
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tc_plenoxel import shrgb2rgb, depth_map, sigma2alpha



class ShDictRender(nn.Module):
    def __init__(self,
                 sh_deg: int,
                 sh_encoder,
                 fine_reso: int,
                 coarse_reso: int,
                 radius: float,
                 num_atoms: int,
                 init_sigma: float,
                 init_rgb: float,
                 num_scenes: int,
                 noise_std: float,
                 efficient_dict=True):
        super().__init__()
        sh_dim = (sh_deg + 1) ** 2
        total_data_channels = sh_dim * 3 + 1
        self.radius = radius
        self.fine_reso = fine_reso
        self.coarse_reso = coarse_reso
        self.num_atoms = num_atoms
        self.data_dim = total_data_channels
        self.noise_std = noise_std
        self.efficient_dict = efficient_dict
        # If we want to reuse the same dictionary for different channels
        if efficient_dict:
            self.grids = nn.ParameterList([nn.Parameter(torch.empty(coarse_reso, coarse_reso, coarse_reso, self.num_atoms, self.data_dim)) for i in range(num_scenes)])
            self.atoms = nn.Parameter(torch.empty(self.fine_reso, self.fine_reso, self.fine_reso, num_atoms, self.data_dim, dtype=torch.float32)) 
        else:
            self.grids = nn.ParameterList([nn.Parameter(torch.empty(coarse_reso, coarse_reso, coarse_reso, self.num_atoms)) for i in range(num_scenes)])
            self.atoms = nn.Parameter(torch.empty(self.fine_reso, self.fine_reso, self.fine_reso, self.num_atoms, self.data_dim, dtype=torch.float32)) 
        self.n_intersections = coarse_reso * 3 * 2 * fine_reso
        with torch.no_grad():
            nn.init.uniform_(self.atoms[..., :-1])
            self.atoms[..., -1].fill_(init_sigma)
        for grid in self.grids:
            nn.init.normal_(grid, std=0.01)
        self.sh_encoder = sh_encoder
        self.coarse_voxel_len = radius * 2 / coarse_reso
        self.fine_voxel_len = self.coarse_voxel_len / fine_reso
        self.step_size = self.fine_voxel_len / 2.

    def __repr__(self):
        return (f"ShDictRender(grids={self.grids}, num_atoms={self.num_atoms}, data_dim={self.data_dim}, "
                f"fine_reso={self.fine_reso})")

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

    def get_neighbors(self, pts):
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
        pre_floor = pts[:,None,:] + offsets_3d[None,...] / 2.
        post_floor = torch.clamp(torch.floor(pre_floor), min=0., max=self.coarse_reso * self.fine_reso - 1)
        return torch.abs(pts[:,None,:] - (post_floor + 0.5)), post_floor.long()

    def encode_patches(self, patches):
        # Compute the inverse dictionary
        if self.efficient_dict:
            atoms = self.atoms.view(-1, self.num_atoms)  # [patch_size, num_atoms]
        else:
            atoms = self.atoms.permute(0,1,2,4,3).reshape(-1, self.num_atoms)  # [patch_size, num_atoms]
        pinv = torch.linalg.pinv(atoms) # [num_atoms, patch_size]
        # Apply to the patches
        vectorized_patches = patches.view(patches.size(0), -1) # [batch_size, patch_size]
        return vectorized_patches @ pinv.T  # [batch_size, num_atoms]

    def patch_consistency_loss(self, weights):
        # Convert weights to patches
        # weights has shape [batch_size, num_atoms]
        if self.efficient_dict:
            atoms = self.atoms.view(-1, self.num_atoms)  # [patch_size, num_atoms]
        else:
            atoms = self.atoms.permute(0,1,2,4,3).reshape(-1, self.num_atoms)  # [patch_size, num_atoms]
        patches = weights @ atoms.T # [batch_size, patch_size]
        # Add noise to patches
        with torch.autograd.no_grad():
            noise = torch.randn_like(patches) * self.noise_std * atoms.abs().max()
        patches = patches + noise
        # Encode patches
        new_weights = self.encode_patches(patches)
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
            # Get indices of 8 fine grid neighbors
            fine_offsets, fine_neighbors = self.get_neighbors((intrs_pts + self.radius) / self.fine_voxel_len)  # [n_pts, 8, 3]
            # Get corresponding coarse grid indices for each fine neighbor
            coarse_neighbors = torch.div(fine_neighbors, self.fine_reso, rounding_mode='floor')  # [n_pts, 8, 3]
            fine_neighbors = fine_neighbors % self.fine_reso  # [n_pts, 8, 3]
            # Apply the dictionary to each of the 8 neighbors
        # This should be faster but goes OOM
        # coarse_neighbors = coarse_neighbors.view(-1,3)
        # fine_neighbors = fine_neighbors.view(-1,3)
        # fine_offsets = fine_offsets.view(-1,3)
        # coarse_neighbor_vals = grid[coarse_neighbors[:,0], coarse_neighbors[:,1], coarse_neighbors[:,2], :]  # [n_pts*8, n_atoms]
        # fine_neighbor_vals = self.atoms[fine_neighbors[:,0], fine_neighbors[:,1], fine_neighbors[:,2], :, :]  # [n_pts*8, n_atoms, data_dim]
        # result = torch.sum(coarse_neighbor_vals[:,:,None] * fine_neighbor_vals, dim=1) # [n_pts*8, data_dim]
        # weights = torch.prod(1. - fine_offsets, dim=-1, keepdim=True)  # [n_pts*8, 1]
        # data_interp = torch.sum(weights.view(-1,8,1) * result.view(-1,8,self.data_dim), dim=1)
        # Slower, but cheaper on memory
        data_interp = torch.zeros(len(fine_neighbors), self.data_dim, device=rays_o.device)
        for i in range(8):
            if self.efficient_dict:
                coarse_neighbor_vals = grid[coarse_neighbors[:,i,0], coarse_neighbors[:,i,1], coarse_neighbors[:,i,2], :, :]  # [n_pts, n_atoms, data_dim]
                fine_neighbor_vals = self.atoms[fine_neighbors[:,i,0], fine_neighbors[:,i,1], fine_neighbors[:,i,2], :, :]  # [n_pts, n_atoms, data_dim]
                result = torch.sum(coarse_neighbor_vals * fine_neighbor_vals, dim=1) # [n_pts, data_dim]
                consistency_loss = torch.tensor(0.0, device=result.device) # TODO: fill this in
            else:
                coarse_neighbor_vals = grid[coarse_neighbors[:,i,0], coarse_neighbors[:,i,1], coarse_neighbors[:,i,2], :]  # [n_pts, n_atoms]
                fine_neighbor_vals = self.atoms[fine_neighbors[:,i,0], fine_neighbors[:,i,1], fine_neighbors[:,i,2], :, :]  # [n_pts, n_atoms, data_dim]
                result = torch.sum(coarse_neighbor_vals[:,:,None] * fine_neighbor_vals, dim=1) # [n_pts, data_dim]
                if i == 0:
                    consistency_loss = self.patch_consistency_loss(coarse_neighbor_vals[::10,:])
            weights = torch.prod(1. - fine_offsets[:,i,:], dim=-1, keepdim=True)  # [n_pts, 1]
            data_interp = data_interp + weights * result

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

        return rgb, alpha, depth, consistency_loss
