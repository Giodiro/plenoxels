from typing import Union, List, Optional, Tuple
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


def ensure_list(el, expand_size: Optional[int] = None) -> list:
    if isinstance(el, list):
        return el
    elif isinstance(el, tuple):
        return list(el)
    else:
        if expand_size:
            return [el] * expand_size
        return [el]


class DictPlenoxels(nn.Module):
    def __init__(self,
                 sh_deg: int,
                 sh_encoder,
                 fine_reso: Union[int, List[int]],
                 coarse_reso: int,
                 radius: Union[float, List[float]],
                 num_atoms: Union[int, List[int]],
                 num_scenes: int,
                 noise_std: float,
                 use_csrc: bool,
                 efficient_dict=True):
        super().__init__()
        assert not efficient_dict
        sh_dim = (sh_deg + 1) ** 2
        total_data_channels = sh_dim * 3 + 1
        self.radius: List[float] = ensure_list(radius, expand_size=num_scenes)
        self.fine_reso: List[int] = ensure_list(fine_reso)
        self.coarse_reso = coarse_reso
        self.num_atoms: List[int] = ensure_list(num_atoms)
        self.data_dim = total_data_channels
        self.efficient_dict = efficient_dict
        self.noise_std = noise_std
        self.sh_encoder = sh_encoder
        self.use_csrc = use_csrc
        self.num_scenes = num_scenes
        self.num_dicts = len(self.num_atoms)
        self.time = 0

        assert len(self.num_atoms) == len(self.fine_reso), "Number of atoms != number of fine-reso items"
        assert len(self.radius) == self.num_scenes, "Radii != number of scenes"
        assert sorted(self.fine_reso) == self.fine_reso, "Fine-reso elements must be sorted increasingly"

        self.step_size, self.n_intersections = self.calc_step_size()
        self.scalings, self.offsets = self.calc_scaling_offset()
        print("Ray-marching with step-size = %.4e  -  %d intersections" %
              (self.step_size, self.n_intersections))

        def get_reso(_in_reso: int) -> Tuple[int, ...]:
            if self.use_csrc:
                return (_in_reso * _in_reso * _in_reso, )
            else:
                return (_in_reso, _in_reso, _in_reso)
        self.atoms = nn.ParameterList()
        for reso, n_atoms in zip(self.fine_reso, self.num_atoms):
            self.atoms.append(nn.Parameter(torch.empty(*get_reso(reso), n_atoms, self.data_dim)))
        self.grids = nn.ModuleList()
        for scene in range(self.num_scenes):
            scene_grids = nn.ParameterList()
            for reso, n_atoms in zip(self.fine_reso, self.num_atoms):
                scene_grids.append(nn.Parameter(torch.empty(*get_reso(coarse_reso), n_atoms)))
            self.grids.append(scene_grids)

        self.init_params()

    def get_radius(self, dset_id: int) -> float:
        return self.radius[dset_id]

    def get_coarse_voxel_len(self, dset_id: int) -> float:
        return self.get_radius(dset_id) * 2 / self.coarse_reso

    def get_fine_voxel_len(self, dset_id: int, dict_id: int):
        return self.get_coarse_voxel_len(dset_id) / self.fine_reso[dict_id]

    def calc_step_size(self) -> Tuple[float, int]:
        # Smallest radius, largest fine-resolution
        smallest_dset = np.argmin(self.radius).item()
        smallest_voxel = self.get_fine_voxel_len(dset_id=smallest_dset, dict_id=self.num_dicts - 1)
        step_size = smallest_voxel / 2
        grid_diag = math.sqrt(3) * np.max(self.radius) * 2
        n_intersections = int(grid_diag / step_size) - 1
        return step_size, n_intersections

    def calc_scaling_offset(self) -> Tuple[List[float], List[float]]:
        scalings = [1 / (radius * 2) for radius in self.radius]
        offsets = [0.5 for _ in self.radius]
        return scalings, offsets

    def init_params(self):
        for scene_grids in self.grids:
            for grid in scene_grids:
                nn.init.normal_(grid, std=0.01)
                # nn.init.uniform_(grid, a=0.01, b=0.02)
        for atoms in self.atoms:
            nn.init.uniform_(atoms[..., :-1])
            nn.init.constant_(atoms[..., -1], 0.01)
            # # Compute the atom norms 
            # shape = atoms.shape # [..., n_atoms, data_dim]
            # dims = [i for i in range(len(shape))]
            # newdims = [dims[-2]] + dims[0:-2] + dims[-1:]
            # norms = torch.permute(atoms, newdims) # [n_atoms, ..., data_dim]
            # norms = torch.reshape(norms, (len(norms), -1))
            # norms = torch.linalg.vector_norm(norms, dim=-1)
            # # Clip so we don't divide by zero
            # norms = torch.clamp(norms, min=1e-2)
            # broadcastable_norms = torch.empty(size=[1]*(len(shape)-2) + [len(norms)] + [1])
            # broadcastable_norms[...,:,0] = norms
            # # Normalize each atom
            # atoms.data = torch.div(atoms, broadcastable_norms)


    @torch.no_grad()
    def sample_proposal(self, rays_o, rays_d, dset_id: int):
        dev, dt = rays_o.device, rays_o.dtype
        offsets_pos = (self.radius[dset_id] - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (-self.radius[dset_id] - rays_o) / rays_d  # [batch, 3]
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

    def normalize01(self, pts: torch.Tensor, dset_id: int) -> torch.Tensor:
        """Normalize from world coordinates to 0-1"""
        radius = self.get_radius(dset_id)
        return (pts + radius) / (radius * 2)

    def normalizegrid(self, pts: torch.Tensor, dset_id: int, dict_id) -> torch.Tensor:
        """Normalize from world coordinates to 0-gridsize"""
        radius = self.get_radius(dset_id)
        voxel_len = self.get_fine_voxel_len(dset_id=dset_id, dict_id=dict_id)
        return (pts + radius) / voxel_len

    def encode_patches(self, patches, dict_id: int, k=5):
        # Compute the inverse dictionary
        atoms = self.atoms[dict_id]
        atoms = atoms.permute(0,1,2,4,3).reshape(-1, self.num_atoms[dict_id])  # [patch_size, num_atoms]
        U, S, Vh = torch.linalg.svd(atoms, full_matrices=False)
        # Keep only the top k singular values
        filtered_S = torch.zeros_like(S)
        filtered_S[0:k] = 1.0 / S[0:k]  # Take the inverse
        pinv = torch.conj(Vh.T) @ torch.diag(filtered_S) @ torch.conj(U.T)
        # pinv = torch.linalg.pinv(atoms) # [num_atoms, patch_size]
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

    def forward(self, rays_o, rays_d, grid_id, consistency_coef=0, level=None, run_fp16=False, verbose=False):
        if level is None:
            level = len(self.fine_reso) - 1
        scene_grids = self.grids[grid_id]
        dtype = torch.float16 if run_fp16 else torch.float32
        if self.training:
            self.time += 1

        # result = None
        # for i, reso in enumerate(self.fine_reso[0: level + 1]):
        #     out = torch.ops.plenoxels.dict_tree_render(
        #         scene_grids[i].to(dtype=dtype), self.atoms[i].to(dtype=dtype), rays_o, rays_d,
        #         reso, self.coarse_reso, self.scalings[i], self.offsets[i], self.step_size,
        #         1e-6, 1e-6)
        #     if result is None:
        #         result = out
        #     else:
        #         result = result + out
        # return result, None, None, None

        with torch.autograd.no_grad():
            intersections = self.sample_proposal(rays_o, rays_d, grid_id)
            intersections_trunc = intersections[:, :-1]  # [batch, n_intrs - 1]
            batch, nintrs = intersections_trunc.shape
            intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intrs - 1, 3]
            # noinspection PyTypeChecker
            intrs_pts_mask = torch.all((-self.radius[grid_id] < intrs_pts) & (intrs_pts < self.radius[grid_id]), dim=-1)  # [batch, n_intrs-1]
            intrs_pts = intrs_pts[intrs_pts_mask]  # masked points

        consistency_loss = torch.tensor(0.0, device=rays_d.device)
        if not self.use_csrc:
            data_interp = torch.zeros(len(intrs_pts), self.data_dim, device=rays_o.device)
            consistency_loss = 0
            # Only work with the fine dicts up to the given level
            for i, reso in enumerate(self.fine_reso[0:level+1]):
                fine_offsets, fine_neighbors = self.get_neighbors(
                    self.normalizegrid(intrs_pts, dset_id=grid_id, dict_id=i), fine_reso=reso)  # [n_pts, 8, 3]
                # Get corresponding coarse grid indices for each fine neighbor
                coarse_neighbors = torch.div(fine_neighbors, reso, rounding_mode='floor')  # [n_pts, 8, 3]
                fine_neighbors = fine_neighbors % reso  # [n_pts, 8, 3]
                for n in range(8):
                    coarse_neighbor_vals = scene_grids[i][
                       coarse_neighbors[:,n,0], coarse_neighbors[:,n,1], coarse_neighbors[:,n,2], ...]  # [n_pts, n_atoms]
                    # Apply Gumbel softmax
                    # coarse_neighbor_vals = F.gumbel_softmax(coarse_neighbor_vals, tau=1. / np.log(1.0 + self.time), dim=-1)
                    # coarse_neighbor_vals = F.softmax(coarse_neighbor_vals, dim=-1)
                    # # Apply softmax passthrough trick
                    # y_soft = F.softmax(coarse_neighbor_vals, dim=-1)
                    # index = y_soft.topk(k=5, dim=-1)[1]
                    # y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)  # This will just pick the best single patch in the forward pass
                    # y_hard = F.softmax(y_hard * coarse_neighbor_vals, dim=-1) # softmax of just the top k entries
                    # coarse_neighbor_vals = y_hard - y_soft.detach() + y_soft  # And take gradients using softmax
                    # Apply the patches
                    fine_neighbor_vals = self.atoms[i][
                        fine_neighbors[:,n,0], fine_neighbors[:,n,1], fine_neighbors[:,n,2], ...]  # [n_pts, n_atoms, data_dim]
                    result = torch.sum(coarse_neighbor_vals[:,:,None] * fine_neighbor_vals, dim=1)  # [n_pts, data_dim]
                    if n == 0 and consistency_coef > 0:
                        consistency_loss = consistency_loss + self.patch_consistency_loss(
                            coarse_neighbor_vals[::10,:], dict_id=i)
                    weights = torch.prod(1. - fine_offsets[:, n, :], dim=-1, keepdim=True)  # [n_pts, 1]
                    data_interp = data_interp + weights * result
        else:
            # Normalize pts in [0, 1]
            intrs_pts = self.normalize01(intrs_pts, grid_id)
            # Only work with the fine dicts up to the given level
            data_interp = None
            for i, reso in enumerate(self.fine_reso[0:level+1]):
                out = torch.ops.plenoxels.dict_interpolate(
                    scene_grids[i].to(dtype=dtype), self.atoms[i].to(dtype=dtype), intrs_pts, reso,
                    self.coarse_reso)
                if data_interp is None:
                    data_interp = out
                else:
                    data_interp = data_interp + out

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

        return rgb, depth, alpha, consistency_loss

    def __repr__(self):
        return (f"DictPlenoxels(grids={self.grids}, num_atoms={self.num_atoms}, data_dim={self.data_dim}, "
                f"fine_reso={self.fine_reso}, coarse_reso={self.coarse_reso}, noise_std={self.noise_std})")
