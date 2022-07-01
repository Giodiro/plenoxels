from typing import Union, List, Tuple
from importlib.machinery import PathFinder
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.models.utils import interp_regular, ensure_list, sample_proposal
from plenoxels.nerf_rendering import shrgb2rgb, depth_map, sigma2alpha

try:
    spec = PathFinder().find_spec("c_ext", [str(Path(__file__).resolve().parents[1])])
    torch.ops.load_library(spec.origin)
except:
    print("Failed to load C-extension necessary for DictPlenoxels model")


class SingleResoDictPlenoxels(nn.Module):
    def __init__(self,
                 sh_deg: int,
                 sh_encoder,
                 fine_reso: int,
                 coarse_reso: int,
                 radius: Union[float, List[float]],
                 num_atoms: int,
                 num_scenes: int,
                 use_csrc: bool,
                 dict_only_sigma: bool):
        super().__init__()
        sh_dim = (sh_deg + 1) ** 2
        total_data_channels = sh_dim * 3 + 1
        self.radius: List[float] = ensure_list(radius, expand_size=num_scenes)
        self.fine_reso = fine_reso
        self.coarse_reso = coarse_reso
        self.num_atoms = num_atoms
        self.data_dim = total_data_channels
        self.sh_encoder = sh_encoder
        self.use_csrc = use_csrc
        self.num_scenes = num_scenes
        self.dict_only_sigma = dict_only_sigma

        assert len(self.radius) == self.num_scenes, "Radii != number of scenes"

        self.step_size, self.n_intersections = self.calc_step_size()
        self.scalings, self.offsets = self.calc_scaling_offset()
        print("Ray-marching with step-size = %.4e  -  %d intersections" %
              (self.step_size, self.n_intersections))

        def get_reso(_in_reso: int) -> Tuple[int, ...]:
            if self.use_csrc:
                return (_in_reso * _in_reso * _in_reso, )
            else:
                return (_in_reso, _in_reso, _in_reso)

        if self.dict_only_sigma:
            self.dict_data_dim = 1
        else:
            self.dict_data_dim = self.data_dim
        self.atoms = nn.Parameter(
            torch.empty(*get_reso(self.fine_reso), self.num_atoms, self.dict_data_dim))

        self.cgrids = nn.ParameterList()
        for scene in range(self.num_scenes):
            self.cgrids.append(nn.Parameter(
                torch.empty(*get_reso(self.coarse_reso), self.num_atoms)))
        # Initialize full grids (resolution coarse * fine) just for SH in case dictionary is just
        # for sigma.
        self.grids = None
        if self.dict_only_sigma:
            self.grids = nn.ParameterList()
            for scene in range(self.num_scenes):
                self.grids.append(nn.Parameter(
                    torch.empty(self.data_dim - 1, *get_reso(self.coarse_reso * self.fine_reso))))

        self.init_params()

    def init_params(self):
        for grid in self.cgrids:
            nn.init.normal_(grid, std=0.01)
        if self.grids is not None:
            for grid in self.grids:
                nn.init.constant_(grid, 0.1)
        nn.init.constant_(self.atoms[..., :-1], 0.1)
        nn.init.constant_(self.atoms[..., -1], 0.01)

    def get_radius(self, dset_id: int) -> float:
        return self.radius[dset_id]

    def get_coarse_voxel_len(self, dset_id: int) -> float:
        return self.get_radius(dset_id) * 2 / self.coarse_reso

    def get_fine_voxel_len(self, dset_id: int):
        return self.get_coarse_voxel_len(dset_id) / self.fine_reso

    def calc_step_size(self) -> Tuple[float, int]:
        # Smallest radius, largest fine-resolution
        smallest_radius = np.min(self.radius)
        resolution = self.coarse_reso * self.fine_reso
        units = (smallest_radius * 2) / (resolution - 1)
        step_size = units / 2
        grid_diag = math.sqrt(3) * np.max(self.radius) * 2
        n_intersections = int(grid_diag / step_size) - 1
        return step_size, n_intersections

    def calc_scaling_offset(self) -> Tuple[List[float], List[float]]:
        scalings = [1 / (radius * 2) for radius in self.radius]
        offsets = [0.5 for _ in self.radius]
        return scalings, offsets

    def normalize01(self, pts: torch.Tensor, dset_id: int) -> torch.Tensor:
        """Normalize from world coordinates to 0-1"""
        radius = self.get_radius(dset_id)
        return (pts + radius) / (radius * 2)

    def forward(self, rays_o, rays_d, grid_id, consistency_coef=0, level=None, run_fp16=False, verbose=False):
        intrs_pts, intersections, intrs_pts_mask = sample_proposal(
            rays_o, rays_d, self.radius[grid_id], self.n_intersections, self.step_size)
        batch = intersections.shape[0]
        nintrs = intersections.shape[1] - 1
        intrs_pts = intrs_pts[intrs_pts_mask]

        # Normalize pts in [0, 1]
        intrs_pts = self.normalize01(intrs_pts, grid_id)
        if self.dict_only_sigma:
            tot_reso = self.coarse_reso * self.fine_reso
            grid = self.grids[grid_id].view(1, -1, tot_reso, tot_reso, tot_reso)
            pts = (intrs_pts * 2 - 1).view(1, -1, 1, 1, 3)
            sh_interp = interp_regular(grid, pts).T

        dict_interp = torch.ops.plenoxels.dict_interpolate(
            self.cgrid[grid_id], self.atoms, intrs_pts, self.fine_reso, self.coarse_reso)

        # 1. Process density: Un-masked sigma (batch, n_intrs-1), and compute.
        sigma_masked = dict_interp[:, -1]
        sigma_masked = F.relu(sigma_masked)
        sigma = torch.zeros(batch, nintrs, dtype=sigma_masked.dtype, device=sigma_masked.device)
        sigma.masked_scatter_(intrs_pts_mask, sigma_masked)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)  # both [batch, n_intrs-1]

        # 3. Create SH coefficients and mask them
        sh_mult = self.sh_encoder(rays_d).unsqueeze(1).expand(batch, nintrs, -1)  # [batch, nintrs, ch/3]
        sh_mult = sh_mult[intrs_pts_mask].unsqueeze(1)  # [mask_pts, 1, ch/3]

        # 4. Interpolate rgbdata, use SH coefficients to get to RGB
        if self.dict_only_sigma:
            sh_masked = sh_interp  # noqa
        else:
            sh_masked = dict_interp[:, :-1]
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
        return (f"SingleResoDictPlenoxels(grids={self.grids}, num_atoms={self.num_atoms}, "
                f"data_dim={self.data_dim}, fine_reso={self.fine_reso}, "
                f"coarse_reso={self.coarse_reso})")
