from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.tc_plenoxel import shrgb2rgb, depth_map, sigma2alpha, TrilinearInterpolate


def interp_regular(grid, pts):
    """Interpolate data on a regular grid at the given points.

    :param grid:
        Tensor of size [1, ch, res, res, res].
    :param pts:
        Tensor of size [1, n, 1, 1, 3]. Points must be normalized between -1 and 1.
    :return:
        Tensor of size [ch, n] or [n] if the `ch` dimension is of size 1.
    """
    pts = pts.to(dtype=grid.dtype)
    interp_data = F.grid_sample(
        grid, pts, mode='bilinear', align_corners=True)  # [1, ch, n, 1, 1]
    interp_data = interp_data.squeeze()  # [ch, n] or [n] if ch is 1
    return interp_data


@torch.jit.script
def normalize_coord(intersections: torch.Tensor,
                    aabb: torch.Tensor,
                    inverse_aabb_size: torch.Tensor) -> torch.Tensor:
    """Returns coordinates normalized between -1 and +1"""
    return (intersections - aabb[0]) * inverse_aabb_size - 1


class RegularGrid(nn.Module):
    def __init__(self,
                 resolution: torch.Tensor,
                 aabb: torch.Tensor,
                 data_dim: int,
                 near_far: Tuple[float]):
        super().__init__()
        self.register_buffer("resolution", resolution.float())
        self.register_buffer("aabb", aabb)
        self.near_far = near_far
        self.data_dim = data_dim

        self.data = nn.Parameter(torch.empty(
            1, data_dim, int(resolution[0]), int(resolution[1]), int(resolution[2]), dtype=torch.float32))
        nn.init.normal_(self.data, std=0.1)

    @property
    def n_intersections(self):
        return int(torch.mean(self.resolution * 3 * 2).item())

    @property
    def inv_aabb_size(self):
        aabb_size = self.aabb[1] - self.aabb[0]
        return 2 / aabb_size

    @property
    def voxel_len(self):
        aabb_size = self.aabb[1] - self.aabb[0]
        aabb_diag = torch.linalg.norm(aabb_size)
        return aabb_diag / self.n_intersections

    @torch.no_grad()
    def sample_proposal(self, rays_o, rays_d):
        dev, dt = rays_o.device, rays_o.dtype
        offsets_pos = (self.aabb[1] - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (self.aabb[0] - rays_o) / rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        start.clamp_(min=self.near_far[0], max=self.near_far[1])  # [batch, 1]

        steps = torch.arange(self.n_intersections, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
        steps = steps.repeat(rays_d.shape[0], 1)   # [batch, n_intrs]
        intersections = start + steps * self.voxel_len  # [batch, n_intrs]
        return intersections

    def forward(self, rays_o, rays_d):
        intersections = self.sample_proposal(rays_o, rays_d)
        intersections_trunc = intersections[:, :-1]  # [batch, n_intrs - 1]
        intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intrs - 1, 3]
        # noinspection PyTypeChecker
        intrs_pts_mask = torch.all((self.aabb[0] < intrs_pts) & (intrs_pts < self.aabb[1]), dim=-1)  # [batch, n_intrs-1]

        intrs_pts = intrs_pts[intrs_pts_mask]  # masked points
        intrs_pts = normalize_coord(intrs_pts, self.aabb, self.inv_aabb_size)
        data_interp = interp_regular(
            self.data, intrs_pts.view(1, -1, 1, 1, 3)).T  # [mask_pts, ch]
        return data_interp, intrs_pts_mask, intersections, intrs_pts


class ShDictRender(nn.Module):
    def __init__(self,
                 sh_deg: int,
                 sh_encoder,
                 grid: nn.Module,
                 init_sigma: float,
                 init_rgb: float,
                 white_bkgd: bool,
                 abs_light_thresh: float,
                 occupancy_thresh: float,):
        super().__init__()
        sh_dim = (sh_deg + 1) ** 2
        total_data_channels = sh_dim * 3 + 1

        self.grid = grid
        self.num_atoms = grid.data_dim
        self.data_dim = total_data_channels
        self.white_bkgd = white_bkgd
        self.abs_light_thresh = abs_light_thresh  # TODO: Unused
        self.occupancy_thresh = occupancy_thresh  # TODO: Unused

        self.atoms = nn.Parameter(torch.empty(self.num_atoms, self.data_dim, 8, dtype=torch.float32))
        with torch.no_grad():
            self.atoms[:, :-1].fill_(init_rgb)
            self.atoms[:, -1].fill_(init_sigma)
        self.sh_encoder = sh_encoder

    def forward(self, rays_o, rays_d):
        # queries: [n_pts, n_atoms]  queries_mask: [batch, n_intrs - 1]  intersections: [batch, n_intrs]
        queries, queries_mask, intersections, intrs_pts = self.grid(rays_o, rays_d)
        batch, nintrs = queries_mask.size()
        n_pts = queries.shape[0]

        # [n_pts, n_atoms] @ [n_atoms, data_dim, 8] => [n_pts, data_dim, 8]
        data_masked = (queries @ self.atoms.view(self.num_atoms, -1)).view(n_pts, *self.atoms.shape[1:])
        # Interpolate atoms.
        with torch.autograd.no_grad():
            # intrs_pts are between -1, 1
            pts = intrs_pts * (self.grid.resolution / 2) + 1e-5
            xyzs = pts - torch.floor(pts)
        data_interp = TrilinearInterpolate.apply(data_masked, xyzs)  # [n_pts, data_dim]

        # 1. Process density: Un-masked sigma (batch, n_intrs-1), and compute.
        sigma_masked = data_interp[:, -1]
        sigma = torch.zeros(batch, nintrs, dtype=sigma_masked.dtype, device=sigma_masked.device)
        sigma.masked_scatter_(queries_mask, sigma_masked)
        sigma = F.relu(sigma)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)  # both [batch, n_intrs-1]

        # 3. Create SH coefficients and mask them
        sh_mult = self.sh_encoder(rays_d).unsqueeze(1).expand(batch, nintrs, -1)  # [batch, nintrs, ch/3]
        sh_mult = sh_mult[queries_mask].unsqueeze(1)  # [mask_pts, 1, ch/3]

        # 4. Interpolate rgbdata, use SH coefficients to get to RGB
        sh_masked = data_interp[:, :-1]
        sh_masked = sh_masked.view(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
        rgb_masked = torch.sum(sh_mult * sh_masked, dim=-1)   # [mask_pts, 3]

        # 5. Post-process RGB
        rgb = torch.zeros(batch, nintrs, 3, dtype=rgb_masked.dtype, device=rgb_masked.device)
        rgb.masked_scatter_(queries_mask.unsqueeze(-1), rgb_masked)
        rgb = shrgb2rgb(rgb, abs_light, self.white_bkgd)

        # 6. Depth map (optional)
        depth = depth_map(abs_light, intersections)

        return rgb, alpha, depth
