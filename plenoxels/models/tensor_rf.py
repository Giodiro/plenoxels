import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.nerf_rendering import sigma2alpha, shrgb2rgb, depth_map


class TensorRf(nn.Module):
    def __init__(self, radius, resolution, sh_encoder, n_rgb_comp, n_sigma_comp, sh_deg, abs_light_thresh):
        super().__init__()
        self.radius = radius
        self.resolution = resolution
        self.step_ratio = 2.0
        self.sh_encoder = sh_encoder
        self.n_rgb_comp = n_rgb_comp
        self.n_density_comp = n_sigma_comp
        self.data_dim = ((sh_deg + 1) ** 2) * 3
        self.abs_light_thresh = abs_light_thresh

        self.density_plane = torch.nn.ParameterList([
            torch.nn.Parameter(0.1 * torch.randn((1, n_sigma_comp[0], self.resolution, self.resolution))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_sigma_comp[1], self.resolution, self.resolution))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_sigma_comp[2], self.resolution, self.resolution)))])
        self.density_line = torch.nn.ParameterList([
            torch.nn.Parameter(0.1 * torch.randn((1, n_sigma_comp[0], self.resolution, 1))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_sigma_comp[1], self.resolution, 1))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_sigma_comp[2], self.resolution, 1)))])
        self.rgb_plane = torch.nn.ParameterList([
            torch.nn.Parameter(0.1 * torch.randn((1, n_rgb_comp[0], self.resolution, self.resolution))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_rgb_comp[1], self.resolution, self.resolution))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_rgb_comp[2], self.resolution, self.resolution)))])
        self.rgb_line = torch.nn.ParameterList([
            torch.nn.Parameter(0.1 * torch.randn((1, n_rgb_comp[0], self.resolution, 1))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_rgb_comp[1], self.resolution, 1))),
            torch.nn.Parameter(0.1 * torch.randn((1, n_rgb_comp[2], self.resolution, 1)))])
        self.basis_mat = torch.nn.Linear(self.n_rgb_comp * 3, self.data_dim, bias=False)
        self.step_size = 0
        self.n_samples = 0
        self.update_stepsize()

    @torch.autograd.no_grad()
    def sample_proposal(self, rays_o, rays_d):
        """Generate intersection points between a batch of rays and the volume.
        :param rays_o: B, 3
        :param rays_d: B, 3
        :return:
            intrs_pts:      B, N, 3
            intrs_offsets:  B, N
            mask_inbox:     B, N, 3
        """
        stepsize = self.step_size
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.radius - rays_o) / vec
        rate_b = (-self.radius - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1)

        rng = torch.arange(self.n_samples)[None].float()
        if self.is_training:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # noinspection PyUnresolvedReferences
        mask = ((-self.radius <= intrs_pts) & (intrs_pts <= self.radius)).all(dim=-1)
        return intrs_pts, interpx, mask

    def update_stepsize(self):
        diameter = self.radius * 2
        units = diameter / (self.resolution - 1)
        self.step_size = units * self.step_ratio
        grid_diag = math.sqrt(3 * (diameter ** 2))
        self.n_samples = int(grid_diag / self.step_size) + 1

    def normalize_coord(self, xyz_sampled):
        """Normalize world coordinates to -1, +1"""
        return (xyz_sampled + self.radius) * (1 / (self.radius * 2)) - 1

    def forward(self, rays_o, rays_d):
        # sample points
        intrs_pts, intersections, intrs_pts_mask = self.sample_proposal(rays_o, rays_d)
        batch = intersections.shape[0]
        nintrs = intersections.shape[1] - 1

        sigma = torch.zeros(intrs_pts.shape[:-1], device=intrs_pts.device)
        rgb = torch.zeros((*intrs_pts.shape[:2], 3), device=intrs_pts.device)

        if intrs_pts_mask.any():
            intrs_pts = self.normalize_coord(intrs_pts)
            sigma = self.compute_density_feature(intrs_pts[intrs_pts_mask])
            sigma = F.relu(sigma)
            sigma[intrs_pts_mask] = sigma

        alpha, weight = sigma2alpha(sigma, intersections, rays_d)
        rgb_mask = weight > self.abs_light_thresh

        if rgb_mask.any():
            rgb_features = self.compute_rgb_feature(intrs_pts[rgb_mask])
            # 3. Create SH coefficients and mask them
            sh_mult = self.sh_encoder(rays_d).unsqueeze(1).expand(batch, nintrs, -1)  # [batch, nintrs, ch/3]
            sh_mult = sh_mult[intrs_pts_mask].unsqueeze(1)  # [mask_pts, 1, ch/3]
            # 4. Interpolate rgbdata, use SH coefficients to get to RGB
            sh_masked = rgb_features.view(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
            rgb_masked = torch.sum(sh_mult * sh_masked, dim=-1)   # [mask_pts, 3]
            rgb[rgb_mask] = rgb_masked
        rgb = shrgb2rgb(rgb, weight, True)
        depth = depth_map(weight, intersections)
        return rgb, depth

    def get_coordinate_plane(self, intrs_pts):
        """intrs_pts: B*N, 3"""
        # coordinate_plane: 3, B, N, 2 -> 3, B*N, 1, 2
        coordinate_plane = torch.stack((intrs_pts[..., [0, 1]], intrs_pts[..., [0, 2]], intrs_pts[..., [1, 2]])).detach().view(3, -1, 1, 2)
        return coordinate_plane

    def get_coordinate_line(self, intrs_pts):
        """intrs_pts: B*N, 3"""
        # coordinate_line: 3, B, N -> 3, B, N, 2 (where the first part is zeros) -> 3, B*N, 1, 2
        coordinate_line = torch.stack((intrs_pts[..., 2], intrs_pts[..., 1], intrs_pts[..., 0]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        return coordinate_line

    def compute_density_feature(self, intrs_pts):
        """intrs_pts: B*N, 3"""
        batch_size = intrs_pts.shape[0]
        coordinate_plane = self.get_coordinate_plane(intrs_pts)
        coordinate_line = self.get_coordinate_line(intrs_pts)

        density_feature = torch.zeros((intrs_pts.shape[0],), device=intrs_pts.device)
        for plane_idx in range(len(self.density_plane)):
            # input: 1, C, R, R - grid: 1, B*N, 1, 2 -> 1, C, B*N, 1 -> C, B*N
            plane_coef_point = F.grid_sample(
                self.density_plane[plane_idx], coordinate_plane[[plane_idx]], align_corners=True
            ).view(-1, batch_size)  # C, B*N
            line_coef_point = F.grid_sample(
                self.density_line[plane_idx], coordinate_line[[plane_idx]], align_corners=True
            ).view(-1, batch_size)
            density_feature = density_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
        return density_feature  # B*N

    def compute_rgb_feature(self, intrs_pts):
        """intrs_pts: B*N, 3"""
        batch_size = intrs_pts.shape[0]
        coordinate_plane = self.get_coordinate_plane(intrs_pts)
        coordinate_line = self.get_coordinate_line(intrs_pts)

        plane_coef_point, line_coef_point = [], []
        for plane_idx in range(len(self.rgb_plane)):
            # input: 1, C, R, R - grid: 1, B*N, 1, 2 -> 1, C, B*N, 1 -> C, B*N
            plane_coef_point.append(F.grid_sample(
                self.rgb_plane[plane_idx], coordinate_plane[[plane_idx]], align_corners=True
            ).view(-1, batch_size))  # C, B*N
            line_coef_point.append(F.grid_sample(
                self.rgb_line[plane_idx], coordinate_line[[plane_idx]], align_corners=True
            ).view(-1, batch_size))
        # C, B*N -> 3*C, B*N
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        # 3*C, B*N -> B*N, OutDim
        return self.basis_mat((plane_coef_point * line_coef_point).T)  # B*N, OutDim

    @torch.autograd.no_grad()
    def upsample_single(self, plane_coef: torch.nn.ParameterList, line_coef: torch.nn.ParameterList, target_reso: int):
        for i in range(3):
            plane_coef[i] = torch.nn.Parameter(F.interpolate(
                plane_coef[i].data, size=(target_reso, target_reso), mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(
                line_coef[i].data, size=(target_reso, 1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    @torch.autograd.no_grad()
    def upsample_volume_grid(self, new_resolution: int):
        self.density_plane, self.density_line = self.upsample_single(
            self.density_plane, self.density_line, target_reso=new_resolution)
        self.rgb_plane, self.rgb_line = self.upsample_single(
            self.rgb_plane, self.rgb_line, target_reso=new_resolution)

        self.resolution = new_resolution
        self.update_stepsize()
        print(f"Upsampled model to {self.resolution}. "
              f"New step-size: {self.step_size} - n_samples: {self.n_samples}")
