import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from plenoxels.models.utils import get_intersections, grid_sample_wrapper
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha


class SHRender(nn.Module):
    def __init__(self, sh_degree: int):
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": sh_degree,
            },
        )
        super().__init__()

    def compute_density(self, density_features, mask, rays_d):
        dim_batch, dim_nintrs = mask.shape
        sigma = torch.zeros(dim_batch, dim_nintrs, device=density_features.device)

        if mask.any():
            sigma_valid = F.relu(density_features)
            sigma[mask] = sigma_valid
        return sigma

    def compute_color(self, color_features, mask, rays_d):
        dim_batch, dim_nintrs = mask.shape

        rgb = torch.zeros((dim_batch, dim_nintrs, 3), device=color_features.device)

        if mask.any():
            # rgb_features = self.compute_rgb_feature(intrs_pts[rgb_mask])
            # 3. Create SH coefficients and mask them
            sh_mult = self.direction_encoder(rays_d).unsqueeze(1).expand(dim_batch, dim_nintrs, -1)  # [batch, nintrs, ch/3]
            sh_mult = sh_mult[mask].unsqueeze(1)  # [mask_pts, 1, ch/3]
            # 4. Interpolate rgbdata, use SH coefficients to get to RGB
            sh_masked = color_features.view(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
            rgb_masked = torch.sum(sh_mult * sh_masked, dim=-1)  # [mask_pts, 3]
            rgb[mask] = rgb_masked

        return rgb


class NNRender(nn.Module):
    def __init__(self, feature_dim, sigma_net_width=64, sigma_net_layers=1):
        super().__init__()

        # Feature decoder (modified from Instant-NGP)
        self.sigma_net = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": sigma_net_width,
                "n_hidden_layers": sigma_net_layers,
            },
        )
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.in_dim_color = self.direction_encoder.n_output_dims + 15
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        self.color = None

    def compute_density(self, density_features, mask, rays_d):
        """
        :param density_features:  [n, feature_dim]
        :param rays_d:            [n, 3] (normalized between -1, +1)
        :return:
        """
        dim_batch, dim_nintrs = mask.shape
        sigma = torch.zeros(dim_batch, dim_nintrs, device=density_features.device, dtype=density_features.dtype)
        color = torch.zeros((dim_batch, dim_nintrs, 3), device=density_features.device, dtype=density_features.dtype)

        density_and_color = self.sigma_net(density_features)  # [n, 16]
        sigma_valid = density_and_color[:, 0]
        sigma_valid = F.relu(sigma_valid)
        sigma.masked_scatter_(mask, sigma_valid)

        # rays_d from [batch, 3] to [n_valid_intrs, n_dir_dims]
        rays_d = self.direction_encoder((rays_d + 1) / 2)  # tcnn SH encoding requires inputs to be in [0, 1]
        pwise_rays_d = torch.repeat_interleave(rays_d, mask.sum(1), dim=0)
        colors_valid = self.color_net(torch.cat((pwise_rays_d, density_and_color[:, 1:]), dim=-1))  # [n, 3]
        color.masked_scatter_(mask, colors_valid)
        self.color = color

        return sigma

    def compute_color(self, color_features=None, mask=None, rays_d=None):
        color = self.color
        del self.color
        return color


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LowrankLearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int,
                 step_size: float, grid_dim: int, rank: int):
        super().__init__()
        self.resolution = resolution
        if grid_dim not in {2, 3, 4}:
            raise ValueError("grid_dim must be 2, 3, or 4.")
        self.grid_dim = grid_dim
        self.num_features_per_dim = int(round(num_features ** (1 / self.grid_dim)))
        print("num_features_per_dim = %d" % (self.num_features_per_dim, ))
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size
        self.rank = rank
        self.renderer = NNRender(feature_dim=feature_dim)

        # Volume representation
        # self.Gxy = nn.Parameter(torch.empty(self.grid_dim, resolution, resolution, 1, rank))
        # self.Gxz = nn.Parameter(torch.empty(self.grid_dim, resolution, 1, resolution, rank))
        # self.Gyz = nn.Parameter(torch.empty(self.grid_dim, 1, resolution, resolution, rank))
        self.Gxy = nn.Parameter(torch.empty(rank, resolution, resolution))
        self.Gxz = nn.Parameter(torch.empty(rank, resolution, resolution))
        self.Gyz = nn.Parameter(torch.empty(rank, resolution, resolution))
        # total memory requirement is reso*reso*rank*3*3 compared to reso*reso*reso*3
        self.basis_mat = nn.Linear(3 * rank, self.grid_dim, bias=False)

        # Feature table
        self.F = nn.Parameter(torch.empty([feature_dim] + [self.num_features_per_dim] * self.grid_dim))

        self.init_params()

    def init_params(self):
        nn.init.normal_(self.Gxy, std=0.1)
        nn.init.normal_(self.Gxz, std=0.1)
        nn.init.normal_(self.Gyz, std=0.1)
        nn.init.normal_(self.F, std=0.05)

    def get_coordinate_plane(self, intrs_pts):
        """intrs_pts: B*N, 3"""
        # coordinate_plane: 3, B, N, 2 -> 3, B*N, 1, 2
        coordinate_plane = torch.stack((intrs_pts[..., [0, 1]], intrs_pts[..., [0, 2]], intrs_pts[..., [1, 2]])).detach().view(3, -1, 2)
        return coordinate_plane

    def compute_features(self, pts):
        # pts should be between 0 and resolution
        # rds should be between -1 and 1 (unit norm ray direction vector)
        # pts [n, 3]
        # rds [n, 3]

        # Sequential interpolation by grid_sample
        # move pts to be in [-1, 1]
        pts = (pts * 2 / self.resolution) - 1
        # Interpolate into G using pts
        # Like tensor-RF
        coo_plane = self.get_coordinate_plane(pts)
        xy_coef = grid_sample_wrapper(self.Gxy, coo_plane[0])  # [n, rank]
        xz_coef = grid_sample_wrapper(self.Gxz, coo_plane[1])  # [n, rank]
        yz_coef = grid_sample_wrapper(self.Gyz, coo_plane[2])  # [n, rank]
        Gvals = self.basis_mat(torch.cat((xy_coef, xz_coef, yz_coef), dim=1))  # [n, grid_dim]
        # Gvals = torch.cat((xy_coef.sum(-1), xz_coef.sum(-1), yz_coef.sum(-1)), dim=1)  # [n, 3]
        # Alternative impl (with 3d sampling)
        # grid = self.Gxy * self.Gxz * self.Gyz  # [grid_dim, reso, reso, reso, rank]
        # grid = torch.sum(grid, dim=-1)  # [grid_dim, reso, reso, reso]
        # Gvals = grid_sample_wrapper(grid, pts)  # [n, grid_dim]

        # Interpolate into F using G
        Fvals = grid_sample_wrapper(self.F, Gvals)  # [n, feature_dim]
        return Fvals

    def forward(self, rays_o, rays_d):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        intersection_pts, intersections, mask = get_intersections(rays_o, rays_d, self.radius, self.n_intersections, self.step_size)
        # mask has shape [batch, n_intrs]
        intersection_pts = intersection_pts[mask]  # [n_valid_intrs, 3] puts together the valid intrs from all rays
        # Normalization (between [-1, 1])
        intersection_pts = intersection_pts / self.radius
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

        # compute features and render
        features = self.compute_features(intersection_pts)

        sigma = self.renderer.compute_density(features, mask, rays_d)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)
        rgb_mask = abs_light > self.abs_light_thresh
        rgb = self.renderer.compute_density(features, rgb_mask, rays_d)
        rgb = shrgb2rgb(rgb, abs_light, True)
        return rgb

    def get_params(self, lr):
        params = [
            {"params": self.renderer.parameters(), "lr": lr},
            {"params": self.Gxy, "lr": lr},
            {"params": self.Gxz, "lr": lr},
            {"params": self.Gyz, "lr": lr},
            {"params": self.F, "lr": lr},
            {"params": self.basis_mat.parameters(), "lr": lr},
        ]
        return params
