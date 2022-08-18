import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
from plenoxels.models.utils import get_intersections
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LowrankLearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int,
                 step_size: float, rank: int):
        super().__init__()
        self.resolution = resolution
        self.num_features_per_dim = int(np.round(num_features**(1./3)))
        print("num_features_per_dim = %d" % (self.num_features_per_dim, ))
        self.feature_dim = feature_dim
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size
        self.rank = rank

        # Volume representation
        # self.G = nn.Parameter(torch.empty(resolution, resolution, resolution, 3))
        self.Gxy = nn.Parameter(torch.empty(resolution, resolution, 1, 3, rank))
        self.Gxz = nn.Parameter(torch.empty(resolution, 1, resolution, 3, rank))
        self.Gyz = nn.Parameter(torch.empty(1, resolution, resolution, 3, rank))
        # total memory requirement is reso*reso*rank*3*3 compared to reso*reso*reso*3
        
        # Feature table
        self.F = nn.Parameter(torch.empty(self.num_features_per_dim, self.num_features_per_dim, self.num_features_per_dim, feature_dim))

        # Feature decoder (modified from Instant-NGP)
        self.sigma_net = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 2,
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
        self.register_buffer('nbr_offsets_01', torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.long))
        self.register_buffer('nbr_offsets_m11', torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=torch.float))

        self.init_params()

    def init_params(self):
        nn.init.normal_(self.Gxy, std=0.1) 
        nn.init.normal_(self.Gxz, std=0.1) 
        nn.init.normal_(self.Gyz, std=0.1)
        nn.init.normal_(self.F, std=0.05)

    def eval_pts(self, pts, rds):
        # pts should be between 0 and resolution
        # rds should be between -1 and 1 (unit norm ray direction vector)
        # pts [n, 3]
        # rds [n, 3]

        # Sequential interpolation by grid_sample
        # move pts to be in [-1, 1]
        pts = (pts * 2 / self.resolution) - 1
        # Interpolate into G using pts
        grid = self.Gxy * self.Gxz * self.Gyz  # [reso, reso, reso, 3, rank]
        grid = torch.sum(grid, dim=-1)  # [reso, reso, reso, 3]
        Gvals = F.grid_sample(
            grid[None, ...].permute(0, 4, 1, 2, 3),  # [1, 3, reso, reso, reso]
            pts[None, None, None, ...],  # [1, 1, 1, n, 3]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, 3]
        # Interpolate into F using G
        Fvals = F.grid_sample(
            self.F[None, ...].permute(0, 4, 1, 2, 3),  # [1, feature_dim, reso, reso, reso]
            Gvals[None, None, None, ...],  # [1, 1, 1, n, 3]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, feature_dim]

        # Extract color and density from features
        Fvals = self.sigma_net(Fvals)  # [n, 16]
        sigmas = Fvals[:, 0]
        encoded_rd = self.direction_encoder((rds + 1) / 2) # tcnn SH encoding requires inputs to be in [0, 1]
        colors = self.color_net(torch.cat([encoded_rd, Fvals[:,1:]], dim=-1))  # [n, 3]
        return sigmas, colors

    def forward(self, rays_o, rays_d):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        intersection_pts, intersections, mask = get_intersections(rays_o, rays_d, self.radius, self.n_intersections, self.step_size)
        # mask has shape [batch, n_intrs]
        intersection_pts = intersection_pts[mask]  # [n_valid_intrs, 3] puts together the valid intrs from all rays
        # Normalization
        intersection_pts = (intersection_pts / self.radius + 1) * self.resolution / 2  # between [0, reso]
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

        pointwise_rays_d = torch.repeat_interleave(rays_d, mask.sum(1), dim=0)  # [n_valid_intrs, 3]
        sigma_masked, color_masked = self.eval_pts(intersection_pts, pointwise_rays_d)
        # Rendering
        batch, nintrs = intersections.shape[0], intersections.shape[1] - 1

        sigma = torch.zeros(batch, nintrs, dtype=sigma_masked.dtype, device=sigma_masked.device)
        sigma.masked_scatter_(mask, sigma_masked)
        sigma = F.relu(sigma)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)  # both [batch, n_intrs-1]

        color = torch.zeros(batch, nintrs, 3, dtype=color_masked.dtype, device=color_masked.device)
        color.masked_scatter_(mask.unsqueeze(-1), color_masked)
        color = shrgb2rgb(color, abs_light, True)
        return color

    def get_params(self, lr):
        params = [
            {"params": self.Gxy, "lr": lr * 1},
            {"params": self.Gxz, "lr": lr * 1},
            {"params": self.Gyz, "lr": lr * 1},
            {"params": self.F, "lr": lr * 1},
            {"params": self.direction_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        return params

