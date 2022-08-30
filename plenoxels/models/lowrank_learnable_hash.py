import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from plenoxels.models.utils import get_intersections, grid_sample_wrapper
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LowrankLearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int,
                 step_size: float, grid_dim: int, rank: int, G_init_std: float, second_G=False):
        super().__init__()
        self.resolution = resolution
        if grid_dim not in {2, 3, 4}:
            raise ValueError("grid_dim must be 2, 3, or 4.")
        self.grid_dim = grid_dim
        self.num_features_per_dim = int(round(num_features**(1/self.grid_dim)))
        print("num_features_per_dim = %d" % (self.num_features_per_dim, ))
        self.feature_dim = feature_dim
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size
        self.second_G = second_G
        self.rank = rank
        print(f"rank: {self.rank}")
        self.G_init_std = G_init_std

        # Volume representation
        self.G = nn.Parameter(torch.empty(3, self.grid_dim * rank, resolution, resolution)) # The 3 is for xy, xz, yz
        # total memory requirement is reso*reso*rank*3*4 compared to reso*reso*reso*4
        if self.second_G:
            self.G2 = nn.Parameter(torch.empty([self.grid_dim] + [8] * self.grid_dim))

        # Feature table
        self.F = nn.Parameter(torch.empty([feature_dim] + [self.num_features_per_dim] * self.grid_dim))

        # Feature decoder (modified from Instant-NGP)
        self.sigma_net = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
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

        self.init_params()

    def init_params(self):
        nn.init.normal_(self.G, std=self.G_init_std)
        if self.second_G:
            nn.init.normal_(self.G2, std=self.G_init_std)
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
        coo_plane = torch.stack((pts[..., [0, 1]], pts[..., [0, 2]], pts[..., [1, 2]]))  # [3, n, 2]
        interp = grid_sample_wrapper(self.G, coo_plane).view(3, -1, self.grid_dim, self.rank)
        Gvals = interp.prod(dim=0).sum(dim=-1)  # [n, grid_dim]
        # Interpolate into G2 using G
        G2vals = Gvals
        if self.second_G:
            G2vals = grid_sample_wrapper(self.G2, Gvals)  # [n, grid_dim]
        # Interpolate into F using G
        Fvals = grid_sample_wrapper(self.F, G2vals)  # [n, feature_dim]

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
            {"params": self.G, "lr": lr * 1},
            {"params": self.F, "lr": lr * 1},
            {"params": self.direction_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        if self.second_G:
            params.append({"params": self.G2, "lr": lr * 1})
        return params

