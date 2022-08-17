import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.jit as jit
import numpy as np
import tinycudann as tcnn
from plenoxels.models.utils import get_intersections
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int,
                 step_size: float, second_G=False):
        super().__init__()
        self.resolution = resolution
        self.num_features_per_dim = int(np.round(num_features**(1./3)))
        # self.num_features_per_dim = int(np.round(num_features**(1./4)))
        # self.quadinterp = QuadInterp()  # was relevant for jit, but doesn't seem to help
        print("num_features_per_dim = %d" % (self.num_features_per_dim, ))
        self.feature_dim = feature_dim
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size
        self.second_G = second_G

        # Volume representation
        # Option 1: High-resolution grid that stores numbers that get rounded (update: used for
        #           interpolation) into indices into the feature table
        # Option 2: Store small features at multiple resolution grids, then concatenate these
        #           features and feed them through an MLP to predict numbers that get rounded into
        #           indices into the feature table
        # Starting with option 1 for simplicity
        self.G1 = nn.Parameter(torch.empty(resolution, resolution, resolution, 3))
        # self.G1 = nn.Parameter(torch.empty(resolution, resolution, resolution, 4))
        if self.second_G:
            self.G2 = nn.Parameter(torch.empty(resolution // 4, resolution // 4, resolution // 4, 3))

        # Feature table
        self.F = nn.Parameter(torch.empty(self.num_features_per_dim, self.num_features_per_dim, self.num_features_per_dim, feature_dim))
        # self.F = nn.Parameter(torch.empty(self.num_features_per_dim, self.num_features_per_dim, self.num_features_per_dim, self.num_features_per_dim, feature_dim))

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
        nn.init.normal_(self.G1, std=0.01)
        if self.second_G:
            nn.init.normal_(self.G2, std=0.01)
        nn.init.normal_(self.F, std=0.05)

    def eval_pts(self, pts, rds):
        # pts should be between 0 and resolution
        # rds should be between -1 and 1 (unit norm ray direction vector)
        # pts [n, 3]
        # rds [n, 3]

        # Sequential interpolation by grid_sample
        # move pts to be in [-1, 1]
        pts = (pts * 2 / self.resolution) - 1
        # Interpolate into G1 using pts
        G1vals = F.grid_sample(
            self.G1[None, ...].permute(0, 4, 1, 2, 3),  # [1, 3, reso, reso, reso]
            pts[None, None, None, ...],  # [1, 1, 1, n, 3]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, 3]
        # Interpolate into G2 using G1
        G2vals = G1vals
        if self.second_G:
            G2vals = F.grid_sample(
                self.G2[None, ...].permute(0, 4, 1, 2, 3),  # [1, 3, reso, reso, reso]
                G1vals[None, None, None, ...],  # [1, 1, 1, n, 3]
                align_corners=True,
                mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, 3]
        # Interpolate into F using G2
        Fvals = F.grid_sample(
            self.F[None, ...].permute(0, 4, 1, 2, 3),  # [1, feature_dim, reso, reso, reso]
            G2vals[None, None, None, ...],  # [1, 1, 1, n, 3]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, feature_dim]
        # Fvals = quad_interp(self.F, G2vals)

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
            {"params": self.G1, "lr": lr * 1},
            {"params": self.F, "lr": lr * 1},
            {"params": self.direction_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
        ]
        if self.second_G:
            params.append({"params": self.G2, "lr": lr * 1})
        return params

# jit scripting makes it slower?
# class QuadInterp(jit.ScriptModule):
#     def __init__(self):
#         super().__init__()

#     @jit.script_method
    # def forward(self, grid, pts):
def quad_interp(grid, pts):
    # grid should be (n, n, n, n, feature_dim)
    # pts should be (n_pts, 4) with values between -1 and 1
    # Transform pts to be between 0 and n
    pts = (pts + 1) * len(grid) / 2.
    lox = torch.clamp(torch.floor(pts[:, 0]), min=0, max=len(grid)-1).long()  # [n_pts]
    loy = torch.clamp(torch.floor(pts[:, 1]), min=0, max=len(grid)-1).long()
    loz = torch.clamp(torch.floor(pts[:, 2]), min=0, max=len(grid)-1).long()
    low = torch.clamp(torch.floor(pts[:, 3]), min=0, max=len(grid)-1).long()
    hix = torch.clamp(torch.ceil(pts[:, 0]), min=0, max=len(grid)-1).long()
    hiy = torch.clamp(torch.ceil(pts[:, 1]), min=0, max=len(grid)-1).long()
    hiz = torch.clamp(torch.ceil(pts[:, 2]), min=0, max=len(grid)-1).long()
    hiw = torch.clamp(torch.ceil(pts[:, 3]), min=0, max=len(grid)-1).long()
    xyzws = pts - torch.floor(pts)
    xs = xyzws[:, 0][:, None]
    ys = xyzws[:, 1][:, None]
    zs = xyzws[:, 2][:, None]
    ws = xyzws[:, 3][:, None]
    values = torch.zeros(len(pts), grid.shape[-1]).to(grid.device)
    values = values + (1 - xs) * (1 - ys) * (1 - zs) * (1 - ws) * grid[lox, loy, loz, low]  # 0000
    values = values + (1 - xs) * (1 - ys) * (1 - zs) * ws * grid[lox, loy, loz, hiw]  # 0001
    values = values + (1 - xs) * (1 - ys) * zs * (1 - ws) * grid[lox, loy, hiz, low]  # 0010
    values = values + (1 - xs) * (1 - ys) * zs * ws * grid[lox, loy, hiz, hiw]  # 0011
    values = values + (1 - xs) * ys * (1 - zs) * (1 - ws) * grid[lox, hiy, loz, low]  # 0100
    values = values + (1 - xs) * ys * (1 - zs) * ws * grid[lox, hiy, loz, hiw]  # 0101
    values = values + (1 - xs) * ys * zs * (1 - ws) * grid[lox, hiy, hiz, low]  # 0110
    values = values + (1 - xs) * ys * zs * ws * grid[lox, hiy, hiz, hiw]  # 0111
    values = values + xs * (1 - ys) * (1 - zs) * (1 - ws) * grid[hix, loy, loz, low]  # 1000
    values = values + xs * (1 - ys) * (1 - zs) * ws * grid[hix, loy, loz, hiw]  # 1001
    values = values + xs * (1 - ys) * zs * (1 - ws) * grid[hix, loy, hiz, low]  # 1010
    values = values + xs * (1 - ys) * zs * ws * grid[hix, loy, hiz, hiw]  # 1011
    values = values + xs * ys * (1 - zs) * (1 - ws) * grid[hix, hiy, loz, low]  # 1100
    values = values + xs * ys * (1 - zs) * ws * grid[hix, hiy, loz, hiw]  # 1101
    values = values + xs * ys * zs * (1 - ws) * grid[hix, hiy, hiz, low]  # 1110
    values = values + xs * ys * zs * ws * grid[hix, hiy, hiz, hiw]  # 1111
    # neighbors = get_quad_neighbors(grid, pts)  # [n_pts, feature_dim, 16]
    # weights = quadrilinear_interpolation_weight(pts - torch.floor(pts))  # [n_pts, 16]
    # return torch.sum(neighbors * weights[:,None,:], dim=-1)  # [n_pts, feature_dim]
    return values

