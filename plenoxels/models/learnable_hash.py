import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn
from plenoxels.models.utils import get_intersections
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha

# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int, step_size: float):
        super().__init__()
        self.resolution = resolution
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size

        # Volume representation
        # Option 1: High-resolution grid that stores numbers that are treated as indices into the feature table (which is interpolated)
        # Option 2: Store small features at multiple resolution grids, then concatenate these features and feed them through an MLP to predict numbers that get rounded into indices into the feature table
        # Starting with option 1 for simplicity
        self.G = nn.Parameter(torch.empty(resolution, resolution, resolution, 1))

        # Feature table
        self.F = nn.Parameter(torch.empty(num_features, feature_dim))

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
        nn.init.uniform_(self.G, -1, 1)
        nn.init.normal_(self.F, 0.1)

    def get_neighbors(self, pts):
        # pts should be in grid coordinates, ranging from 0 to resolution
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
        post_floor = torch.clamp(torch.floor(pre_floor), min=0., max=self.resolution - 1)  # [n, 8, 3]
        return pts - post_floor[:,0,:], post_floor.long() 

    def eval_pts(self, pts, rds):
        # pts should be between 0 and resolution
        # rds should be between -1 and 1 (unit norm ray direction vector)
        # pts [n, 3]
        # rds [n, 3]
        offsets, neighbors = self.get_neighbors(pts)
        Gneighborvals = self.G[neighbors[:,:,0], neighbors[:,:,1], neighbors[:,:,2], 0]  # [n, 8]
        weights = trilinear_interpolation_weight(offsets)  # [n, 8]
        # Fneighborindices = ((Gneighborvals.clamp(-1, 1) + 1) * self.num_features / 2).floor().long()  # [n, 8] between 0 and num_features
        # Fneighborvals = self.F[Fneighborindices.view(-1)].view(-1, 8, self.feature_dim)  # [n, 8, feature_dim]
        # Interpolate the F vals using the G indices
        Fneighborindices = ((Gneighborvals.clamp(-1, 1) + 1) * self.num_features / 2).view(-1) # [n, 8] between 0 and num_features
        Fneighborindices_floor = Fneighborindices.floor().clamp(0, self.num_features - 1)
        Fneighborindices_ceil = Fneighborindices.ceil().clamp(0,self.num_features - 1)
        Fneighborvals = self.F[Fneighborindices_floor.long()].view(-1,8,self.feature_dim) * (Fneighborindices_ceil - Fneighborindices).view(-1,8,1) + self.F[Fneighborindices_ceil.long()].view(-1,8,self.feature_dim) * (Fneighborindices - Fneighborindices_floor).view(-1,8,1)
        Fvals = torch.sum(weights[..., None] * Fneighborvals, dim=1)  # [n, feature_dim]
        Fvals = self.sigma_net(Fvals)  # [n, 16]
        sigmas = Fvals[:, 0]
        # colors = torch.zeros(len(Fvals), 3).cuda()
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
        pointwise_rays_d = rays_d[:, None, :].expand(-1, mask.shape[1], -1)  # [batch, n_intersections, 3]
        pointwise_rays_d = pointwise_rays_d[mask]   # [n_valid_intrs, 3]
        sigma_masked, color_masked = self.eval_pts(intersection_pts, pointwise_rays_d)  
        # Rendering
        batch, nintrs = intersections.shape[0], intersections.shape[1] - 1
        sigma = torch.zeros(batch, nintrs, dtype=sigma_masked.dtype, device=sigma_masked.device)
        color = torch.zeros(batch, nintrs, 3, dtype=color_masked.dtype, device=color_masked.device)
        sigma = F.relu(sigma)
        sigma.masked_scatter_(mask, sigma_masked)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)  # both [batch, n_intrs-1]
        color.masked_scatter_(mask.unsqueeze(-1), color_masked)
        color = shrgb2rgb(color, abs_light, True)
        return color

    def get_params(self, lr):
        return [
            # {"params": self.G, "lr": lr},  # Try making G fixed, like in INGP
            {"params": self.F, "lr": lr},
            {"params": self.direction_encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr / 10},
        ]

def trilinear_interpolation_weight(xyzs):
    # xyzs should have shape [n_pts, 3] and denote the offset (as a fraction of voxel_len) from the 000 interpolation point
    xs = xyzs[:, 0]
    ys = xyzs[:, 1]
    zs = xyzs[:, 2]
    weight000 = (1 - xs) * (1 - ys) * (1 - zs)  # [n_pts]
    weight001 = (1 - xs) * (1 - ys) * zs  # [n_pts]
    weight010 = (1 - xs) * ys * (1 - zs)  # [n_pts]
    weight011 = (1 - xs) * ys * zs  # [n_pts]
    weight100 = xs * (1 - ys) * (1 - zs)  # [n_pts]
    weight101 = xs * (1 - ys) * zs  # [n_pts]
    weight110 = xs * ys * (1 - zs)  # [n_pts]
    weight111 = xs * ys * zs  # [n_pts]
    weights = torch.stack(
        [weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111],
        dim=-1)  # [n_pts, 8]
    return weights