import torch
import torch.nn as nn
from plenoxels.models.utils import get_intersections, pos_encode, grid_sample_wrapper
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha
from .renderers import NNRender, SHRender


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int,
                 step_size: float, grid_dim: int, num_scenes: int = 1, second_G: bool = False):
        super().__init__()
        self.resolution = resolution
        if grid_dim not in {2, 3, 4}:
            raise ValueError("grid_dim must be 2, 3, or 4.")
        self.grid_dim = grid_dim
        self.num_features_per_dim = int(round(num_features ** (1 / self.grid_dim)))
        print("num_features_per_dim = %d" % (self.num_features_per_dim, ))
        self.feature_dim = feature_dim
        self.offset_pe_dim = 4 * 6
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size
        self.second_G = second_G
        self.renderer = NNRender(feature_dim=feature_dim, sigma_net_width=64, sigma_net_layers=1)

        # Volume representation
        # Option 1: High-resolution grid that stores numbers that get rounded (update: used for
        #           interpolation) into indices into the feature table
        # Option 2: Store small features at multiple resolution grids, then concatenate these
        #           features and feed them through an MLP to predict numbers that get rounded into
        #           indices into the feature table
        # Starting with option 1 for simplicity
        self.G1 = nn.ParameterList()
        for scene in range(num_scenes):
            self.G1.append(nn.Parameter(torch.empty(self.grid_dim, resolution, resolution, resolution)))
        if self.second_G:
            self.G2 = nn.Parameter(torch.empty([self.grid_dim] + [16] * self.grid_dim))
        # Feature table
        self.F = nn.Parameter(torch.empty([feature_dim] + [self.num_features_per_dim] * self.grid_dim))

        self.init_params()

    def init_params(self):
        for p in self.G1.parameters():
            nn.init.normal_(p, std=0.01)
        if self.second_G:
            nn.init.normal_(self.G2, std=0.05)
        nn.init.normal_(self.F, std=0.05)

    def compute_features(self, pts, grid_id):
        # pts should be between 0 and resolution
        # rds should be between -1 and 1 (unit norm ray direction vector)
        # pts [n, 3]
        # rds [n, 3]

        # Sequential interpolation by grid_sample
        G1vals = grid_sample_wrapper(self.G1[grid_id], pts)  # [n, grid_dim]
        # Interpolate into G2 using G1
        G2vals = G1vals
        if self.second_G:
            G2vals = grid_sample_wrapper(self.G2, G1vals)  # [n, grid_dim]
        Fvals = grid_sample_wrapper(self.F, G2vals)  # [n, feature_dim]
        return Fvals

    def forward(self, rays_o, rays_d, grid_id=0):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        """
        intersection_pts, intersections, mask = get_intersections(rays_o, rays_d, self.radius, self.n_intersections, self.step_size)
        # mask has shape [batch, n_intrs]
        intersection_pts = intersection_pts[mask]  # [n_valid_intrs, 3] puts together the valid intrs from all rays
        # Normalization
        intersection_pts = intersection_pts / self.radius  # between [-1, +1]
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)

        # compute features and render
        features = self.compute_features(intersection_pts, grid_id)

        sigma = self.renderer.compute_density(features, mask, rays_d)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)

        rgb_mask = mask#abs_light > self.abs_light_thresh
        rgb = self.renderer.compute_color(features, rgb_mask, rays_d)
        rgb = shrgb2rgb(rgb, abs_light, True)
        return rgb

    def get_params(self, lr):
        params = [
            {"params": self.G1.parameters(), "lr": lr},
            {"params": self.F, "lr": lr},
            {"params": self.renderer.parameters(), "lr": lr},
        ]
        if self.second_G:
            params.append({"params": self.G2, "lr": lr})
        return params
