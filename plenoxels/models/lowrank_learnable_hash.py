import torch
import torch.nn as nn
from plenoxels.models.utils import get_intersections, grid_sample_wrapper
from plenoxels.nerf_rendering import shrgb2rgb, sigma2alpha
from .renderers import NNRender, SHRender


# Some pieces modified from https://github.com/ashawkey/torch-ngp/blob/6313de18bd8ec02622eb104c163295399f81278f/nerf/network_tcnn.py
class LowrankLearnableHash(nn.Module):
    def __init__(self, resolution, num_features, feature_dim, radius: float, n_intersections: int,
                 step_size: float, grid_dim: int, rank: int, num_scenes: int = 1):
        super().__init__()
        self.resolution = resolution
        if grid_dim not in {2, 3, 4}:
            raise ValueError("grid_dim must be 2, 3, or 4.")
        self.grid_dim = grid_dim
        self.num_features_per_dim = int(round(num_features ** (1 / self.grid_dim)))
        self.radius = radius
        self.n_intersections = n_intersections
        self.step_size = step_size
        self.rank = rank
        self.renderer = NNRender(feature_dim=feature_dim, sigma_net_width=64, sigma_net_layers=1)

        # Volume representation
        self.Gxy = nn.ParameterList()
        self.Gxz = nn.ParameterList()
        self.Gyz = nn.ParameterList()
        self.basis_mat = nn.ModuleList()
        for scene in range(num_scenes):
            self.Gxy.append(nn.Parameter(torch.empty(rank, resolution, resolution)))
            self.Gxz.append(nn.Parameter(torch.empty(rank, resolution, resolution)))
            self.Gyz.append(nn.Parameter(torch.empty(rank, resolution, resolution)))
            self.basis_mat.append(nn.Linear(rank, self.grid_dim, bias=False))
            #self.Gxy.append(nn.Parameter(torch.empty(self.grid_dim, resolution, resolution, 1, rank)))
            #self.Gxz.append(nn.Parameter(torch.empty(self.grid_dim, resolution, 1, resolution, rank)))
            #self.Gyz.append(nn.Parameter(torch.empty(self.grid_dim, 1, resolution, resolution, rank)))
            # total memory requirement is reso*reso*rank*3*3 compared to reso*reso*reso*3

        # Feature table
        self.F = nn.Parameter(torch.empty([feature_dim] + [self.num_features_per_dim] * self.grid_dim))

        self.init_params()

        print("Initialized LowrankLearnableHash model")
        print("num_features_per_dim = %d" % (self.num_features_per_dim, ))
        print("renderer = %s" % (self.renderer))
        print("rank = %d" % (self.rank))
        print("grid_dim = %d" % (self.grid_dim))

    def init_params(self):
        g_std = 0.1
        for grid_id in range(len(self.Gxy)):
            nn.init.normal_(self.Gxy[grid_id], std=g_std)
            nn.init.normal_(self.Gxz[grid_id], std=g_std)
            nn.init.normal_(self.Gyz[grid_id], std=g_std)
            if hasattr(self, "basis_mat"):
                nn.init.normal_(self.basis_mat[grid_id].weight, std=0.1)
        nn.init.normal_(self.F, std=0.05)

    def get_coordinate_plane(self, intrs_pts):
        """intrs_pts: B*N, 3"""
        # coordinate_plane: 3, B, N, 2 -> 3, B*N, 1, 2
        coordinate_plane = torch.stack((intrs_pts[..., [0, 1]], intrs_pts[..., [0, 2]], intrs_pts[..., [1, 2]])).detach().view(3, -1, 2)
        return coordinate_plane

    def compute_features(self, pts, grid_id):
        # pts should be between 0 and resolution
        # rds should be between -1 and 1 (unit norm ray direction vector)
        # pts [n, 3]
        # rds [n, 3]

        # Sequential interpolation by grid_sample
        # Interpolate into G using pts
        # Like tensor-RF
        coo_plane = self.get_coordinate_plane(pts)
        xy_coef = grid_sample_wrapper(self.Gxy[grid_id], coo_plane[0])  # [n, rank]
        xz_coef = grid_sample_wrapper(self.Gxz[grid_id], coo_plane[1])  # [n, rank]
        yz_coef = grid_sample_wrapper(self.Gyz[grid_id], coo_plane[2])  # [n, rank]
        Gvals = self.basis_mat[grid_id](xy_coef * xz_coef * yz_coef)
        #Gvals = self.basis_mat(torch.cat((xy_coef, xz_coef, yz_coef), dim=1))  # [n, grid_dim]
        #Gvals = torch.cat((xy_coef.sum(-1), xz_coef.sum(-1), yz_coef.sum(-1)), dim=1)  # [n, 3]
        # Alternative impl (with 3d sampling)
        #grid = self.Gxy[grid_id] * self.Gxz[grid_id] * self.Gyz[grid_id]  # [grid_dim, reso, reso, reso, rank]
        #grid = torch.sum(grid, dim=-1)  # [grid_dim, reso, reso, reso]
        #Gvals = grid_sample_wrapper(grid, pts)  # [n, grid_dim]

        # Interpolate into F using G
        Fvals = grid_sample_wrapper(self.F, Gvals)  # [n, feature_dim]
        return Fvals

    def forward(self, rays_o, rays_d, grid_id=0):
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
        features = self.compute_features(intersection_pts, grid_id)

        sigma = self.renderer.compute_density(features, mask, rays_d)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)

        rgb_mask = mask#abs_light > self.abs_light_thresh
        rgb = self.renderer.compute_color(features, rgb_mask, rays_d)
        rgb = shrgb2rgb(rgb, abs_light, True)
        return rgb

    def get_params(self, lr):
        params = [
            {"params": self.renderer.parameters(), "lr": lr},
            {"params": self.Gxy.parameters(), "lr": lr},
            {"params": self.Gxz.parameters(), "lr": lr},
            {"params": self.Gyz.parameters(), "lr": lr},
            {"params": self.F, "lr": lr},
            {"params": self.basis_mat.parameters(), "lr": lr},
        ]
        return params
