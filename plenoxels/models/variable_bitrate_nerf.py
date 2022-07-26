from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.models.utils import pos_encode, get_intersections
from plenoxels.nerf_rendering import sigma2alpha, shrgb2rgb
from plenoxels.tc_interpolate import TrilinearInterpolate


def morton3d(x, y, z):
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)
    return xx | (yy << 1) | (zz << 2)


def expand_bits(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v


class VBrNerfLayer(nn.Module):
    def __init__(self, resolution, num_codebook_bits, data_dim):
        super().__init__()

        self.resolution = resolution
        self.num_codebook_bits = num_codebook_bits
        self.num_feat = 2 ** self.num_codebook_bits
        self.data_dim = data_dim
        self.use_morton = not self.resolution & self.resolution - 1

        self.grid = nn.Parameter(torch.empty(self.resolution ** 3, self.num_feat))
        self.codebook = nn.Parameter(torch.empty(self.num_feat, self.data_dim))
        self.register_buffer('nbr_offsets', torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.long))
        self.init_params()

    def init_params(self):
        nn.init.normal_(self.grid, 0, 0.01)
        nn.init.normal_(self.codebook, 0, 0.1)  # TODO: no mention in paper of initialization for codebook

    def forward(self, ray_p):
        tl_coo = torch.floor(ray_p)
        tl_offset = ray_p - tl_coo
        nbr_coo = torch.clamp(tl_coo[:, None, :].long() + self.nbr_offsets[None, :, :], 0, self.resolution - 1)
        nbr_coo = nbr_coo.view(-1, 3)
        if self.use_morton:
            nbr_idx = morton3d(nbr_coo[:, 0], nbr_coo[:, 1], nbr_coo[:, 2])
        else:
            nbr_idx = nbr_coo[:, 0] ** 3 + nbr_coo[:, 1] ** 2 + nbr_coo[:, 2]
        feat_soft = F.softmax(self.grid[nbr_idx], dim=-1)
        feat_idx = feat_soft.max(dim=-1, keepdim=True)[1]
        feat_hard = torch.zeros_like(feat_soft).scatter_(-1, feat_idx, 1.0)
        feat_hard = feat_hard - feat_soft.detach() + feat_soft

        data = feat_hard @ self.codebook
        data_interp = TrilinearInterpolate.apply(data.view(-1, 8, data.shape[-1]), tl_offset)
        return data_interp


class VBrRenderer(nn.Module):
    def __init__(self, data_dim, mlp_width=128, num_freqs=9):
        super().__init__()

        self.viewdir_pe_freqs = num_freqs
        self.viewdir_pe_dim = self.viewdir_pe_freqs * 3 * 2
        self.data_dim = data_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.data_dim + self.viewdir_pe_dim, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, 4)
        )

    def forward(self, rays_d, intrs, intrs_mask, data):
        batch = intrs.shape[0]
        nintrs = intrs.shape[1] - 1

        dir_pe = pos_encode(
            torch.repeat_interleave(rays_d, intrs_mask.sum(1), dim=0),
            self.viewdir_pe_freqs
        )
        mlp_input = torch.cat((data, dir_pe), dim=-1)
        preact = self.mlp(mlp_input)

        density_mskd = F.relu(preact[:, 0])
        color_mskd = preact[:, 1:]  # Will be passed through sigmoid in shrgb2rgb

        density = torch.zeros(batch, nintrs, dtype=density_mskd.dtype, device=density_mskd.device)
        density.masked_scatter_(intrs_mask, density_mskd)
        alpha, abs_light = sigma2alpha(density, intrs, rays_d)  # both [batch, n_intrs-1]

        color = torch.zeros(batch, nintrs, 3, dtype=color_mskd.dtype, device=color_mskd.device)
        color.masked_scatter_(intrs_mask.unsqueeze(-1), color_mskd)
        color = shrgb2rgb(color, abs_light, True)

        return color, alpha


class VBrNerf(nn.Module):
    def __init__(self, reso_list: List[int], cb_bits: int, scene_radius: float):
        super().__init__()
        # TODO: assert reso_list is increasing
        self.grids = nn.ModuleList([
            VBrNerfLayer(reso, num_codebook_bits=cb_bits, data_dim=16)
            for reso in reso_list
        ])
        self.renderer = VBrRenderer(data_dim=16, mlp_width=128, num_freqs=4)
        self.reso_list = reso_list
        self.scene_radius = scene_radius
        self.sampling_weights = 2 ** torch.arange(len(reso_list), dtype=torch.float)

    def forward(self, rays_o, rays_d, level):
        if level is None:
            # In training level is None, and we sample it as in the paper
            level = torch.multinomial(self.sampling_weights, num_samples=1).item()

        # The sample proposal is different in paper, but this makes more sense maybe.
        # In paper: sample 16 points (how?) for each intersected voxel.
        step_size = self.scene_radius / self.reso_list[level]
        n_intersections = int(1.732 * self.scene_radius * 2 / step_size)
        intrs_pts, intrs, intrs_mask = get_intersections(
            rays_o, rays_d, self.scene_radius, n_intersections, step_size)
        intrs_pts = intrs_pts[intrs_mask]
        intrs_pts = (intrs_pts / self.scene_radius + 1) * (self.reso_list[level] / 2)

        feats = self.grids[level](intrs_pts)
        color, alpha = self.renderer(rays_d, intrs, intrs_mask, feats)
        return color, alpha
