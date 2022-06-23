import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.models.utils import get_intersections, positional_encoding, interp_regular
from plenoxels.nerf_rendering import sigma2alpha, shrgb2rgb, depth_map


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.empty(num_embeddings, self._embedding_dim))

        self._decay = decay
        self._epsilon = epsilon
        self.reset_params()

    def reset_params(self):
        torch.nn.init.normal_(self._embedding.weight)
        torch.nn.init.normal_(self._ema_w)

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class MLPDecoderWithPE(nn.Module):
    def __init__(self, embedding_dim: int, out_dim: int, coarse_reso: int, radius: int, num_freqs_pt: int, num_freqs_dir: int):
        super(MLPDecoderWithPE, self).__init__()
        self.coarse_reso = coarse_reso
        self.radius = radius
        self.num_freqs_pt = num_freqs_pt
        self.num_freqs_dir = num_freqs_dir
        self.fine_mlp_h_size = 32
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim

        self.fine_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + 6 * (self.num_freqs_dir + self.num_freqs_pt), self.fine_mlp_h_size),
            nn.ReLU(),
            nn.Linear(self.fine_mlp_h_size, self.fine_mlp_h_size),
            nn.ReLU(),
            nn.Linear(self.fine_mlp_h_size, self.out_dim)
        )
        self.reset_params()

    def reset_params(self):
        torch.nn.init.zeros_(self.fine_mlp[-1].bias)

    def forward(self, patch_data, pts, rays_d, pts_mask):
        """From the patch-quantized data + position within the patch to some SH or RGB data"""
        # Convert world coordinates to the relative coordinate of points within their coarse-grid cell.
        # Relative coordinates are normalized between -1, 1.
        pts_coarse_coo = pts * (self.coarse_reso / (self.radius * 2)) + self.coarse_reso / 2
        coarse_voxel_centers = torch.floor(pts_coarse_coo)
        pts_fine_coo = (pts_coarse_coo - coarse_voxel_centers) * 2 - 1  # [-1, 1]

        expanded_dirs = rays_d.unsqueeze(1).expand(pts_mask.shape[0], pts_mask.shape[1], 3)[pts_mask]
        pts_fine_coo_pe = positional_encoding(pts_fine_coo, expanded_dirs, self.num_freqs_pt, self.num_freqs_dir)
        fine_data = torch.cat((patch_data, pts_fine_coo_pe), dim=1)
        return self.fine_mlp(fine_data)


class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super(ResBlock3d, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        if in_channels == out_channels:
            self.conv_id = torch.nn.Identity()
        else:
            self.conv_id = torch.nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        self.norm1 = torch.nn.GroupNorm(num_groups, in_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups, out_channels)

    def forward(self, x0):
        x = self.norm1(x0)
        x = F.hardswish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.hardswish(x)
        x = self.conv2(x)
        x = x + self.conv_id(x0)
        return x


class UpsamplingPatchDecoder(nn.Module):
    def __init__(self, embedding_dim: int, out_dim: int, coarse_reso: int, radius: int):
        super(UpsamplingPatchDecoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.coarse_reso = coarse_reso
        self.radius = radius
        self.out_dim = out_dim
        self.fine_reso = 4
        self.fine_reso_1 = 3
        self.fine_reso_2 = (6, 6, 6)
        self.fine_reso_3 = (12, 12, 12)
        self.fine_reso_start = 3
        self.ch_start = 32
        self.upsmp_fc1 = nn.Linear(self.embedding_dim, (self.fine_reso_1 ** 3) * self.ch_start)
        self.conv_blocks = nn.Sequential(
            torch.nn.Conv3d(self.ch_start, self.ch_start * 2, kernel_size=(1, 1, 1)),   # size: fr1, ch: *2
            torch.nn.Upsample(size=self.fine_reso_2, mode='nearest'),                   # size: fr2, ch: *2
            ResBlock3d(self.ch_start * 2, self.ch_start * 4),                           # size: fr2, ch: *4
            torch.nn.Upsample(size=self.fine_reso_3, mode='nearest'),                   # size: fr3, ch: *4
            ResBlock3d(self.ch_start * 4, self.ch_start * 4),                           # size: fr3, ch: *4
            torch.nn.GroupNorm(32, self.ch_start * 4),
            torch.nn.Hardswish(),
            torch.nn.Conv3d(self.ch_start * 4, self.out_dim, kernel_size=(3, 3, 3), padding=1)
        )

    def forward(self, patch_data, pts, rays_d, pts_mask):
        """Upsample patch_data until it's of some reasonable fine-reso size, then fetch the
        right pt interpolating"""
        pts_coarse_coo = pts * (self.coarse_reso / (self.radius * 2)) + self.coarse_reso / 2
        coarse_voxel_centers = torch.floor(pts_coarse_coo)
        pts_fine_coo = (pts_coarse_coo - coarse_voxel_centers) * 2 - 1  # [-1, 1]

        patch_data = self.upsmp_fc1(patch_data)
        patch_data = patch_data.reshape(
            patch_data.shape[0], self.ch_start, self.fine_reso_start, self.fine_reso_start, self.fine_reso_start)
        patch_data = self.conv_blocks(patch_data)  # B, Od, D, W, H
        point_in_patch = interp_regular(grid=patch_data, pts=pts_fine_coo.view(pts_fine_coo.shape[0], 1, 1, 1, 3))
        return point_in_patch


class VqVaePlenoxel(nn.Module):
    def __init__(self, coarse_reso, embedding_dim, num_embeddings, commitment_cost, radius,
                 num_freqs_pt, num_freqs_dir, sh_encoder, sh_deg):
        super(VqVaePlenoxel, self).__init__()

        self.coarse_reso = coarse_reso
        self.radius = radius
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.num_freqs_pt = num_freqs_pt
        self.num_freqs_dir = num_freqs_dir
        self.sh_encoder = sh_encoder
        self.fine_data_dim = ((sh_deg + 1) ** 2) * 3 + 1
        self.step_size, self.n_intersections = self.calc_step_size()
        print("Ray-marching with step-size = %.4e  -  %d intersections" %
              (self.step_size, self.n_intersections))

        self.data = nn.Parameter(torch.empty(
            self.embedding_dim, coarse_reso, coarse_reso, coarse_reso))
        self.vq_vae = VectorQuantizerEMA(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.commitment_cost,
            decay=0.99)
        self.decoder = MLPDecoderWithPE(
            embedding_dim=self.embedding_dim, out_dim=self.fine_data_dim,
            coarse_reso=self.coarse_reso, radius=self.radius,
            num_freqs_pt=self.num_freqs_pt, num_freqs_dir=self.num_freqs_dir)
        self.reset_params()

    def reset_params(self):
        torch.nn.init.normal_(self.data)

    def calc_step_size(self) -> Tuple[float, int]:
        step_size_factor = 4
        units = (self.radius * 2) / (self.coarse_reso - 1)
        step_size = units / (2 * step_size_factor)
        grid_diag = math.sqrt(3) * self.radius * 2
        n_intersections = int(grid_diag / step_size) - 1
        return step_size, n_intersections

    def fetch_coarse(self, pts: torch.Tensor) -> torch.Tensor:
        pts = pts / self.radius  # -1 +1
        pts = (pts + 1) * self.coarse_reso / 2  # 0, coarse_reso
        pts = torch.floor(pts).clamp(0, self.coarse_reso - 1).long()
        coarse_data = self.data[:, pts[:, 0], pts[:, 1], pts[:, 2]]
        return coarse_data.T  # [n_pts, coarse_dim]

    def fetch_coarse_interp(self, pts):
        pts = pts / self.radius
        coarse_data = interp_regular(
            self.data.unsqueeze(0), pts.view(1, -1, 1, 1, 3))
        return coarse_data.T

    def render(self, sh_data, mask, rays_d, intersections):
        batch, nintrs = mask.shape
        # 1. Process density: Un-masked sigma (batch, n_intrs-1), and compute.
        sigma_masked = F.relu(sh_data[:, -1])
        sigma = torch.zeros(batch, nintrs, dtype=sh_data.dtype, device=sh_data.device)
        sigma.masked_scatter_(mask, sigma_masked)
        alpha, abs_light = sigma2alpha(sigma, intersections, rays_d)  # both [batch, n_intrs-1]

        # 3. Create SH coefficients and mask them
        sh_mult = self.sh_encoder(rays_d).unsqueeze(1).expand(batch, nintrs, -1)  # [batch, nintrs, ch/3]
        sh_mult = sh_mult[mask].unsqueeze(1)  # [mask_pts, 1, ch/3]

        # 4. Interpolate rgbdata, use SH coefficients to get to RGB
        sh_masked = sh_data[:, :-1]
        sh_masked = sh_masked.view(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
        rgb_masked = torch.sum(sh_mult * sh_masked, dim=-1)   # [mask_pts, 3]

        # 5. Post-process RGB
        rgb = torch.zeros(batch, nintrs, 3, dtype=rgb_masked.dtype, device=rgb_masked.device)
        rgb.masked_scatter_(mask.unsqueeze(-1), rgb_masked)
        rgb = shrgb2rgb(rgb, abs_light, True)

        # 6. Depth map (optional)
        depth = depth_map(abs_light, intersections)  # [batch]

        return rgb, depth, alpha

    def forward(self, rays_o, rays_d, grid_id):
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=1, keepdim=True)
        intrs_pts, intersections, intrs_pts_mask = get_intersections(
            rays_o, rays_d, self.radius, self.n_intersections, self.step_size)
        intrs_pts = intrs_pts[intrs_pts_mask]

        """Get the coarse patches corresponding to the points"""
        coarse_patches = self.fetch_coarse_interp(intrs_pts)  # [n_pts, coarse_dim]

        """Quantize"""
        # quantized: n_pts, coarse_dim
        commitment_loss, quantized, perplexity, _ = self.vq_vae(coarse_patches)

        """From the patch-quantized data + position within the patch to some SH or RGB data"""
        interp_data = self.decoder(quantized, intrs_pts, rays_d, intrs_pts_mask)

        """Rendering"""
        rgb, depth, alpha = self.render(interp_data, intrs_pts_mask, rays_d, intersections)

        return rgb, depth, commitment_loss, perplexity
