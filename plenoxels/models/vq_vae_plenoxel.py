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


class Decoder(nn.Module):
    def __init__(self, fine_reso, output_dim):
        super(Decoder, self).__init__()
        self.reso0 = fine_reso
        self.output_dim = output_dim

    def forward(self, inputs):
        """
        :param inputs: B, D
        :return: B, R, R, R, Do
        """
        return inputs.view(inputs.shape[0], self.reso0, self.reso0, self.reso0, self.output_dim)


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
        self.fine_mlp_h_size = 32

        self.data = nn.Parameter(torch.empty(
            self.embedding_dim, coarse_reso, coarse_reso, coarse_reso))
        self.vq_vae = VectorQuantizerEMA(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            commitment_cost=self.commitment_cost,
            decay=0.99)
        self.fine_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim + 6 * (self.num_freqs_dir + self.num_freqs_pt), self.fine_mlp_h_size),
            nn.ReLU(),
            nn.Linear(self.fine_mlp_h_size, self.fine_mlp_h_size),
            nn.ReLU(),
            nn.Linear(self.fine_mlp_h_size, self.fine_data_dim)
        )
        self.reset_params()

    def reset_params(self):
        torch.nn.init.normal_(self.data, std=0.1)
        torch.nn.init.zeros_(self.fine_mlp[-1].bias)

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

    def world2finecoo(self, pts: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to the relative coordinate of points within their coarse-grid cell.
        Relative coordinates are normalized between -1, 1.
        """
        # from [-radius, +radius] to [0, coarse_reso]
        pts_coarse_coo = pts * (self.coarse_reso / (self.radius * 2)) + self.coarse_reso / 2
        coarse_voxel_centers = torch.floor(pts_coarse_coo)
        pts_fine_coo = pts_coarse_coo - coarse_voxel_centers
        return pts_fine_coo * 2 - 1

    def positional_encoding(self, pts, dirs, pts_mask):
        """
        :param pts:   [n_mask, 3]
        :param dirs:    [batch, 3]
        :param pts_mask:    [batch, n_intrs]
        :return:  [n_mask, 6 * (self.num_freqs_pt + self.num_freqs_dir)]
        """
        expanded_dirs = dirs.unsqueeze(1).expand(pts_mask.shape[0], pts_mask.shape[1], 3)[pts_mask]
        return positional_encoding(pts, expanded_dirs, self.num_freqs_pt, self.num_freqs_dir)

    def encode_fine(self, patch_data, coo_data):
        fine_data = torch.cat((patch_data, coo_data), dim=1)
        return self.fine_mlp(fine_data)

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
        pts_fine_coo = self.world2finecoo(intrs_pts)
        pts_fine_coo_pe = self.positional_encoding(pts_fine_coo, rays_d, intrs_pts_mask)
        interp_data = self.encode_fine(quantized, pts_fine_coo_pe)

        """Rendering"""
        rgb, depth, alpha = self.render(interp_data, intrs_pts_mask, rays_d, intersections)

        return rgb, depth, commitment_loss, perplexity
