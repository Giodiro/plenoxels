from typing import Optional, Dict

import torch
import torch.nn as nn

from .grids.learnable_hash import LearnableHashGrid
from .decoders import NNDecoder, SHDecoder


class NeuralRadianceField(nn.Module):
    """Model for encoding radiance fields (density and plenoptic color)"""
    def __init__(self,
                 decoder_lod_idx: int,
                 grid_type: str,
                 decoder_type: str,
                 **kwargs):
        super().__init__()
        self.decoder = None
        self.grid = None

        self.grid_type = grid_type
        self.decoder_type = decoder_type
        self.extra_args = kwargs

        self.decoder_lod_idx = decoder_lod_idx
        self.embedder_output_dim = 0
        self.input_dim = 0

    def init_grid(self):
        """Initialize the grid object.
        """
        if self.grid_type == "LearnableHashGrid":
            grid_class = LearnableHashGrid
        else:
            raise NotImplementedError(self.grid_type)
        self.grid = grid_class(**self.extra_args)

    def init_decoder(self):
        """Initializes the decoder object."""
        self.input_dim = self.grid.feature_dim(self.decoder_lod_idx) + self.embedder_output_dim

        if self.decoder_type == "ingp":
            self.decoder = NNDecoder(self.input_dim, sigma_net_width=self.extra_args['sigma_net_width'],
                                     sigma_net_layers=self.extra_args['sigma_net_layers'])
        elif self.decoder_type == "sh":
            self.decoder = SHDecoder(self.input_dim, sh_degree=self.extra_args['sh_degree'])
        else:
            raise ValueError(f"decoder type {self.decoder_type} invalid.")

    def forward(self, coords, rays_d, lod_idx: Optional[int], scene_idx: int) -> Dict[str, torch.Tensor]:
        if lod_idx is None:
            lod_idx = self.grid.num_lods - 1
        batch, num_samples, _ = coords.shape

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx, scene_idx)

        # Decode high-dimensional vectors to RGBA.
        density = torch.relu(
            self.decoder.compute_density(feats, rays_d=rays_d))
        rgb = torch.sigmoid(
            self.decoder.compute_color(feats, rays_d=rays_d))
        return dict(rgb=rgb, density=density)

