import torch
import tinycudann as tcnn
import numpy as np

from .base_decoder import BaseDecoder


class SHDecoder(BaseDecoder):
    def __init__(self, feature_dim: int):
        super().__init__()
        sh_degree = int(np.round(np.sqrt((feature_dim - 1) / 3) - 1))
        if feature_dim != ((sh_degree + 1) ** 2) * 3 + 1:
            raise ValueError(f"feature_dim is incorrect for SHDecoder with {sh_degree} degrees")
        print(f'using spherical harmonic degree {sh_degree}, which corresponds to {feature_dim} features')
        self.sh_degree = sh_degree + 1
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": self.sh_degree,
            },
        )

    def compute_density(self, features, rays_d, precompute_color=False):
        return features[..., -2:-1]

    def compute_color(self, features, rays_d):
        color_features = features[..., :-1]
        sh_mult = self.direction_encoder(rays_d)[:, None, :]  # [batch, 1, harmonic_components]
        rgb = color_features.view(color_features.shape[0], 3, sh_mult.shape[-1])  # [batch, 3, harmonic_components]
        return torch.sum(sh_mult * rgb, dim=-1)  # [batch, 3]

    def __repr__(self):
        return f"SHRender(sh_degree={self.sh_degree - 1})"
