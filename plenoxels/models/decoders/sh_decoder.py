import torch
import tinycudann as tcnn

from .base_decoder import BaseDecoder


class SHDecoder(BaseDecoder):
    def __init__(self, feature_dim: int, sh_degree: int):
        super().__init__()
        if feature_dim != ((sh_degree + 1) ** 2) * 3:
            raise ValueError(f"feature_dim is incorrect for SHDecoder with {sh_degree} degrees")
        self.sh_degree = sh_degree + 1
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": sh_degree,
            },
        )
        super().__init__()

    def compute_density(self, features, rays_d):
        return features[..., -2:-1]

    def compute_color(self, features, rays_d):
        color_features = features[..., :-1]
        sh_mult = self.direction_encoder(rays_d)[:, None, :]  # [batch, 1, ch/3]
        rgb = color_features.view(-1, 3, sh_mult.shape[-1])  # [batch, 3, ch/3]
        return torch.sum(sh_mult * rgb, dim=-1)  # [batch, 3]

    def __repr__(self):
        return f"SHRender(sh_degree={self.sh_degree})"
