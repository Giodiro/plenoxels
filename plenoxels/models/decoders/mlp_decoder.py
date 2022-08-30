import torch
import tinycudann as tcnn

from .base_decoder import BaseDecoder


class NNDecoder(BaseDecoder):
    def __init__(self, feature_dim, sigma_net_width=64, sigma_net_layers=1):
        super().__init__()

        self.feature_dim = feature_dim
        self.sigma_net_width = sigma_net_width
        self.sigma_net_layers = sigma_net_layers
        # Feature decoder (modified from Instant-NGP)
        self.sigma_net = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": sigma_net_width,
                "n_hidden_layers": sigma_net_layers,
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
        self.color = None

    def compute_density(self, features, rays_d):
        density_rgb = self.sigma_net(features)  # [batch, 16]
        density = density_rgb[:, :1]

        enc_rays_d = self.direction_encoder((rays_d + 1) / 2)
        color_features = torch.cat((density_rgb[:, 1:], enc_rays_d), dim=-1)
        self.color = self.color_net(color_features)

        return density

    def compute_color(self, features, rays_d):
        color = self.color
        del self.color
        return color  # noqa

    def __repr__(self):
        return f"NNRender(feature_dim={self.feature_dim}, sigma_net_width={self.sigma_net_width}, sigma_net_layers={self.sigma_net_layers})"
