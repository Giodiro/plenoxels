import torch
import tinycudann as tcnn

from .base_decoder import BaseDecoder


class SigmaNNDecoder(BaseDecoder):
    def __init__(self, feature_dim, sigma_net_width=64, sigma_net_layers=1):
        super(SigmaNNDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.sigma_net_width = sigma_net_width
        self.sigma_net_layers = sigma_net_layers
        self.sigma_net = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": sigma_net_width,
                "n_hidden_layers": sigma_net_layers,
            },
        )

    def compute_density(self, features, rays_d, **kwargs):
        sigmas = self.sigma_net(features)  # [batch, 1]
        return sigmas[:, 0]  # [batch,]

    def compute_color(self, features, rays_d):
        raise NotImplementedError("SigmaNNDecoder does not implement color.")

    def forward(self, *args, **kwargs):
        return self.compute_density(*args, **kwargs)

    def __repr__(self):
        return f"SigmaNNDecoder(feature_dim={self.feature_dim}, sigma_net_width={self.sigma_net_width}, sigma_net_layers={self.sigma_net_layers})"


class NNDecoder(BaseDecoder):
    def __init__(self, feature_dim, sigma_net_width=64, sigma_net_layers=1, color_net=2, appearance_code_size=0):
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
                "activation": "None", #  "ReLU",
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
        self.in_dim_color = self.direction_encoder.n_output_dims + 15 + appearance_code_size
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": color_net,
            },
        )
        self.density_rgb = None  # output of the sigma-net

    def compute_density(self, features, rays_d, precompute_color: bool = True):
        density_rgb = self.sigma_net(features)  # [batch, 16]
        density = density_rgb[:, -1]
        self.density_rgb = density_rgb
        return density

    def compute_color(self, features, rays_d):
        enc_rays_d = self.direction_encoder((rays_d + 1) / 2)
        color_features = torch.cat((self.density_rgb[:, :-1], enc_rays_d), dim=-1)
        color = self.color_net(color_features)
        del self.density_rgb  # delete to avoid surprises
        return color

    def __repr__(self):
        return (f"NNRender(feature_dim={self.feature_dim}, sigma_net_width={self.sigma_net_width}, "
                f"sigma_net_layers={self.sigma_net_layers})")
