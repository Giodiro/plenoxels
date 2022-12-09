"""
SH decoder takes a view direction and a set of spherical harmonic coefficients
and produces a color by evaluating the spherical harmonic basis functions at the
view direction and then combining them by a weighted sum according to the
coefficients.

This decoder replaces the spherical harmonic basis with a learned basis
defined by an MLP that takes in view direction and predicts a vector of 
values that are then combined by weighted average with the features as weights
to produce color.

Density is modeled with a linear decoder.
"""
import torch
import tinycudann as tcnn

from .base_decoder import BaseDecoder

class LearnedBasisDecoder(BaseDecoder):
    def __init__(self, feature_dim, net_width=64, net_layers=1, appearance_code_size=0):
        super().__init__()

        self.feature_dim = feature_dim
        self.net_width = net_width
        self.net_layers = net_layers
        
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        # The NN learns a basis that is used instead of spherical harmonics
        # Input is an encoded view direction, output is weights for 
        # combining the color features into RGB
        # This architecture is based on instant-NGP
        self.color_basis = tcnn.Network(
            n_input_dims=self.direction_encoder.n_output_dims,
            n_output_dims=3 * self.feature_dim,  # * (self.feature_dim - 1),  # The last feature is sigma (density)
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.net_width,
                "n_hidden_layers": self.net_layers,
            },
        )
        # sigma_net just does a linear transformation on the features to get density
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "None",
                "output_activation": "None",
                "n_neurons": self.net_width,
                "n_hidden_layers": 0,
            },
        )


    def compute_color(self, features, rays_d):
        if len(rays_d) < 1:
            return torch.zeros((0, 3)).to(rays_d.device)
        # color_features = features[..., :-1]  # [batch, color_feature_len]
        color_features = features
        enc_rays_d = self.direction_encoder((rays_d + 1) / 2)
        basis_values = self.color_basis(enc_rays_d)  # [batch, color_feature_len * 3]
        basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
        return torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]


    def compute_density(self, features, rays_d, **kwargs):
        if len(features) < 1:
            return torch.zeros((0, 1)).to(features.device)
        return self.sigma_net(features)  # [batch, 1]


    def __repr__(self):
        return (f"NNRender(feature_dim={self.feature_dim}, net_width={self.net_width}, "
                f"net_layers={self.net_layers})")
