import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


class NNRender(nn.Module):
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

    def compute_density(self, density_features, mask, rays_d):
        """
        :param density_features:  [n, feature_dim]
        :param rays_d:            [n, 3] (normalized between -1, +1)
        :return:
        """
        dim_batch, dim_nintrs = mask.shape

        density_and_color = self.sigma_net(density_features)  # [n, 16]
        sigma_valid = density_and_color[:, 0]
        sigma_valid = F.relu(sigma_valid)
        sigma = torch.zeros(dim_batch, dim_nintrs, device=sigma_valid.device, dtype=sigma_valid.dtype)
        sigma.masked_scatter_(mask, sigma_valid)

        # rays_d from [batch, 3] to [n_valid_intrs, n_dir_dims]
        rays_d = self.direction_encoder((rays_d + 1) / 2)  # tcnn SH encoding requires inputs to be in [0, 1]
        pwise_rays_d = torch.repeat_interleave(rays_d, mask.sum(1), dim=0)
        colors_valid = self.color_net(torch.cat((pwise_rays_d, density_and_color[:, 1:]), dim=-1))  # [n, 3]
        color = torch.zeros((dim_batch, dim_nintrs, 3), device=colors_valid.device, dtype=colors_valid.dtype)
        color.masked_scatter_(mask.unsqueeze(-1), colors_valid)
        #color[mask] = colors_valid
        self.color = color

        return sigma

    def compute_color(self, color_features=None, mask=None, rays_d=None):
        color = self.color
        del self.color
        return color

    def __repr__(self):
        return f"NNRender(feature_dim={self.feature_dim}, sigma_net_width={self.sigma_net_width}, sigma_net_layers={self.sigma_net_layers})"
