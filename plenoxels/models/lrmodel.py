from typing import Optional

import torch
import torch.nn as nn

from plenoxels.raymarching.ray_samplers import RaySamples


class LowrankModel(nn.Module):
    def __init__(self):
        super().__init__()

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()


class LowrankField(nn.Module):
    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        positions = SceneBox.get_normalized_positions(positions, self.aabb)
        selector = ((positions >= -1.0) & (positions <= 1.0)).all(dim=-1)
        positions_flat = positions.view(-1, 3)

        features = self.compute_features(positions_flat)
        density = (
            self.density_act(self.decoder.compute_density(
                features, rays_d=None, precompute_color=False)).view((*positions.shape[:-1], 1))
            * selector[..., None]
        )
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[torch.Tensor] = None):
        assert density_embedding is not None
        outputs = {}

        positions = ray_samples.get_positions()
        directions = ray_samples.directions.expand(positions.shape)
        rgb = self.decoder.compute_color(density_embedding, rays_d=ray_samples.directions)

        return outputs

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray.
        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)
        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)

        field_outputs["density"] = density
        return field_outputs
