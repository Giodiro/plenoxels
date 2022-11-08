from typing import Union

import torch
import torch.nn as nn

from plenoxels.raymarching.ray_samplers import RaySamples


class DepthRenderer(nn.Module):
    """Calculate depth along ray.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        weights: torch.Tensor,
        ray_samples: RaySamples,
    ) -> torch.Tensor:
        """Composite samples along ray and calculate disparities.
        Args:
            weights: Weights for each sample.  [n_rays, n_samples, 1]
            ray_samples: Set of ray samples.
        Returns:
            Outputs of depth values. [n_rays, 1]
        """

        eps = 1e-10
        steps = (ray_samples.starts + ray_samples.ends) / 2

        depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)

        depth = torch.clip(depth, steps.min(), steps.max())

        return depth


class RGBRenderer(nn.Module):
    """Standard volumetic rendering.
    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color: Union[str, torch.Tensor] = "random") -> None:
        super().__init__()
        self.background_color = background_color

    @classmethod
    def combine_rgb(
        cls,
        rgb: torch.Tensor,
        weights: torch.Tensor,
        background_color: Union[str, torch.Tensor] = "random",
    ) -> torch.Tensor:
        """Composite samples along ray and render color image
        Args:
            rgb: RGB for each sample  [n_rays, n_samples, 3]
            weights: Weights for each sample  [n_rays, n_samples, 3]
            background_color: Background color as RGB.
        Returns:
            Outputs rgb values.  [n_rays, 3]
        """
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)

        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        if background_color == "random":
            background_color = torch.rand_like(comp_rgb).to(rgb.device)

        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color.to(weights.device) * (1.0 - accumulated_weight)

        return comp_rgb

    def forward(
        self,
        rgb: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Composite samples along ray and render color image
        Args:
            rgb: RGB for each sample  [n_rays, n_samples, 3]
            weights: Weights for each sample  [n_rays, n_samples, 3]
        Returns:
            Outputs of rgb values.
        """
        background_color = self.background_color
        if not self.training:
            background_color = torch.tensor([0.5, 0.5, 0.5]).to(rgb.device)

        rgb = self.combine_rgb(
            rgb, weights, background_color=background_color
        )
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb
