from typing import Tuple

import torch

__all__ = ("shrgb2rgb", "depth_map", "sigma2alpha")


@torch.jit.script
def shrgb2rgb(sh_rgb: torch.Tensor, abs_light: torch.Tensor, white_bkgd: bool) -> torch.Tensor:
    # Accumulated color over the samples, ignoring background
    rgb = torch.sigmoid(sh_rgb)  # [batch, n_intrs-1, 3]
    rgb_map: torch.Tensor = (abs_light.unsqueeze(-1) * rgb).sum(dim=-2)  # [batch, 3]

    if white_bkgd:
        acc_map = abs_light.sum(-1)    # [batch]
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map


@torch.jit.script
def depth_map(abs_light: torch.Tensor, intersections: torch.Tensor) -> torch.Tensor:
    with torch.autograd.no_grad():  # Depth & Inverse Depth-map
        # Weighted average of depths by contribution to final color
        depth: torch.Tensor = (abs_light * intersections[..., :-1]).sum(dim=-1)
        return depth


@torch.jit.script
def sigma2alpha(sigma: torch.Tensor, intersections: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    alpha: torch.Tensor = 1 - torch.exp(-sigma * dists)            # alpha: [batch, n_intrs-1]

    # the absolute amount of light that gets stuck in each voxel
    # This quantity can be used to threshold the intersections which must be processed (only if
    # abs_light > threshold). Often the variable is called 'weights'
    cum_light = torch.cat((torch.ones(sigma.shape[0], 1, dtype=sigma.dtype, device=sigma.device),
                           torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)), dim=-1)  # [batch, n_intrs-1]
    abs_light = alpha * cum_light  # [batch, n_intersections - 1]
    return alpha, abs_light
