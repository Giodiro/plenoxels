from typing import Optional, Union

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render

from ..models.nerf import NeuralRadianceField
from ..core import RenderBuffer


class PackedRFTracer(nn.Module):
    """Tracer class for sparse (packed) radiance fields.
    """
    def __init__(self, nef: NeuralRadianceField, **kwargs):
        super().__init__()
        self.nef = nef
        self.extra_args = kwargs

    def forward(self,
                rays_o: torch.Tensor,
                rays_d: torch.Tensor,
                lod_idx: Optional[int],
                scene_idx: int,
                num_steps: int,
                bg_color: Optional[Union[torch.Tensor, int]],
                perturb: bool,
                **kwargs) -> RenderBuffer:
        n_rays = rays_o.shape[0]
        # By default, PackedRFTracer will attempt to use the highest level of detail for the ray sampling.
        # This however may not actually do anything; the ray sampling behaviours are often single-LOD
        # and is governed by however the underlying feature grid class uses the BLAS to implement the sampling.
        if lod_idx is None:
            lod_idx = self.nef.grid.num_lods - 1
        if bg_color is None:
            bg_color = 1

        ridx, samples, depths, deltas, boundary = self.nef.grid.raymarch(
            rays_o, rays_d, lod_idx=lod_idx, scene_idx=scene_idx, num_samples=num_steps, perturb=perturb)

        # Check for the base case where the BLAS traversal hits nothing
        if ridx.shape[0] == 0:
            hit = torch.zeros(n_rays, device=ridx.device).bool()
            if isinstance(bg_color, torch.Tensor) and bg_color.shape == (n_rays, 3):
                rgb = bg_color
            else:
                rgb = torch.ones(n_rays, 3, device=ridx.device) * bg_color
            alpha = torch.zeros(n_rays, 1, device=ridx.device)
            depth = torch.zeros(n_rays, 1, device=ridx.device)
            return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=alpha)

        # Get the indices of the ray tensor which correspond to hits
        ridx_hit = ridx[boundary]

        # Compute the color and density for each ray and their samples
        nef_out = self.nef(
            coords=samples, rays_d=rays_d.index_select(0, ridx),
            lod_idx=lod_idx, scene_idx=scene_idx)
        color = nef_out['rgb']
        density = nef_out['density']
        del ridx, rays_o, rays_d

        # Compute optical thickness
        tau = density.reshape(-1, 1) * deltas
        del density, deltas

        # Perform volumetric integration
        ray_colors, transmittance = spc_render.exponential_integration(
            color.reshape(-1, 3), tau, boundary, exclusive=True)

        ray_depth = spc_render.sum_reduce(depths.reshape(-1, 1) * transmittance, boundary)
        depth = torch.zeros(n_rays, 1, device=ray_depth.device)
        depth[ridx_hit.long(), :] = ray_depth

        alpha = spc_render.sum_reduce(transmittance, boundary)
        out_alpha = torch.zeros(n_rays, 1, device=color.device)
        out_alpha[ridx_hit.long()] = alpha
        hit = torch.zeros(n_rays, device=color.device).bool()
        hit[ridx_hit.long()] = alpha[..., 0] > 0.0

        # Populate the background
        rgb = torch.ones(n_rays, 3, device=color.device)
        if isinstance(bg_color, torch.Tensor) and bg_color.shape[0] == n_rays:
            color = alpha * ray_colors + (1.0 - alpha) * bg_color[ridx_hit.long(), :]
        else:
            color = alpha * ray_colors + (1.0 - alpha) * bg_color
        rgb[ridx_hit.long(), :] = color

        return RenderBuffer(depth=depth, hit=hit, rgb=rgb, alpha=out_alpha)
