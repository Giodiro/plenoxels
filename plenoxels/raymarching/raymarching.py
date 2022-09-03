from typing import Optional, Tuple
import math
import logging as log

import torch
import kaolin.render.spc as spc_render


class RayMarcher():
    def __init__(self,
                 n_intersections: Optional[int],
                 sampling_resolution: Optional[int] = None,
                 num_sample_multiplier: Optional[int] = None,
                 raymarch_type: str = "",
                 **kwargs):
        if raymarch_type == "fixed":
            assert n_intersections is not None
            log.info(f"Initialized 'fixed' ray-marcher with {n_intersections} intersections")
        elif raymarch_type == "voxel_size":
            assert sampling_resolution is not None and num_sample_multiplier is not None
            log.info(f"Initialized 'voxel_size' ray-marcher with resolution {sampling_resolution}, "
                     f"multiplier {num_sample_multiplier}")
        else:
            raise ValueError(raymarch_type)
        self.raymarch_type = raymarch_type
        self.n_intersections = n_intersections
        self.sampling_resolution = sampling_resolution
        self.num_sample_multiplier = num_sample_multiplier

    @torch.autograd.no_grad()
    def get_intersections(self, rays_o, rays_d, radius: float, perturb: bool = False, timestamps=None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev, dt = rays_o.device, rays_o.dtype
        n_rays = rays_o.shape[0]
        inv_rays_d = torch.reciprocal(torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d))
        offsets_pos = (radius - rays_o) * inv_rays_d  # [batch, 3]
        offsets_neg = (-radius - rays_o) * inv_rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        offsets_out = torch.maximum(offsets_pos, offsets_neg)
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        end = torch.amin(offsets_out, dim=-1, keepdim=True)

        if self.raymarch_type == "fixed":
            steps = torch.linspace(0, 1.0, self.n_intersections, device=dev)[None]  # [1, num_samples]
            steps = steps.expand((n_rays, self.n_intersections))  # [num_rays, num_samples]
            intersections = start + (end - start) * steps
            n_intersections = self.n_intersections
        else:
            step_size = (radius * 2) / (self.sampling_resolution * self.num_sample_multiplier)
            # step-size and n-intersections are scaled to artificially increment resolution of model
            n_intersections = int(math.sqrt(3.) * self.sampling_resolution * self.num_sample_multiplier)
            steps = torch.arange(n_intersections, dtype=dt, device=dev)[None]  # [1, num_samples]
            steps = steps.expand((n_rays, n_intersections))  # [num_rays, num_samples]
            intersections = start + steps * step_size  # [batch, n_intrs]

        deltas = intersections.diff(dim=-1, prepend=torch.zeros(intersections.shape[0], 1, device=dev) + start)

        if perturb:
            sample_dist = (end - start) / n_intersections
            intersections += (torch.rand_like(intersections) - 0.5) * sample_dist

        intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections[..., None]  # [batch, n_intrs, 3]
        mask = ((-radius <= intrs_pts) & (intrs_pts <= radius)).all(dim=-1)
        
        ridx = torch.arange(0, n_rays, device=dev)
        ridx = ridx[..., None].repeat(1, n_intersections)[mask]
        boundary = spc_render.mark_pack_boundaries(ridx)
        deltas = deltas[mask]
        intrs_pts = intrs_pts[mask]

        # Apply mask to timestamps as well
        if timestamps is not None:
            assert len(timestamps) == len(rays_o)
            timestamps = timestamps[:,None].repeat(1, mask.shape[1])[mask]
            return intrs_pts, ridx, boundary, deltas, timestamps

        return intrs_pts, ridx, boundary, deltas
