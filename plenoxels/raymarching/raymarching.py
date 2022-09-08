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

    def get_samples_ndc(self, rays_o, rays_d, perturb):
        # TODO: These are hardcoded here and in llff_dataset.
        #       We should fetch them from llff_dataset to have a single source of truth
        start, end = 0.0, 1.0
        dev, dt = rays_o.device, rays_o.dtype

        steps = torch.linspace(0, 1.0, self.n_intersections, device=dev)[None]  # [1, num_samples]
        if perturb:
            steps += torch.rand_like(steps) / self.n_intersections
        steps = steps.expand(rays_o.shape[0], self.n_intersections)

        return steps, start

    def get_samples(self, rays_o, rays_d, perturb, aabb):
        dev, dt = rays_o.device, rays_o.dtype
        n_rays = rays_o.shape[0]

        inv_rays_d = torch.reciprocal(torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d))
        offsets_pos = (aabb[1] - rays_o) * inv_rays_d  # [batch, 3]
        offsets_neg = (aabb[0] - rays_o) * inv_rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        offsets_out = torch.maximum(offsets_pos, offsets_neg)
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        end = torch.amin(offsets_out, dim=-1, keepdim=True)

        if self.raymarch_type == "fixed":
            steps = torch.linspace(0, 1.0, self.n_intersections, device=dev)[None]  # [1, num_samples]
            steps = steps.expand((n_rays, self.n_intersections))  # [num_rays, num_samples]
            n_intersections = self.n_intersections
            intersections = start + (end - start) * steps
            if perturb:
                intersections += (torch.rand_like(intersections) - 0.5) * ((end - start) / n_intersections)
        else:
            step_size = torch.mean(aabb[1] - aabb[0]) / (self.sampling_resolution * self.num_sample_multiplier)
            # step-size and n-intersections are scaled to artificially increment resolution of model
            n_intersections = int(math.sqrt(3.) * self.sampling_resolution * self.num_sample_multiplier)
            steps = torch.arange(n_intersections, dtype=dt, device=dev)[None]  # [1, num_samples]
            steps = steps.repeat(n_rays, 1)  # [num_rays, num_samples]
            if perturb:
                # Apply the same random perturbation to each ray.
                steps += torch.rand(n_rays, 1, dtype=dt, device=dev)

            intersections = start + steps * step_size  # [batch, n_intrs]

        return intersections, start

    @torch.autograd.no_grad()
    def get_intersections(self,
                          rays_o,
                          rays_d,
                          aabb: torch.Tensor,
                          perturb: bool = False,
                          is_ndc: bool = False,
                          timestamps=None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev, dt = rays_o.device, rays_o.dtype
        if is_ndc:
            intersections, start = self.get_samples_ndc(rays_o, rays_d, perturb)
        else:
            intersections, start = self.get_samples(rays_o, rays_d, perturb, aabb)
        n_rays = rays_o.shape[0]
        n_intersections = intersections.shape[1]

        deltas = intersections.diff(dim=-1, prepend=torch.zeros(intersections.shape[0], 1, device=dev) + start)

        intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections[..., None]  # [batch, n_intrs, 3]
        mask = ((aabb[0] <= intrs_pts) & (intrs_pts <= aabb[1])).all(dim=-1)  # noqa

        ridx = torch.arange(0, n_rays, device=dev)
        ridx = ridx[..., None].repeat(1, n_intersections)[mask]
        boundary = spc_render.mark_pack_boundaries(ridx)
        deltas = deltas[mask]
        intrs_pts = intrs_pts[mask]

        # Apply mask to timestamps as well TODO: better to return mask than to add other stuff in here.
        if timestamps is not None:
            assert len(timestamps) == len(rays_o)
            timestamps = timestamps[:,None].repeat(1, mask.shape[1])[mask]
            return intrs_pts, ridx, boundary, deltas, timestamps

        return intrs_pts, ridx, boundary, deltas
