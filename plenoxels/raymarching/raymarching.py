from typing import Optional, Tuple, Mapping
import math
import logging as log

import torch
import kaolin.render.spc as spc_render


class RayMarcher():
    def __init__(self,
                 n_intersections: Optional[int] = None,
                 sampling_resolution: Optional[int] = None,
                 num_sample_multiplier: Optional[int] = None,
                 raymarch_type: str = "fixed",
                 spacing_fn: str = "linear",
                 single_jitter: bool = False,
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
        self.single_jitter = single_jitter

        if spacing_fn is None or spacing_fn == "linear":
            self.spacing_fn = lambda x: x
            self.inv_spacing_fn = self.spacing_fn
        elif spacing_fn == "log":
            self.spacing_fn = torch.log
            self.inv_spacing_fn = torch.exp
        elif spacing_fn == "reciprocal":
            self.spacing_fn = torch.reciprocal
            self.inv_spacing_fn = torch.reciprocal
        else:
            raise ValueError(f"Spacing function {spacing_fn} invalid.")

    def calc_n_samples(self, aabb: torch.Tensor) -> int:
        if self.raymarch_type == "fixed":
            return self.n_intersections
        elif self.raymarch_type == "voxel_size":
            # step-size and n-intersections are scaled to artificially increment resolution of model
            step_size = torch.mean(aabb[1] - aabb[0]) / (self.sampling_resolution * self.num_sample_multiplier)
            n_intersections = int(((aabb[1] - aabb[0]).square().sum().sqrt() / step_size).item()) + 1
            return n_intersections

    def get_step_size(self, aabb: torch.Tensor) -> float:
        if self.raymarch_type == "fixed":
            return (aabb[1] - aabb[0]).square().sum().sqrt() / self.n_intersections
        elif self.raymarch_type == "voxel_size":
            return torch.mean(aabb[1] - aabb[0]) / (self.sampling_resolution * self.num_sample_multiplier)

    def get_samples2(self,
                     rays_o: torch.Tensor,
                     near: torch.Tensor,
                     far: torch.Tensor,
                     n_samples: int,
                     perturb: bool) -> torch.Tensor:
        steps = genspace(near,#[..., None],
                         far,#[..., None],
                         n_samples + 1,
                         fn=self.spacing_fn,
                         inv_fn=self.inv_spacing_fn)  # [n_samples + 1]

        sample_shape = list(rays_o.shape[:-1]) + [n_samples + 1]
        if not perturb:
            steps = steps.expand(sample_shape)  # [n_rays, n_samples + 1]
        else:
            mids = 0.5 * (steps[..., 1:] + steps[..., :-1])  # [n_rays?, n_samples]
            upper = torch.cat((mids, steps[..., -1:]), dim=-1)  # [n_rays?, n_samples + 1]
            lower = torch.cat((steps[..., :1], mids), dim=-1)  # [n_rays?, n_samples + 1]
            if self.single_jitter:  # each ray gets the same perturbation
                step_rand = torch.rand(sample_shape[:-1], device=steps.device)[..., None]  # [n_rays, 1]
            else:
                step_rand = torch.rand(sample_shape, device=steps.device)  # [n_rays, n_samples + 1]
            steps = lower + (upper - lower) * step_rand  # [n_rays, n_samples + 1]
        return steps

    @torch.autograd.no_grad()
    def get_intersections2(self,
                           rays_o: torch.Tensor,
                           rays_d: torch.Tensor,
                           aabb: torch.Tensor,
                           perturb: bool = False,
                           is_ndc: bool = False,
                           ) -> Mapping[str, torch.Tensor]:
        dev, dt = rays_o.device, rays_o.dtype

        if is_ndc:
            near = torch.tensor(0.0, device=dev, dtype=dt)
            far = torch.tensor(1.0, device=dev, dtype=dt)
        else:
            inv_rays_d = torch.reciprocal(torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d))
            offsets_pos = (aabb[1] - rays_o) * inv_rays_d  # [batch, 3]
            offsets_neg = (aabb[0] - rays_o) * inv_rays_d  # [batch, 3]
            offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
            offsets_out = torch.maximum(offsets_pos, offsets_neg)
            near = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
            far = torch.amin(offsets_out, dim=-1, keepdim=True)

        n_samples = self.calc_n_samples(aabb)
        intersections = self.get_samples2(rays_o, near, far, n_samples, perturb)  # [n_rays, n_samples + 1]
        deltas = intersections.diff(dim=-1)    # [n_rays, n_samples]
        intersections = intersections[:, :-1]  # [n_rays, n_samples]
        intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections[..., None]  # [n_rays, n_samples, 3]
        mask = ((aabb[0] <= intrs_pts) & (intrs_pts <= aabb[1])).all(dim=-1)  # noqa

        # Normalize rays_d and deltas
        dir_norm = torch.linalg.norm(rays_d, dim=1, keepdim=True)
        rays_d = rays_d / dir_norm
        deltas = deltas * dir_norm

        n_rays, n_intersections = intersections.shape[0:2]
        ridx = torch.arange(0, n_rays, device=dev)
        ridx = ridx[..., None].repeat(1, n_intersections)[mask]
        boundary = spc_render.mark_pack_boundaries(ridx)

        deltas = deltas[mask]
        intrs_pts = intrs_pts[mask]

        return {
            "intersections": intrs_pts,
            "ridx": ridx,
            "boundary": boundary,
            "deltas": deltas,
            "rays_d": rays_d,
            "mask": mask,
        }


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

    def get_samples(self, rays_o, rays_d, perturb, aabb, single_jitter: bool):
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
                          single_jitter: bool = True,
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

        # Normalize rays_d and deltas
        dir_norm = torch.linalg.norm(rays_d, dim=1, keepdim=True)
        rays_d = rays_d / dir_norm
        deltas = deltas * dir_norm

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

        return intrs_pts, ridx, boundary, deltas, rays_d


def genspace(start, stop, num, fn, inv_fn):
    """A generalization of linspace(), geomspace(), and NeRF's "lindisp".
    Behaves like jnp.linspace(), except it allows an optional function that
    "curves" the values to make the spacing between samples not linear.
    If no `fn` value is specified, genspace() is equivalent to jnp.linspace().
    If fn=jnp.log, genspace() is equivalent to jnp.geomspace().
    If fn=jnp.reciprocal, genspace() is equivalent to NeRF's "lindisp".
    Args:
    start: float tensor. The starting value of each sequence.
    stop: float tensor. The end value of each sequence.
    num: int. The number of samples to generate for each sequence.
    fn: function. A jnp function handle used to curve `start`, `stop`, and the
      intermediate samples.
    Returns:
    A tensor of length `num` spanning [`start`, `stop`], according to `fn`.
    """
    # Linspace between the curved start and stop values.
    t = torch.linspace(0., 1., num, device=start.device)
    s = fn(start) * (1. - t) + fn(stop) * t

    # Apply `inv_fn` and clamp to the range of valid values.
    return torch.clip(inv_fn(s), torch.minimum(start, stop), torch.maximum(start, stop))
