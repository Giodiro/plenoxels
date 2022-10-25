from typing import Optional, Mapping
import logging as log
import numpy as np
import torch


class RayMarcher():
    def __init__(self,
                 n_intersections: Optional[int] = None,
                 num_sample_multiplier: Optional[int] = None,
                 raymarch_type: str = "fixed",
                 spacing_fn: str = "linear",
                 single_jitter: bool = False,
                 **kwargs):
        if raymarch_type == "fixed":
            assert n_intersections is not None
            log.info(f"Initialized 'fixed' ray-marcher with {n_intersections} intersections")
        elif raymarch_type == "voxel_size":
            assert num_sample_multiplier is not None
            log.info(f"Initialized 'voxel_size' ray-marcher with multiplier {num_sample_multiplier}")
        else:
            raise ValueError(raymarch_type)
        self.raymarch_type = raymarch_type
        self.n_intersections = n_intersections
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

    def calc_n_samples(self, aabb: torch.Tensor, resolution: torch.Tensor) -> int:
        if self.raymarch_type == "fixed":
            return self.n_intersections
        elif self.raymarch_type == "voxel_size":
            # step-size and n-intersections are scaled to artificially increment resolution of model
            step_size = torch.mean((aabb[1] - aabb[0]) / (resolution - 1)) / self.num_sample_multiplier
            n_intersections = int(((aabb[1] - aabb[0]).square().sum().sqrt() / step_size).item()) + 1
            return n_intersections

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
        if not perturb and len(steps.shape) != len(sample_shape):
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
                           resolution: torch.Tensor,
                           perturb: bool = False,
                           is_ndc: bool = False,
                           is_contracted: bool=False,
                           near_far: Optional[torch.Tensor] = None,
                           ) -> Mapping[str, torch.Tensor]:
        dev, dt = rays_o.device, rays_o.dtype

        if is_ndc:
            near = torch.tensor([0.0], device=dev, dtype=dt)
            far = torch.tensor([1.0], device=dev, dtype=dt)
            perturb = False # Don't perturb for ndc or contracted scenes
        if is_contracted:
            if near_far is not None:
                # near_far should have shape [batch, 2]
                near = torch.tensor(near_far[:, 0], device=dev, dtype=dt)  # [batch]
                far = torch.tensor(near_far[:, 1], device=dev, dtype=dt)  # [batch]
            else:
                near = torch.tensor([3.0], device=dev, dtype=dt) 
                far = torch.tensor([100.0], device=dev, dtype=dt)
            dir_norm = torch.linalg.norm(rays_d, dim=1, keepdim=True)
            rays_d = rays_d / dir_norm
            # Note: masking out samples does not work when using scene contraction!
            # Note: sampling is really important! You need to cover all the range where stuff is, but not much extra
            perturb = False # Don't perturb for ndc or contracted scenes
        else:
            inv_rays_d = torch.reciprocal(torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d))
            offsets_pos = (aabb[1] - rays_o) * inv_rays_d  # [batch, 3]
            offsets_neg = (aabb[0] - rays_o) * inv_rays_d  # [batch, 3]
            offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
            offsets_out = torch.maximum(offsets_pos, offsets_neg)
            near = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
            far = torch.amin(offsets_out, dim=-1, keepdim=True)

        n_samples = self.calc_n_samples(aabb, resolution)
        intersections = self.get_samples2(rays_o, near, far, n_samples, perturb)  # [n_rays, n_samples + 1]
        intersection_mids = 0.5 * (intersections[..., :-1] + intersections[..., 1:])  # [n_rays, n_samples]
        deltas = intersections.diff(dim=-1)    # [n_rays, n_samples]
        intersections = intersections[:, :-1]  # [n_rays, n_samples]
        intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections[..., None]  # [n_rays, n_samples, 3]

        if is_contracted:
            # Do the contraction to map Euclidean coordinates into [-2, 2] which is also the aabb for contracted scenes
            # Use infinity norm so we map onto the cube instead of the sphere
            intrs_pts = contract(intrs_pts)

        mask = ((aabb[0] <= intrs_pts) & (intrs_pts <= aabb[1])).all(dim=-1)  # noqa
        # assert torch.min(deltas) > 0, f'min delta is {torch.min(deltas)}, but it should be positive!'
        if torch.min(deltas) < 0:
            print(f'found a negative delta! This happened with near {near} and far {far}')
            mask[deltas < 0] = False
        if is_contracted:
            assert torch.min(mask) > 0, f'mask should all be true when using contraction'
        
        # Normalize rays_d and deltas
        if not is_contracted:  # Contraction pre-normalizes rays_d so we don't need to do it again
            dir_norm = torch.linalg.norm(rays_d, dim=1, keepdim=True)
            rays_d = rays_d / dir_norm
            deltas = deltas * dir_norm
        if is_ndc:
            # Make deltas linear in disparity, as in the NDC derivation (but not in the NeRF code)
            deltas = 1. / (1 - intersections) - 1. / (1 - intersections + deltas)

        n_rays, n_intersections = intersections.shape[0:2]
        ridx = torch.arange(0, n_rays, device=dev)
        ridx = ridx[..., None].repeat(1, n_intersections)#[mask]

        return {
            "intersections": intrs_pts,
            "z_vals": intersections,
            "z_mids": intersection_mids,
            "ridx": ridx,
            "deltas": deltas,
            "rays_d": rays_d,
            "mask": mask,
        }


# Apply the square (L_infinity norm) version of the scene contraction from MipNeRF360
def contract(pts):
    # pts should have shape [n_rays, n_samples, 3]
    norms = torch.linalg.vector_norm(pts, ord=np.inf, dim=-1, keepdim=True).expand(-1, -1, 3)  # [n_rays, n_samples, 3]
    norm_mask = norms > 1
    pts[norm_mask] = (2.0 - 1.0 / norms[norm_mask]) * pts[norm_mask] / norms[norm_mask]
    return pts


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
    # t = torch.linspace(0., 1., num, device=start.device)
    # s = fn(start) * (1. - t) + fn(stop) * t

    # # Apply `inv_fn` and clamp to the range of valid values.
    # return torch.clip(inv_fn(s), torch.minimum(start, stop), torch.maximum(start, stop))

    # Linspace between the curved start and stop values.
    t = torch.linspace(0., 1., num, device=start.device)[None,:]
    s = fn(start[:,None]) * (1. - t) + fn(stop[:,None]) * t

    # Apply `inv_fn` and clamp to the range of valid values.
    return torch.clip(inv_fn(s), torch.broadcast_to(torch.minimum(start, stop)[:,None], s.shape), torch.broadcast_to(torch.maximum(start, stop)[:,None], s.shape))
