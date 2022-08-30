import abc
from typing import List

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render


class BaseGrid(nn.Module):
    def __init__(self,
                 num_lods: int,
                 scene_radii: List[float]):
        super(BaseGrid, self).__init__()
        self.num_lods = num_lods
        self.register_buffer('radii', torch.tensor(scene_radii, dtype=torch.float))

    @abc.abstractmethod
    def query_mask(self, coords, lod_idx, scene_idx=0):
        pass

    @abc.abstractmethod
    def interpolate(self, coords, lod_idx, scene_idx=0) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def feature_dim(self, lod_idx=0) -> int:
        pass

    def raymarch(self, rays_o, rays_d, perturb: bool, lod_idx, scene_idx=0, num_samples=64):
        """Samples points along the ray inside the SPC structure.

        Returns:
            (torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor):
                Note that in the return tensors num_intersections is num_rays * num_samples,
                and num_samples is 1.
                - Indices into rays.origins and rays.dirs of shape [num_intersections] (ridx)
                - Sample coordinates of shape [num_intersections, num_samples, 3] (samples)
                - Sample depths of shape [num_intersections*num_samples, 1]
                - Sample depth diffs of shape [num_intersections*num_samples, 1]
                - Boundary tensor which marks the beginning of each variable-sized sample pack of shape [num_intersections*num_samples]
        """
        if lod_idx is None:
            lod_idx = self.num_lods - 1
        num_rays = rays_o.shape[0]
        dev = rays_o.device

        radius = self.radii[scene_idx]
        inv_rays_d = torch.reciprocal(torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d))
        offset_neg = (-radius - rays_o) * inv_rays_d
        offset_pos = (radius - rays_o) * inv_rays_d
        near = torch.amax(torch.minimum(offset_neg, offset_pos), dim=-1, keepdim=True)
        far = torch.amin(torch.maximum(offset_neg, offset_pos), dim=-1, keepdim=True)

        # Samples points along the rays, and then uses the SPC object the filter out samples that don't hit
        # the SPC objects. This is a much more well-spaced-out sampling scheme and will work well for
        # inside-looking-out scenes. The camera near and far planes will have to be adjusted carefully, however.
        # Sample points along 1D line (depth: [num_rays, num_samples])
        depth = (torch.linspace(0, 1.0, num_samples, device=dev)[None])  # [1, num_samples]
        depth = depth.expand((num_rays, num_samples))  # [num_rays, num_samples]
        depth = near + (far - near) * depth

        if perturb:
            sample_dist = (far - near) / num_samples
            depth = depth + (torch.rand_like(depth) - 0.5) * sample_dist

        # Batched generation of samples
        samples = rays_o[:, None, :] + rays_d[:, None, :] * depth[..., None]  # [num_rays, num_samples, 3]

        # Normalize between -1, 1
        samples = samples / radius

        deltas = depth.diff(dim=-1, prepend=near)

        mask = self.query_mask(samples.reshape(-1, 3), lod_idx, scene_idx).reshape(num_rays, num_samples)

        ridx = torch.arange(0, num_rays, device=dev)
        ridx = ridx[..., None].repeat(1, num_samples)[mask]
        boundary = spc_render.mark_pack_boundaries(ridx)

        depth_samples = depth[mask]

        deltas = deltas[mask].reshape(-1, 1)
        samples = samples[mask]

        return ridx, samples, depth_samples, deltas, boundary
