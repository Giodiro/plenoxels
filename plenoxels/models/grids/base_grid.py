import abc

import torch
import torch.nn as nn
import kaolin.render.spc as spc_render


class BaseGrid(nn.Module):
    def __init__(self,
                 num_lods: int):
        super(BaseGrid, self).__init__()
        self.num_lods = num_lods

    @abc.abstractmethod
    def query_mask(self, coords, lod_idx, scene_idx=0):
        pass

    @abc.abstractmethod
    def interpolate(self, coords, lod_idx, scene_idx=0) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def feature_dim(self, lod_idx=0) -> int:
        pass

    def raymarch(self, rays, lod_idx, scene_idx=0, num_samples=64):
        """Samples points along the ray inside the SPC structure.

        Args:
            rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
            lod_idx (int) : The level of the octree to raytrace. If None, traces the highest level.
            scene_idx (int) : ID of the scene
            num_samples (int) : Number of samples per voxel

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
        num_rays = rays.origins.shape[0]
        dev = rays.origins.device

        # Samples points along the rays, and then uses the SPC object the filter out samples that don't hit
        # the SPC objects. This is a much more well-spaced-out sampling scheme and will work well for
        # inside-looking-out scenes. The camera near and far planes will have to be adjusted carefully, however.
        # Sample points along 1D line (depth: [num_rays, num_samples])
        depth = (
            torch.linspace(0, 1.0, num_samples, device=dev)[None] +
            torch.rand(num_rays, num_samples, device=dev) / num_samples
        ) ** 2

        # Normalize between near and far plane
        depth *= (rays.dist_max - rays.dist_min)
        depth += rays.dist_min

        # Batched generation of samples
        samples = rays.origins[:, None] + rays.dirs[:, None] * depth[..., None]  # [num_rays, num_samples, 3]
        samples = samples / 2.8
        print(f"generating samples in range {samples.min()} -- {samples.max()}")
        deltas = depth.diff(dim=-1, prepend=torch.zeros(num_rays, 1, device=dev) + rays.dist_min)

        mask = self.query_mask(samples.reshape(-1, 3), lod_idx, scene_idx).reshape(num_rays, num_samples)

        ridx = torch.arange(0, num_rays, device=dev)
        ridx = ridx[..., None].repeat(1, num_samples)[mask]
        boundary = spc_render.mark_pack_boundaries(ridx)

        depth_samples = depth[mask]

        deltas = deltas[mask].reshape(-1, 1)
        samples = samples[mask]

        return ridx, samples, depth_samples, deltas, boundary
