import torch

from ..core import RenderBuffer, Rays
from ..tracers import PackedRFTracer


class OfflineRenderer():
    def __init__(self, render_batch, **kwargs):
        self.batch_size = render_batch
        self.extra_args = kwargs

    def render(self, tracer: PackedRFTracer, rays: Rays, lod_idx: int, scene_idx: int) -> RenderBuffer:
        with torch.no_grad():
            if self.batch_size > 0:
                rb = RenderBuffer(xyz=None, hit=None, normal=None, shadow=None, dirs=None)
                for ray_pack in rays.split(self.batch_size):
                    rb += tracer(rays=ray_pack, lod_idx=lod_idx, scene_idx=scene_idx, **self.extra_args)
            else:
                rb = tracer(rays=rays, lod_idx=lod_idx, scene_idx=scene_idx, **self.extra_args)

            # Use segmentation
            if rb.normal is not None:
                rb.normal[~rb.hit] = 1.0
            if rb.rgb is not None:
                rb.rgb[~rb.hit] = 1.0
        return rb
