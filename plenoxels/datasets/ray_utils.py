from typing import Tuple, Optional

import torch

from .intrinsics import Intrinsics

__all__ = (
    "get_ray_directions",
    "get_ray_directions_blender",
    "get_rays",
    "ndc_rays_blender",
)


def create_meshgrid(height: int, width: int, normalized_coordinates: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    # generate grid by stacking coordinates
    yy, xx = torch.meshgrid([ys, xs], indexing="ij")  # both HxW
    return xx, yy


def get_ray_directions(intrinsics: Intrinsics):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    xx, yy = create_meshgrid(intrinsics.height, intrinsics.width, normalized_coordinates=False)

    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = torch.stack([
        (xx - intrinsics.center_x) / intrinsics.focal_x,
        (yy - intrinsics.center_y) / intrinsics.focal_y,
        torch.ones_like(xx)
    ], -1)  # (H, W, 3)

    return directions


def get_ray_directions_blender(intrinsics: Intrinsics):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    xx, yy = create_meshgrid(intrinsics.height, intrinsics.width, normalized_coordinates=False)

    directions = torch.stack([
        (xx - intrinsics.center_x) / intrinsics.focal_x,
        -(yy - intrinsics.center_y) / intrinsics.focal_y,
        -torch.ones_like(xx)
    ], -1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def ndc_rays_blender(intrinsics: Intrinsics, near: float, rays_o: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    ndc_coef_x = - (2 * intrinsics.focal_x) / intrinsics.width
    ndc_coef_y = - (2 * intrinsics.focal_y) / intrinsics.height
    o0 = ndc_coef_x * rays_o[..., 0] / rays_o[..., 2]
    o1 = ndc_coef_y * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = ndc_coef_x * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = ndc_coef_y * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
