from typing import Tuple

import torch

__all__ = (
    "get_ray_directions",
    "get_ray_directions_blender",
    "get_rays",
    "ndc_rays_blender",
)


def create_meshgrid(height: int, width: int, normalized_coordinates: bool = True) -> torch.Tensor:
    xs = torch.linspace(0, width - 1, width)
    ys = torch.linspace(0, height - 1, height)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def get_ray_directions(height: int, width: int, focal: Tuple[float, float], center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(height, width, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)  # both 1xHxW
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [height / 2, width / 2]
    directions = torch.stack([
        (i - cent[0]) / focal[0],
        (j - cent[1]) / focal[1],
        torch.ones_like(i)
    ], -1)  # (H, W, 3)

    return directions


def get_ray_directions_blender(height: int, width: int, focal: Tuple[float, float], center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(height, width, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)  # both 1xHxW
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [height / 2, width / 2]
    directions = torch.stack([
        (i - cent[0]) / focal[0],
        -(j - cent[1]) / focal[1],
        -torch.ones_like(i)
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


def ndc_rays_blender(ndc_coefs, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = ndc_coefs[0] * (rays_o[..., 0] / rays_o[..., 2])
    o1 = ndc_coefs[1] * (rays_o[..., 1] / rays_o[..., 2])
    o2 = 1 - 2 * near / rays_o[..., 2]

    d0 = ndc_coefs[0] * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = ndc_coefs[1] * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = 2 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
