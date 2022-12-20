from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .intrinsics import Intrinsics

__all__ = (
    "get_ray_directions",
    "get_ray_directions_blender",
    "get_rays",
    "ndc_rays_blender",
    "center_poses",
    "generate_spiral_path",
    "generate_hemispherical_orbit",
    "gen_camera_dirs",
    "normalize",
    "average_poses",
    "viewmatrix",
)


def create_meshgrid(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    # generate grid by stacking coordinates
    yy, xx = torch.meshgrid([ys, xs], indexing="ij")  # both HxW
    return xx, yy


def get_ray_directions(intrinsics: Intrinsics) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    xx, yy = create_meshgrid(intrinsics.height, intrinsics.width)

    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = torch.stack([
        (xx - intrinsics.center_x) / intrinsics.focal_x,
        (yy - intrinsics.center_y) / intrinsics.focal_y,
        torch.ones_like(xx)
    ], -1)  # (H, W, 3)

    return directions


def get_ray_directions_blender(intrinsics: Intrinsics) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    xx, yy = create_meshgrid(intrinsics.height, intrinsics.width)

    directions = torch.stack([
        (xx - intrinsics.center_x) / intrinsics.focal_x,
        -(yy - intrinsics.center_y) / intrinsics.focal_y,
        -torch.ones_like(xx)
    ], -1)  # (H, W, 3)

    return directions


def get_rays(directions: torch.Tensor, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def ndc_rays_blender(intrinsics: Intrinsics, near: float, rays_o: torch.Tensor,
                     rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([-vec0, vec1, vec2, pos], axis=1)


def generate_spiral_path(poses: np.ndarray,
                         near_fars: np.ndarray,
                         n_frames=120,
                         n_rots=2,
                         zrate=.5,
                         dt=0.75) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.

    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py

    :param poses: [N, 3, 4]
    :param near_fars:
    :param n_frames:
    :param n_rots:
    :param zrate:
    :param dt:
    :return:
    """
    # center pose
    c2w = average_poses(poses)  # [3, 4]

    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = np.min(near_fars) * 0.9, np.max(near_fars) * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 70, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = c2w @ t
        lookat = c2w @ np.array([0, 0, -focal, 1.0])
        z_axis = normalize(position - lookat)
        render_poses.append(viewmatrix(z_axis, up, position))
    return np.stack(render_poses, axis=0)


def generate_hemispherical_orbit(poses: torch.Tensor, n_frames=120):
    """Calculates a render path which orbits around the z-axis.
    Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    """
    origins = poses[:, :3, 3]
    radius = torch.sqrt(torch.mean(torch.sum(origins ** 2, dim=-1)))

    # Assume that z-axis points up towards approximate camera hemisphere
    sin_phi = torch.mean(origins[:, 2], dim=0) / radius
    cos_phi = torch.sqrt(1 - sin_phi ** 2)
    render_poses = []

    up = torch.tensor([0., 0., 1.])
    for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
        camorigin = radius * torch.tensor(
            [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
        render_poses.append(torch.from_numpy(viewmatrix(camorigin, up, camorigin)))

    render_poses = torch.stack(render_poses, dim=0)
    return render_poses


def gen_camera_dirs(x: torch.Tensor, y: torch.Tensor, intrinsics: Intrinsics, opengl_camera: bool):
    return F.pad(
        torch.stack(
            [
                (x - intrinsics.center_x + 0.5) / intrinsics.focal_x,
                (y - intrinsics.center_y + 0.5) / intrinsics.focal_y
                * (-1.0 if opengl_camera else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl_camera else 1.0),
    )  # [num_rays, 3]


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def generate_spherical_poses(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w
