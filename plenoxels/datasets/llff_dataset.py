import glob
import logging as log
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions_blender, ndc_rays_blender, get_rays


def normalize(v):
    """Normalize a vector."""
    return v / torch.linalg.norm(v)


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
    x = normalize(torch.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = torch.cross(x, z)  # (3)

    pose_avg = torch.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_homo = torch.eye(4)
    pose_avg_homo[:3] = pose_avg
    last_row = torch.tile(torch.tensor([0, 0, 0, 1], dtype=poses.dtype), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = torch.cat([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = torch.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


class LLFFDataset(TensorDataset):
    def __init__(self, datadir, split='train', downsample=1.0, resolution=512, hold_every: int = 8):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.datadir = datadir
        self.split = split
        self.hold_every = hold_every
        self.downsample = downsample
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.resolution = resolution
        self.is_ndc = True

        self.blender2opencv = torch.eye(4)

        self.near_far = [0.0, 1.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        imgs, poses, intrinsics = self.load_from_disk()
        self.imgs, self.rays_o, self.rays_d = self.init_rays(imgs, poses, intrinsics)
        super().__init__(self.rays_o, self.rays_d, self.imgs)

    def load_poses(self):
        poses_bounds = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))  # (N, 17)
        poses_bounds = torch.from_numpy(poses_bounds)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
        near_fars = poses_bounds[:, -2:]  # (N, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_h, self.img_w = int(H / self.downsample), int(W / self.downsample)
        focal = (focal * self.img_h / H, focal * self.img_w / W)

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = torch.cat([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        poses, pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        near_fars /= scale_factor
        poses[..., 3] /= scale_factor

        return poses, focal

    def load_from_disk(self):
        poses, focal = self.load_poses()

        image_paths = sorted(glob.glob(os.path.join(self.datadir, 'images_4/*')))

        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert poses.shape[0] == len(image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        # Use numpy arange here: torch arange doesn't work with sets properly.
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

        imgs = parallel_load_images(
            [image_paths[i] for i in img_list],
            tqdm_title=f'Loading {self.split} data', dset_type='llff',
            data_dir=self.datadir,
            out_h=self.img_h,
            out_w=self.img_w,
            resolution=self.resolution)
        poses = torch.stack([poses[i] for i in img_list], 0)
        imgs = torch.stack(imgs, 0)
        intrinsics = torch.tensor([focal[1], focal[0]])

        log.info(f"LLFFDataset - Loaded {self.split} set from {self.datadir}: {imgs.shape[0]} "
                 f"images of size {imgs.shape[1]}x{imgs.shape[2]} and {imgs.shape[3]} channels. "
                 f"Focal length {intrinsics[0]:.2f}, {intrinsics[1]:.2f}.")
        return imgs, poses, intrinsics

    def init_rays(self, imgs, poses, intrinsics):
        num_frames = imgs.shape[0]
        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions_blender(
            height=self.img_h, width=self.img_w, focal=(intrinsics[0], intrinsics[1]))  # (H, W, 3)
        rays = []
        for i in range(num_frames):
            rays_o, rays_d = get_rays(directions, poses[i])  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(self.img_h, self.img_w, intrinsics[0], 1.0, rays_o, rays_d)
            rays.append(torch.stack((rays_o, rays_d), 1))  # h*w, 2, 3
        rays = torch.stack(rays)  # n, h*w, 2, 3
        # Merge N, H, W dimensions
        rays_o = rays[:, :, 0, :].reshape(-1, 3)  # [N*H*W, 3]
        rays_o = rays_o.to(dtype=torch.float32).contiguous()
        rays_d = rays[:, :, 1, :].reshape(-1, 3)  # [N*H*W, 3]
        rays_d = rays_d.to(dtype=torch.float32).contiguous()
        if self.split != "train":
            imgs = imgs.view(num_frames, self.img_w * self.img_h, -1)  # [N, H*W, 3/4]
            rays_o = rays_o.view(num_frames, self.img_w * self.img_h, 3)  # [N, H*W, 3]
            rays_d = rays_d.view(num_frames, self.img_w * self.img_h, 3)  # [N, H*W, 3]
        return imgs, rays_o, rays_d
