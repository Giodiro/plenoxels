import glob
import os
import logging as log

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


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


class LLFFDataset(BaseDataset):
    def __init__(self, datadir, split='train', downsample=4, hold_every=8, resolution=512):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        super().__init__(datadir=datadir,
                         scene_bbox=torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]),
                         split=split,
                         is_ndc=True)
        self.hold_every = hold_every
        self.downsample = 4

        self.near_fars = None
        self.near_far = [0.0, 1.0]

        self.poses, rgbs, self.intrinsics = self.load_from_disk()
        self.num_frames = len(self.poses)
        self.rays_o, self.rays_d, self.rgbs = self.init_rays(rgbs)

        self.set_tensors(self.rays_o, self.rays_d, self.rgbs)

    @property
    def img_w(self):
        return self.intrinsics.width

    @property
    def img_h(self):
        return self.intrinsics.height

    def load_from_disk(self):
        poses_bounds = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))  # (N_images, 17)
        image_paths = sorted(glob.glob(os.path.join(self.datadir, 'images_4/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        intrinsics = Intrinsics(width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2,
                                center_y=H / 2)
        intrinsics.scale(1 / self.downsample)

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        poses, pose_avg = center_poses(poses)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 1.  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        poses[..., 3] /= scale_factor

        # average_pose = average_poses(self.poses)
        # dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = np.arange(0, poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(
            set(np.arange(len(poses))) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        all_rgbs = parallel_load_images(
            [image_paths[i] for i in img_list],
            tqdm_title=f'Loading {self.split} data',
            dset_type='llff',
            data_dir='/',
            out_h=intrinsics.height,
            out_w=intrinsics.width,
            resolution=(None, None)
        )
        poses = [torch.from_numpy(poses[i]).float() for i in img_list]

        log.info(f"LLFFDataset - Loaded {self.split} set from {self.datadir}: {len(all_rgbs)} "
                 f"images of with {all_rgbs[0].shape[-1]} channels. {intrinsics}")

        return poses, all_rgbs, intrinsics

    def init_rays(self, imgs):
        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions_blender(self.intrinsics)  # H, W, 3

        all_rays_o, all_rays_d = [], []
        for i in range(len(self.poses)):
            rays_o, rays_d = get_rays(directions, self.poses[i])  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(
                intrinsics=self.intrinsics, near=1.0, rays_o=rays_o, rays_d=rays_d)
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        all_rgbs = (torch.cat(imgs, 0)
                    .reshape(-1, imgs[0].shape[-1])
                    .to(dtype=torch.float32))  # [n_frames * h * w, C]
        if self.split != 'train':
            num_pixels = self.intrinsics.height * self.intrinsics.width
            all_rays_o = all_rays_o.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rays_d = all_rays_d.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rgbs = all_rgbs.view(-1, num_pixels, all_rgbs.shape[-1])

        return all_rays_o, all_rays_d, all_rgbs
