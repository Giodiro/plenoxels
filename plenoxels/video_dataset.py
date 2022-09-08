import json
import logging as log
import os
from typing import Optional

import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torchvision.transforms
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm
import glob
import imageio

from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender


# This version uses normalized device coordinates, as in LLFF, for forward-facing videos


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


class VideoDataset(TensorDataset):
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self, datadir, split='train', downsample=1.0, subsample_time=1):
        # subsample_time (in [0,1]) lets you use a random percentage of the frames from each video
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.subsample_time = subsample_time

        # self.aabb = torch.tensor([[-1.5, -1.5, -4], [1.5, 2.0, 1.0]]).cuda()  # This is made up...
        self.aabb = torch.tensor([[-3.0, -3.0, -2.0], [3, 3, 100.0]]).cuda() 
        self.len_time = None
        self.near_fars = None

        self.poses, rgbs, self.intrinsics, self.timestamps = self.load_from_disk()
        self.num_frames = len(self.poses)
        self.rays_o, self.rays_d, self.rgbs = self.init_rays(rgbs)
        # Broadcast timestamps to match rays
        if split == 'train':
            self.timestamps = (torch.ones(len(self.timestamps), self.intrinsics.height, self.intrinsics.width) * self.timestamps[:,None,None]).reshape(-1)
        super().__init__(self.rays_o, self.rays_d, self.rgbs, self.timestamps)

    def load_image(self, img, out_h, out_w):
        img = self.tensor2pil(img)
        img = img.resize((out_w, out_h), Image.LANCZOS)  # PIL has x and y reversed from torch
        img = self.pil2tensor(img)  # [C, H, W]
        img = img.permute(1, 2, 0)  # [H, W, C]
        return img

    def load_from_disk(self):
        poses_bounds = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))  # [n_cameras, 17]
        imgs, poses, timestamps = [], [], []
        videopaths = np.array(glob.glob(os.path.join(self.datadir, '*.mp4')))  # [n_cameras]
        assert len(poses_bounds) == len(videopaths), \
            'Mismatch between number of cameras and number of poses!'
        videopaths.sort()
        len_time = 0
        # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
        if self.split == 'train':
            poses_bounds = poses_bounds[1:]
            videopaths = videopaths[1:]
        # else:
            # poses_bounds = [poses_bounds[0]]
            # videopaths = [videopaths[0]]
            # Eval on all the views just so we can see more pictures
            # keep_idx = (np.arange(len(videopaths)) != 1)
            # poses_bounds = poses_bounds[keep_idx]
            # videopaths = videopaths[keep_idx]

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # [N_cameras, 3, 5]
        self.near_fars = poses_bounds[:, -2:]  # [N_cameras, 2]

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        intrinsics = Intrinsics(width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2,
                                center_y=H / 2)
        intrinsics.scale(1 / self.downsample)

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # [N_images, 3, 4] exclude H, W, focal
        poses, pose_avg = center_poses(poses)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 1.  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        poses[..., 3] /= scale_factor

        for camera_id in tqdm(range(len(videopaths))): 
            cam_video = imageio.get_reader(videopaths[camera_id], 'ffmpeg')
            for frame_idx, frame in enumerate(cam_video):
                if frame_idx > len_time:
                    len_time = frame_idx
                # Decide whether to keep this frame or not
                # if np.random.uniform() > self.subsample_time:
                #     continue
                # Only keep frame zero, for debugging
                if frame_idx > 0:
                    len_time = 300
                    break
                # Do any downsampling on the image
                img = self.load_image(frame, intrinsics.height, intrinsics.width)
                imgs.append(img)
                timestamps.append(frame_idx)
        self.len_time = len_time
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3]
        timestamps = torch.from_numpy(np.array(timestamps))  # [N]
        poses = [torch.from_numpy(pose).float() for pose in poses]  # [N, 3, 4]

        log.info(f"LLFFDataset - Loaded {self.split} set from {self.datadir}: {len(imgs)} "
                 f"images of size {imgs[0].shape}. {intrinsics}")

        return poses, imgs, intrinsics, timestamps

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
        all_rgbs = (imgs
                    .reshape(-1, imgs[0].shape[-1])
                    .to(dtype=torch.float32))  # [n_frames * h * w, C]
        if self.split != 'train':
            num_pixels = self.intrinsics.height * self.intrinsics.width
            all_rays_o = all_rays_o.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rays_d = all_rays_d.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rgbs = all_rgbs.view(-1, num_pixels, all_rgbs.shape[-1])

        return all_rays_o, all_rays_d, all_rgbs
