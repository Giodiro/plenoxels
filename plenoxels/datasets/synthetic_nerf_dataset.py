import json
import logging as log
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data_loading import parallel_load_images
from .ray_utils import get_rays, get_ray_directions


class SyntheticNerfDataset(TensorDataset):
    def __init__(self, datadir, split='train', downsample=1.0, resolution=512, max_frames=None):
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.resolution = (resolution, resolution)
        self.max_frames = max_frames
        self.is_ndc = False

        self.near_far = [2.0, 6.0]
        # y indexes from top to bottom so flip it
        # camera looks along negative z axis so flip that also
        self.blender2opencv = torch.tensor([
            [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        if "ship" in datadir:
            radius = 1.5
        else:
            radius = 1.3
        self.scene_bbox = torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])

        imgs, self.poses, self.intrinsics = self.load_from_disk()
        self.imgs, self.rays_o, self.rays_d = self.init_rays(imgs)

        super().__init__(self.rays_o, self.rays_d, self.imgs)

    def load_intrinsics(self, transform):
        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.img_w / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.img_h / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.img_w / 2)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.img_h / 2)

        return np.array([fl_x, fl_y, cx, cy])

    def subsample_dataset(self, frames):
        tot_frames = len(frames)

        num_frames = min(tot_frames, self.max_frames or tot_frames)
        if self.split == 'train' or self.split == 'test':
            subsample = int(round(tot_frames / num_frames))
            frame_ids = np.arange(tot_frames)[::subsample]
            if subsample > 1:
                log.info(f"Subsampling {self.split} set to 1 every {subsample} images.")
        else:
            frame_ids = np.arange(num_frames)
        return np.take(frames, frame_ids).tolist()

    def load_from_disk(self):
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            frames = self.subsample_dataset(meta['frames'])
            img_poses = parallel_load_images(
                image_iter=frames, dset_type="synthetic", data_dir=self.datadir,
                out_h=self.img_h, out_w=self.img_w, downsample=self.downsample,
                resolution=self.resolution, tqdm_title=f'Loading {self.split} data')
            imgs, poses = zip(*img_poses)
            self.img_h, self.img_w = imgs[0].shape[:2]
            intrinsics = self.load_intrinsics(meta)
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
        poses = torch.stack(poses, 0)  # [N, ????]
        log.info(f"SyntheticNerfDataset - Loaded {self.split} set from {self.datadir}: {imgs.shape[0]} images of size "
                 f"{imgs.shape[1]}x{imgs.shape[2]} and {imgs.shape[3]} channels. "
                 f"Focal length {intrinsics[0]:.2f}, {intrinsics[1]:.2f}. "
                 f"Center {intrinsics[2]:.2f}, {intrinsics[3]:.2f}")
        return imgs, poses, intrinsics

    def init_rays(self, imgs):
        assert imgs is not None and self.poses is not None and self.intrinsics is not None
        # Low-pass the images at required resolution
        num_frames = imgs.shape[0]
        imgs = imgs.view(-1, imgs.shape[-1])  # [N*H*W, 3]

        directions = get_ray_directions(
            height=self.img_h, width=self.img_w, focal=(self.intrinsics[0], self.intrinsics[1]))  # (h, w, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        rays = []
        for i in range(self.poses.shape[0]):
            pose = self.poses[i]
            pose_opencv = pose @ self.blender2opencv
            rays.append(torch.stack(get_rays(directions, pose_opencv), 1))  # h*w, 2, 3
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
