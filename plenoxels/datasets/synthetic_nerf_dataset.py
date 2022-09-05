import json
import logging as log
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import TensorDataset

from .data_loading import parallel_load_images


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
    cent = center if center is not None else [width / 2, height / 2]
    directions = torch.stack([
        (i - cent[0]) / focal[0],
        (j - cent[1]) / focal[1],
        torch.ones_like(i)
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


class SyntheticNerfDataset(TensorDataset):
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self, datadir, split='train', downsample=1.0, resolution=512, max_frames=None):
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.resolution = (resolution, resolution)
        self.max_frames = max_frames

        self.near_far = [2.0, 6.0]
        # y indexes from top to bottom so flip it
        # camera looks along negative z axis so flip that also
        self.blender2opencv = torch.tensor([
            [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        if "ship" in datadir:
            self.radius = 1.5
        else:
            self.radius = 1.3

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
            imgs, poses = parallel_load_images(
                frames=frames, data_dir=self.datadir, out_h=self.img_h, out_w=self.img_w,
                downsample=self.downsample, resolution=self.resolution,
                tqdm_title=f'Loading {self.split} data')
            self.img_h, self.img_w = imgs[0].shape[:2]
            intrinsics = self.load_intrinsics(meta)
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
        poses = torch.stack(poses, 0)  # [N, ????]
        log.info(f"Loaded {self.split} set from {self.datadir}: {imgs.shape[0]} images of size "
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
