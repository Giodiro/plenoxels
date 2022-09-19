import json
import logging as log
import os
from typing import Tuple, Optional

import numpy as np
import torch

from .data_loading import parallel_load_images
from .ray_utils import get_rays, get_ray_directions, generate_hemispherical_orbit
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


class SyntheticNerfDataset(BaseDataset):
    # y indexes from top to bottom so flip it
    # camera looks along negative z axis so flip that also
    blender2opencv = torch.tensor([
        [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    poses: Optional[torch.Tensor]
    intrinsics: Optional[Intrinsics]
    imgs: Optional[torch.Tensor]
    rays_o: Optional[torch.Tensor]
    rays_d: Optional[torch.Tensor]

    def __init__(self,
                 datadir,
                 split: str,
                 downsample: float = 1.0,
                 resolution: Optional[int] = 512,
                 max_frames: Optional[int] = None,
                 extra_views: bool = False):
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=self.init_bbox(datadir),
                         is_ndc=False)
        self.downsample = downsample
        self.resolution = (resolution, resolution)
        self.max_frames = max_frames
        self.near_far = [2.0, 6.0]

        tensors = self.fetch_data()
        self.set_tensors(*tensors)

        if self.split == 'train' and extra_views:
            self.extra_poses = generate_hemispherical_orbit(self.poses, n_frames=30)
            _, self.extra_rays_o, self.extra_rays_d = self.init_rays(
                imgs=None, poses=self.extra_poses, merge_all=False, is_blender_format=True)
            self.extra_rays_o = self.extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            self.extra_rays_d = self.extra_rays_d.view(-1, self.img_h, self.img_w, 3)

    def fetch_data(self) -> Tuple[torch.Tensor, ...]:
        imgs, self.poses, self.intrinsics = self.load_from_disk()
        self.imgs, self.rays_o, self.rays_d = self.init_rays(
            imgs, self.poses, merge_all=self.split == 'train', is_blender_format=True)
        return self.rays_o, self.rays_d, self.imgs

    def init_bbox(self, datadir):
        if "ship" in datadir:
            radius = 1.5
        else:
            radius = 1.3
        return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])

    @property
    def img_h(self) -> int:
        return self.intrinsics.height

    @property
    def img_w(self) -> int:
        return self.intrinsics.width

    @property
    def num_frames(self) -> int:
        return self.poses.shape[0]

    def load_intrinsics(self, transform, height, width) -> Intrinsics:
        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = width / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = height / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None:
                fl_x = fl_y
            if fl_y is None:
                fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (width / 2)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (height / 2)

        return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)

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

    def load_from_disk(self) -> Tuple[torch.Tensor, torch.Tensor, Intrinsics]:
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            frames = self.subsample_dataset(meta['frames'])
            img_poses = parallel_load_images(
                image_iter=frames, dset_type="synthetic", data_dir=self.datadir,
                out_h=None, out_w=None, downsample=self.downsample,
                resolution=self.resolution, tqdm_title=f'Loading {self.split} data')
            imgs, poses = zip(*img_poses)
            intrinsics = self.load_intrinsics(meta, height=imgs[0].shape[0], width=imgs[0].shape[1])
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
        poses = torch.stack(poses, 0)  # [N, ????]
        log.info(f"SyntheticNerfDataset - Loaded {self.split} set from {self.datadir}: {imgs.shape[0]} images of size "
                 f"{imgs.shape[1]}x{imgs.shape[2]} and {imgs.shape[3]} channels. {intrinsics}")
        return imgs, poses, intrinsics

    def init_rays(self,
                  imgs,
                  poses,
                  merge_all: bool,
                  is_blender_format: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        directions = get_ray_directions(self.intrinsics)  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        num_frames = poses.shape[0]

        all_rays_o, all_rays_d = [], []
        for i in range(num_frames):
            pose_opencv = poses[i]
            if is_blender_format:
                pose_opencv = pose_opencv @ self.blender2opencv
            rays_o, rays_d = get_rays(directions, pose_opencv)  # h*w, 3
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        if imgs is not None:
            imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)   # [N*H*W, 3/4]
        if not merge_all:
            num_pixels = self.intrinsics.height * self.intrinsics.width
            if imgs is not None:
                imgs = imgs.view(num_frames, num_pixels, -1)  # [N, H*W, 3/4]
            all_rays_o = all_rays_o.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
            all_rays_d = all_rays_d.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
        return imgs, all_rays_o, all_rays_d
