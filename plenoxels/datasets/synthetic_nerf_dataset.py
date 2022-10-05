import json
import logging as log
import os
from typing import Tuple, Optional, Any

import numpy as np
import torch

from .data_loading import parallel_load_images
from .ray_utils import get_rays, get_ray_directions, generate_hemispherical_orbit
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset
from .patchloader import PatchLoader

# y indexes from top to bottom so flip it
# camera looks along negative z axis so flip that also
blender2opencv = torch.tensor([
    [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)


class SyntheticNerfDataset(BaseDataset):
    patchloader: Optional[PatchLoader]

    def __init__(self,
                 datadir,
                 split: str,
                 dset_id: int,
                 batch_size: Optional[int] = None,
                 generator: Optional[torch.random.Generator] = None,
                 downsample: float = 1.0,
                 resolution: Optional[int] = 512,
                 max_frames: Optional[int] = None,
                 patch_size: Optional[int] = 8,
                 extra_views: bool = False,
                 color_bkgd_aug: str = 'white'):

        self.downsample = downsample
        self.resolution = (resolution, resolution)
        self.max_frames = max_frames
        self.patch_size = patch_size
        self.dset_id = dset_id
        self.near_far = [2.0, 6.0]
        self.extra_views = split == 'train' and extra_views
        self.patchloader = None
        self.color_bkgd_aug = color_bkgd_aug

        frames, transform = load_360_frames(datadir, split, self.max_frames)
        imgs, poses = load_360_images(frames, datadir, split, self.downsample, self.resolution)
        intrinsics = load_360_intrinsics(transform, imgs, self.downsample)
        rays_o, rays_d, imgs = create_360_rays(
            imgs, poses, merge_all=split == 'train', intrinsics=intrinsics, is_blender_format=True)
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=get_360_bbox(datadir),
                         is_ndc=False,
                         generator=generator,
                         batch_size=batch_size,
                         imgs=imgs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         intrinsics=intrinsics)

        if self.extra_views:
            extra_poses = generate_hemispherical_orbit(poses, n_frames=120)
            extra_rays_o, extra_rays_d, _ = create_360_rays(
                imgs=None, poses=extra_poses, merge_all=False, intrinsics=intrinsics,
                is_blender_format=True)
            extra_rays_o = extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            extra_rays_d = extra_rays_d.view(-1, self.img_h, self.img_w, 3)
            self.patchloader = PatchLoader(extra_rays_o, extra_rays_d, len_time=None,
                                           batch_size=self.batch_size, patch_size=self.patch_size,
                                           generator=self.generator)
        log.info(f"SyntheticNerfDataset - Loaded {split} set from {datadir}: {len(poses)} images of size "
                 f"{self.img_h}x{self.img_w} and {imgs.shape[-1]} channels. {intrinsics}")

    def __getitem__(self, index):
        out = super().__getitem__(index)
        out["dset_id"] = self.dset_id
        if self.split == 'train' and self.extra_views:
            out.update(self.patchloader[index])

        pixels, alpha = torch.split(out["imgs"], [3, 1], dim=-1)
        if self.split == 'train':
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=pixels.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=pixels.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=pixels.device)
            else:
                raise ValueError('color_bkgd_aug')
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)
        out["imgs"] = pixels * alpha + color_bkgd * (1.0 - alpha)
        out["color_bg"] = color_bkgd

        return out


def get_360_bbox(datadir):
    if "ship" in datadir:
        radius = 1.5
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def create_360_rays(
              imgs,
              poses,
              merge_all: bool,
              intrinsics: Intrinsics,
              is_blender_format: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics)  # [H, W, 3]
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    num_frames = poses.shape[0]

    all_rays_o, all_rays_d = [], []
    for i in range(num_frames):
        pose_opencv = poses[i]
        if is_blender_format:
            pose_opencv = pose_opencv @ blender2opencv
        rays_o, rays_d = get_rays(directions, pose_opencv)  # h*w, 3
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

    all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    if imgs is not None:
        imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)   # [N*H*W, 3/4]
    if not merge_all:
        num_pixels = intrinsics.height * intrinsics.width
        if imgs is not None:
            imgs = imgs.view(num_frames, num_pixels, -1)  # [N, H*W, 3/4]
        all_rays_o = all_rays_o.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
        all_rays_d = all_rays_d.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
    return all_rays_o, all_rays_d, imgs


def load_360_frames(datadir, split, max_frames: int) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as f:
        meta = json.load(f)
        frames = meta['frames']

        # Subsample frames
        tot_frames = len(frames)
        num_frames = min(tot_frames, max_frames or tot_frames)
        if split == 'train' or split == 'test':
            subsample = int(round(tot_frames / num_frames))
            frame_ids = np.arange(tot_frames)[::subsample]
            if subsample > 1:
                log.info(f"Subsampling {split} set to 1 every {subsample} images.")
        else:
            frame_ids = np.arange(num_frames)
        frames = np.take(frames, frame_ids).tolist()
    return frames, meta


def load_360_images(frames, datadir, split, downsample, resolution) -> Tuple[torch.Tensor, torch.Tensor]:
    img_poses = parallel_load_images(
        image_iter=frames, dset_type="synthetic", data_dir=datadir,
        out_h=None, out_w=None, downsample=downsample,
        resolution=resolution, tqdm_title=f'Loading {split} data')
    imgs, poses = zip(*img_poses)
    imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
    poses = torch.stack(poses, 0)  # [N, ????]
    return imgs, poses


def load_360_intrinsics(transform, imgs, downsample) -> Intrinsics:
    height = imgs[0].shape[0]
    width = imgs[0].shape[1]
    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downsample
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downsample
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

    cx = (transform['cx'] / downsample) if 'cx' in transform else (width / 2)
    cy = (transform['cy'] / downsample) if 'cy' in transform else (height / 2)
    return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)
