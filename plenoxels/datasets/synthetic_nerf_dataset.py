import json
import logging as log
import os
from typing import Tuple, Optional, Any

import numpy as np
import torch
import torch.utils.data

from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset
from .patchloader import PatchLoader


class MultiSceneDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets):
        super(MultiSceneDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'

    def __iter__(self):
        idx = 0
        while True:
            yield self.datasets[idx % len(self.datasets)][0]
            idx += 1


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
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=get_360_bbox(datadir),
                         is_ndc=False,
                         generator=generator,
                         batch_size=batch_size,
                         imgs=imgs,
                         poses=poses,
                         intrinsics=intrinsics)

        log.info(f"SyntheticNerfDataset - Loaded {split} set from {datadir}: {len(poses)} images of size "
                 f"{self.img_h}x{self.img_w} and {imgs.shape[-1]} channels. {intrinsics}")

    def fetch_data(self, index):
        batch_size = self.batch_size
        if self.split == 'train':
            image_id = torch.randint(
                0,
                len(self.imgs),
                size=(batch_size,),
                device=self.imgs.device,
            )
            x = torch.randint(
                0, self.intrinsics.width, size=(batch_size,), device=self.imgs.device
            )
            y = torch.randint(
                0, self.intrinsics.height, size=(batch_size,), device=self.imgs.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.intrinsics.width, device=self.imgs.device),
                torch.arange(self.intrinsics.height, device=self.imgs.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        rays_o, rays_d = create_360_rays_v2(
            x, y, intrinsics=self.intrinsics, poses=self.poses[image_id])
        rgba = self.imgs[image_id, y, x]  # (num_rays, 4)

        if self.split == 'train':
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rgba = rgba.reshape(-1, 4)
        else:
            rays_o = rays_o.reshape(self.intrinsics.width, self.intrinsics.height, 3)
            rays_d = rays_d.reshape(self.intrinsics.width, self.intrinsics.height, 3)
            rgba = rgba.reshape(self.intrinsics.width, self.intrinsics.height, 4)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "rgba": rgba,
        }

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays_o, rays_d = data["rgba"], data["rays_o"], data["rays_d"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.split == 'train':
            if self.color_bkgd_aug == "random":
                bg_color = torch.rand(3, device=self.imgs.device)
            elif self.color_bkgd_aug == "white":
                bg_color = torch.ones(3, device=self.imgs.device)
            elif self.color_bkgd_aug == "black":
                bg_color = torch.zeros(3, device=self.imgs.device)
            else:
                raise ValueError(self.color_bkgd_aug)
        else:
            # just use white during inference
            bg_color = torch.ones(3, device=self.imgs.device)

        pixels = pixels * alpha + bg_color * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays_o": rays_o,  # [n_rays, 3] or [h, w, 3]
            "rays_d": rays_d,  # [n_rays, 3] or [h, w, 3]
            "bg_color": bg_color,  # [3,]
            **{k: v for k, v in data.items() if k not in {"rgba", "rays_o", "rays_d"}},
        }

    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        data["dset_id"] = self.dset_id
        return data


def get_360_bbox(datadir):
    if "ship" in datadir:
        radius = 1.5
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def ray_directions(x: torch.Tensor, y:  torch.Tensor, intrinsics: Intrinsics, opengl_camera: bool) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        height, width, focal: image height, width and focal length
    Outputs:
        directions: (height, width, 3), the direction of the rays in camera coordinate
    """
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    return torch.stack([
        (x - intrinsics.center_x + 0.5) / intrinsics.focal_x,
        (y - intrinsics.center_y + 0.5) / intrinsics.focal_y * (-1 if opengl_camera else 1),
        torch.ones_like(x) * (-1 if opengl_camera else 1)
    ], -1)  # (H, W, 3)


def create_360_rays_v2(
            x: torch.Tensor,
            y: torch.Tensor,
            poses,
            intrinsics: Intrinsics,
            is_blender_format: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    camera_dirs = ray_directions(x, y, intrinsics, is_blender_format)  # [H, W, 3]
    num_frames = poses.shape[0]

    directions = (camera_dirs[:, None, :] * poses[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(poses[:, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )
    return origins, viewdirs


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
    imgs = torch.stack(imgs, 0).float()  # [N, H, W, 3/4]
    poses = torch.stack(poses, 0).float()  # [N, ????]
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
