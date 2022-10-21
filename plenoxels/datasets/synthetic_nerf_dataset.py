import json
import logging as log
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.data

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .ray_utils import gen_pixel_samples, gen_camera_dirs, add_color_bkgd


def _load_renderings(data_dir: str,
                     split: str,
                     max_frames: int,
                     downsample: float
                     ) -> Tuple[torch.Tensor, torch.Tensor, Intrinsics]:
    """Load images from disk."""
    with open(os.path.join(data_dir, f"transforms_{split}.json"), "r") as fp:
        meta = json.load(fp)

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

    img_poses = parallel_load_images(
        dset_type="synthetic",
        tqdm_title=f'Loading {split} data',
        num_images=len(frames),
        frames=frames,
        data_dir=data_dir,
        out_h=None,
        out_w=None,
        downsample=downsample,
        resolution=(None, None),
    )
    images, camtoworlds = zip(*img_poses)
    images = torch.stack(images, 0).float()  # [N, H, W, 3/4]
    camtoworlds = torch.stack(camtoworlds, 0).float()  # [N, ????]

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / math.tan(0.5 * camera_angle_x)
    intrinsics = Intrinsics(
        width=800, height=800, focal_x=focal,
        focal_y=focal, center_x=800 / 2, center_y=800 / 2)
    intrinsics.scale(1 / downsample)

    return images, camtoworlds, intrinsics


def _get_360_bbox(datadir):
    if "ship" in datadir:
        radius = 1.5
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


class SyntheticNerfDataset(BaseDataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        datadir: str,
        split: str,
        dset_id: int,
        color_bkgd_aug: str = "white",
        batch_size: int = None,
        downsample: float = 1.0,
        max_frames: Optional[int] = None,
        generator: Optional[torch.random.Generator] = None,
    ):
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.dset_id = dset_id
        self.near = self.NEAR
        self.far = self.FAR
        self.downsample = downsample
        self.max_frames = max_frames
        self.training = (batch_size is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        images, camtoworlds, intrinsics = _load_renderings(
            datadir, split=split, max_frames=self.max_frames, downsample=self.downsample
        )
        images = (images * 255).to(torch.uint8)
        camtoworlds = camtoworlds.to(torch.float32)
        super().__init__(
            datadir=datadir,
            scene_bbox=_get_360_bbox(datadir),
            split=split,
            is_ndc=False,
            camtoworlds=camtoworlds,
            intrinsics=intrinsics,
            generator=generator,
            batch_size=batch_size,
            images=images,
        )
        log.info(f"SyntheticNerfDataset - Loaded {split} set from {datadir}: "
                 f"{self.images.shape[0]} images of size {self.images.shape[1]}x{self.images.shape[2]} "
                 f"and {self.images.shape[3]} channels. {self.intrinsics}")

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays_o, rays_d = data["rgba"], data["rays_o"], data["rays_d"]
        pixels, color_bkgd = add_color_bkgd(rgba, self.color_bkgd_aug, self.training)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays_o": rays_o,  # [n_rays,] or [h, w]
            "rays_d": rays_d,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            "dset_id": self.dset_id,
            **{k: v for k, v in data.items() if k not in {"rgba", "rays_o", "rays_d"}},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.batch_size
        image_id, x, y = gen_pixel_samples(
            self.training, self.images, index, num_rays, self.intrinsics)
        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)   this converts to f32
        c2w = self.camtoworlds[image_id]            # (num_rays, 3, 4)
        camera_dirs = gen_camera_dirs(
            x, y, self.intrinsics, self.OPENGL_CAMERA)  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.intrinsics.height, self.intrinsics.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.intrinsics.height, self.intrinsics.width, 3))
            rgba = torch.reshape(rgba, (self.intrinsics.height, self.intrinsics.width, 4))

        return {
            "rgba": rgba,       # [h, w, 4] or [num_rays, 4]
            "rays_o": origins,  # [h, w, 3] or [num_rays, 3]
            "rays_d": viewdirs,
        }
