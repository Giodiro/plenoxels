import json
import logging as log
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


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
        image_iter=frames, dset_type="synthetic", data_dir=data_dir,
        out_h=None, out_w=None, downsample=downsample,
        resolution=(None, None), tqdm_title=f'Loading {split} data')
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
        images = images.to(torch.float32)
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
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
            else:
                raise ValueError(self.color_bkgd_aug)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
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

        if self.training:
            image_id = torch.randint(
                0,
                len(self.images),
                size=(num_rays,),
                device=self.images.device,
            )
            x = torch.randint(
                0, self.intrinsics.width, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.intrinsics.height, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.intrinsics.width, device=self.images.device),
                torch.arange(self.intrinsics.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x]  # (num_rays, 4)   this converts to f32
        c2w = self.camtoworlds[image_id]    # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.intrinsics.center_x + 0.5) / self.intrinsics.focal_x,
                    (y - self.intrinsics.center_y + 0.5) / self.intrinsics.focal_y
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

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
