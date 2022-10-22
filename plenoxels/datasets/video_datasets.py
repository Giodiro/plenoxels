import glob
import json
import logging as log
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper
from .ray_utils import (
    gen_pixel_samples, gen_camera_dirs, add_color_bkgd, ndc_rays_blender
)
from .synthetic_nerf_dataset import _get_360_bbox

pil2tensor = torchvision.transforms.ToTensor()
tensor2pil = torchvision.transforms.ToPILImage()


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]
    OPENGL_CAMERA: bool = True

    def __init__(self,
                 datadir: str,
                 split: str,
                 color_bkgd_aug: str = "white",
                 batch_size: Optional[int] = None,
                 generator: Optional[torch.random.Generator] = None,
                 downsample: float = 1.0,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 ist: bool = False
                 ):
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.near = 2.0
        self.far = 6.0
        if isg and ist:
            raise ValueError("isg and ist weighting cannot both be True.")
        self.isg = isg
        self.ist = ist
        self.training = (batch_size is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.use_keyframes = max_tsteps is not None

        images, camtoworlds, intrinsics, timestamps, median_images = load_360video_frames(
            datadir, split, self.max_cameras, self.max_tsteps, downsample=downsample)

        gamma = 2e-2
        if self.use_keyframes:
            gamma = 1e-3
        self.isg_weights = None
        if self.isg:
            self.isg_weights = (
                dynerf_isg_weight(images, median_images, gamma)  # [num_cam, num_t, h, w]
                .reshape(-1)
            )
            # Normalize into a probability distribution, to speed up sampling
            self.isg_weights.div_(torch.sum(self.isg_weights))
        self.ist_weights = None
        if self.ist:
            self.ist_weights = dynerf_ist_weight(images, num_cameras=median_images.shape[0]).reshape(-1)
            # Normalize into a probability distribution, to speed up sampling
            self.ist_weights.div_(torch.sum(self.ist_weights))

        images = (images * 255).to(torch.uint8)
        camtoworlds = camtoworlds.to(torch.float32)
        self.timestamps = timestamps.to(torch.float32)[:, None]
        self.len_time = 25#int(torch.amax(self.timestamps).item())
        super().__init__(datadir=datadir,
                         split=split,
                         batch_size=batch_size,
                         is_ndc=False,
                         scene_bbox=_get_360_bbox(datadir),
                         generator=generator,
                         intrinsics=intrinsics,
                         images=images,
                         camtoworlds=camtoworlds)

        log.info(f"Video360Dataset - Loaded {self.split} set from {self.datadir}: "
                 f"{self.images.shape[0]} images of size {self.img_h}x{self.img_w} and "
                 f"{self.images.shape[-1]} channels. "
                 f"{len(torch.unique(timestamps))} timestamps up to time={self.len_time}. "
                 f"ISG={self.isg} - IST={self.ist}. {intrinsics}")

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.batch_size
        if self.isg and self.training:
            image_id, x, y = gen_pixel_samples_weighted(
                num_rays, self.isg_weights, self.intrinsics, self.generator)
        elif self.ist and self.training:
            image_id, x, y = gen_pixel_samples_weighted(
                num_rays, self.ist_weights, self.intrinsics, self.generator)
        else:
            image_id, x, y = gen_pixel_samples(
                self.training, self.images, index, num_rays, self.intrinsics, self.generator)

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)   this converts to f32
        c2w = self.camtoworlds[image_id]            # (num_rays, 3, 4)
        ts = self.timestamps[image_id]              # (num_rays or 1, )
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
            "rgba": rgba,        # [h, w, 4] or [num_rays, 4]
            "rays_o": origins,   # [h, w, 3] or [num_rays, 3]
            "rays_d": viewdirs,  # [h, w, 3] or [num_rays, 3]
            "timestamps": ts     # [h * w]   or [num_rays]
        }

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays_o, rays_d, timestamps = data["rgba"], data["rays_o"], data["rays_d"], data["timestamps"]
        pixels, color_bkgd = add_color_bkgd(rgba, self.color_bkgd_aug, self.training)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays_o": rays_o,  # [n_rays,] or [h, w]
            "rays_d": rays_d,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            "timestamps": timestamps,  # [n_rays,] or [h*w,]
            **{k: v for k, v in data.items() if k not in {"rgba", "rays_o", "rays_d", "timestamps"}},
        }


def gen_pixel_samples_weighted(num_samples: int, weights: torch.Tensor, intrinsics, generator=None):
    # Sample from isg_weights. rids between [0, N*H*W)
    if len(weights) >= 1677721:  # 2^24 is the max for torch.multinomial
        subset = torch.from_numpy(np.random.choice(len(weights), size=1677721)).to(device=weights.device)
        samples = torch.multinomial(
            input=weights[subset], num_samples=num_samples, generator=generator)
        rids = subset[samples]
    else:
        rids = torch.zeros((num_samples,), dtype=torch.int64, device=weights.device)
        torch.multinomial(
            input=weights,
            num_samples=num_samples,
            generator=generator,
            out=rids)
    hw = intrinsics.height * intrinsics.width
    image_id = torch.div(rids, hw, rounding_mode='floor')
    x = torch.remainder(rids, hw).div(intrinsics.height, rounding_mode='floor')
    y = torch.remainder(rids, hw).remainder(intrinsics.height)
    return image_id, x, y


def _parse_360_file_path(fp):
    timestamp = int(fp.split('t')[-1].split('_')[0])
    pose_id = int(fp.split('r')[-1])
    return timestamp, pose_id


def load_360video_frames(datadir,
                         split,
                         max_cameras: int,
                         max_tsteps: int,
                         downsample: float,
                         ) -> Tuple[torch.Tensor, torch.Tensor, Intrinsics, torch.Tensor, torch.Tensor]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as f:
        meta = json.load(f)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    for frame in frames:
        timestamp, pose_id = _parse_360_file_path(frame['file_path'])
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    # Subsampling timestamps: always include first and last!
    num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
    subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
    timestamps = set(timestamps[::subsample_time])
    log.info(f"Selected subset of timestamps: {timestamps}")

    num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
    subsample_poses = int(round(len(pose_ids) / num_poses))
    pose_ids = set(pose_ids[::subsample_poses])

    log_txt = f"Subsampling {split}: "
    if subsample_time > 1:
        log_txt += f"time (1 every {subsample_time}) "
    if subsample_poses > 1:
        log_txt += f"poses (1 every {subsample_poses})"
    log.info(log_txt)

    sub_frames = []
    camera_ids = []
    for frame in frames:
        timestamp, pose_id = _parse_360_file_path(frame['file_path'])
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
            camera_ids.append(pose_id)
    # sort sub_frames by camera_id
    sub_frames = list(zip(*sorted(zip(sub_frames, camera_ids), key=lambda tup: tup[1])))[0]
    num_frames_per_cam = len(sub_frames) // len(np.unique(camera_ids))

    # Load images + poses from frames
    img_poses = parallel_load_images(
        dset_type="synthetic", tqdm_title=f'Loading {split} data', num_images=len(sub_frames),
        frames=sub_frames, data_dir=datadir,
        out_h=None, out_w=None, downsample=downsample, resolution=(None, None))
    images, camtoworlds = zip(*img_poses)
    images = torch.stack(images, 0).float()  # [N, H, W, 3/4]
    camtoworlds = torch.stack(camtoworlds, 0).float()  # [N, ????]

    # Figure out median image per camera
    median_imgs = (
        images.view(num_frames_per_cam, -1, *images.shape[1:])
              .median(0).values
    )  # [num_cams, H, W, 3/4]

    # Load intrinsics
    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / math.tan(0.5 * camera_angle_x)
    intrinsics = Intrinsics(
        width=800, height=800, focal_x=focal,
        focal_y=focal, center_x=800 / 2, center_y=800 / 2)
    intrinsics.scale(1 / downsample)

    # Load timestamps
    timestamps = [_parse_360_file_path(frame['file_path'])[0] for frame in sub_frames]
    timestamps = torch.tensor(timestamps, dtype=torch.int32)  # [N]
    return images, camtoworlds, intrinsics, timestamps, median_imgs


def load_llffvideo_frames(datadir, split, downsample: float, keyframes_take_each: int, near_scaling: float):
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))  # [n_cameras]
    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0])
    else:
        split_ids = np.array([0])
    poses = torch.from_numpy(poses[split_ids])
    videopaths = videopaths[split_ids].tolist()

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each,
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return imgs, poses, intrinsics, timestamps, median_imgs


class VideoLLFFDataset(BaseDataset):
    """This version uses normalized device coordinates, as in LLFF, for forward-facing videos
    """
    len_time: int
    timestamps: Optional[torch.Tensor]
    subsample_time: float

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 generator: Optional[torch.random.Generator] = None,
                 downsample: float = 1.0,
                 keyframes: bool = True,
                 isg: bool = False,
                 ist: bool = False):
        if isg and ist:
            raise ValueError("isg and ist weighting cannot both be True.")
        self.isg = isg
        self.ist = ist
        self.downsample = downsample
        self.near_far = [0.0, 1.0]
        self.keyframes = keyframes
        self.training = split == 'train'
        if self.keyframes:
            self.keyframes_take_each = 30
        else:
            self.keyframes_take_each = 1

        images, camtoworlds, intrinsics, timestamps, median_images = load_llffvideo_frames(
            datadir=datadir, split=split, keyframes_take_each=self.keyframes_take_each,
            downsample=downsample, near_scaling=1.0)
        gamma = 2e-2
        if self.keyframes:
            gamma = 1e-3
        self.isg_weights = None
        if self.isg:
            self.isg_weights = (
                dynerf_isg_weight(images, median_images, gamma)  # [num_cam, num_t, h, w]
                .reshape(-1)
            )
            # Normalize into a probability distribution, to speed up sampling
            self.isg_weights.div_(torch.sum(self.isg_weights))
        self.ist_weights = None
        if self.ist:
            self.ist_weights = dynerf_ist_weight(images, num_cameras=median_images.shape[0]).reshape(-1)
            # Normalize into a probability distribution, to speed up sampling
            self.ist_weights.div_(torch.sum(self.ist_weights))

        images = (images * 255).to(torch.uint8)
        super().__init__(datadir=datadir,
                         split=split,
                         batch_size=batch_size,
                         is_ndc=True,
                         scene_bbox=torch.tensor([[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]]),
                         generator=generator,
                         intrinsics=intrinsics,
                         images=images,
                         camtoworlds=camtoworlds)
        self.timestamps = timestamps
        self.len_time = torch.amax(self.timestamps).item()
        # self.len_time = 300  # This is true for the 10-second sequences from DyNerf

        log.info(f"VideoLLFFDataset - Loaded {self.split} set from {self.datadir}: "
                 f"{images.shape[0]} images of shape {images.shape[1]}x{images.shape[2]} with {images.shape[3]} "
                 f"channels. {len(torch.unique(timestamps))} timestamps up to time={self.len_time}. "
                 f"ISG={self.isg} - IST={self.ist}. {intrinsics}")

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.batch_size
        if self.isg and self.training:
            image_id, x, y = gen_pixel_samples_weighted(
                num_rays, self.isg_weights, self.intrinsics, self.generator)
        elif self.ist and self.training:
            image_id, x, y = gen_pixel_samples_weighted(
                num_rays, self.ist_weights, self.intrinsics, self.generator)
        else:
            image_id, x, y = gen_pixel_samples(
                self.training, self.images, index, num_rays, self.intrinsics, self.generator)

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)   this converts to f32
        c2w = self.camtoworlds[image_id]            # (num_rays, 3, 4)
        ts = self.timestamps[image_id]              # (num_rays or 1, )
        camera_dirs = gen_camera_dirs(
            x, y, self.intrinsics, self.OPENGL_CAMERA)  # [num_rays, 3]
        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions
        origins, viewdirs = ndc_rays_blender(
            intrinsics=self.intrinsics, near=1.0, rays_o=origins, rays_d=viewdirs)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.intrinsics.height, self.intrinsics.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.intrinsics.height, self.intrinsics.width, 3))
            rgba = torch.reshape(rgba, (self.intrinsics.height, self.intrinsics.width, 4))
            ts = ts.repeat(ts, self.intrinsics.height * self.intrinsics.width)  # (num_rays)

        return {
            "rgba": rgba,        # [h, w, 4] or [num_rays, 4]
            "rays_o": origins,   # [h, w, 3] or [num_rays, 3]
            "rays_d": viewdirs,  # [h, w, 3] or [num_rays, 3]
            "timestamps": ts     # [h * w]   or [num_rays]
        }

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays_o, rays_d, timestamps = data["rgba"], data["rays_o"], data["rays_d"], data["timestamps"]
        return {
            "pixels": rgba,  # [n_rays, 3] or [h, w, 3]
            "rays_o": rays_o,  # [n_rays,] or [h, w]
            "rays_d": rays_d,  # [n_rays,] or [h, w]
            "timestamps": timestamps,  # [n_rays,] or [h*w,]
            "color_bkgd": None,  # [3,]
            **{k: v for k, v in data.items() if k not in {"rgba", "rays_o", "rays_d", "timestamps"}},
        }


def dynerf_isg_weight(imgs, median_imgs, gamma):
    """
    Calculates a weighting for every ray pointing to pixels in `imgs`.
    :param imgs:
        torch.Tensor[num_cameras * num_tsteps, h, w, 3]
    :param median_imgs:
        torch.Tensor[num_cameras, h, w, 3]
    :param gamma:
        A scalar
    :return:
        psidiff: valid probabilities in [0, 1]. a tensor of shape [num_cameras, num_tsteps, h, w].
    """
    num_cameras, h, w, c = median_imgs.shape
    differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_tsteps, h, w, 3]
    squarediff = torch.square_(differences)
    psidiff = squarediff / (squarediff + gamma**2)
    psidiff = (1 / 3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_tsteps, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


def dynerf_ist_weight(imgs, num_cameras, alpha=0.1):
    """
    :param imgs:
        torch.Tensor[num_cameras * num_tsteps, h, w, 3]
    :param num_cameras:
        int
    :param alpha:
    :return:
        torch.Tensor[num_cameras, num_tsteps, h, 2]
    """
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_tsteps, h, w, 3]
    shift_left = frames[:, 1:, ...]  # [num_cameras, num_tsteps - 1, h, w, 3]
    shift_right = frames[:, :-1, ...]  # [num_cameras, num_tsteps - 1, h, w, 3]
    shift_right = torch.cat(
        [torch.zeros(num_cameras, 1, h, w, c), shift_right],
        dim=1)  # [num_cameras, num_tsteps, h, w, 3]
    shift_left = torch.cat([shift_left, torch.zeros(num_cameras, 1, h, w, c)], dim=1)  # [num_cameras, num_tsteps, h, w, 3]
    left_difference = torch.abs(frames - shift_left)
    right_difference = torch.abs(frames - shift_right)
    difference = torch.mean(0.5 * (left_difference + right_difference), dim=-1)  # [num_cameras, num_tsteps, h, w]
    difference = torch.clamp(difference, min=alpha)
    return difference
