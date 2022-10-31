import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any

import numpy as np
import torch

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper
from .ray_utils import gen_camera_dirs, ndc_rays_blender
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
    get_360_bbox
)


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 is_contracted: bool = False,
                 is_ndc: bool = False):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        self.lookup_time = False
        self.per_cam_near_fars = None
        dset_type = None
        if is_contracted and is_ndc:
            raise ValueError("Options 'is_contracted' and 'is_ndc' are exclusive.")
        if "lego" in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        if dset_type == "llff":
            per_cam_poses, self.per_cam_near_fars, intrinsics, videopaths = load_llffvideo_poses(
                datadir, downsample=self.downsample, split=split, near_scaling=1.0)
            poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics, split=split,
                keyframes=keyframes, keyframes_take_each=30)
            self.poses = poses.float()
            self.per_cam_near_fars = self.per_cam_near_fars.float()
            print(f'per_cam_near_fars is {self.per_cam_near_fars}')
        elif dset_type == "synthetic":
            frames, transform = load_360video_frames(datadir, split, max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
            imgs, self.poses = load_360_images(frames, datadir, split, self.downsample)
            self.median_imgs = calc_360_camera_medians(frames, imgs)
            timestamps = [parse_360_file_path(frame['file_path'])[0] for frame in frames]
            timestamps = torch.tensor(timestamps, dtype=torch.int32)
            intrinsics = load_360_intrinsics(transform, imgs, self.downsample)
        else:
            raise ValueError(datadir)


        imgs = (imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        else:
            imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])
        isg_weights = None
        if self.isg:
            t_s = time.time()
            gamma = 1e-3 if self.keyframes else 2e-2
            isg_weights = dynerf_isg_weight(
                imgs.view(-1, intrinsics.height, intrinsics.width, imgs.shape[-1]),
                self.median_imgs,
                gamma)
            # Normalize into a probability distribution, to speed up sampling
            isg_weights = (isg_weights.reshape(-1) / torch.sum(isg_weights))
            t_e = time.time()
            log.info(f"Computed {isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

        super().__init__(datadir=datadir,
                         split=split,
                         batch_size=batch_size,
                         is_ndc=is_ndc,
                         is_contracted=is_contracted,
                         scene_bbox=get_360_bbox(datadir, is_contracted=is_contracted),
                         rays_o=None,
                         rays_d=None,
                         intrinsics=intrinsics,
                         imgs=imgs,
                         sampling_weights=isg_weights
                         )

        if dset_type == "synthetic":
            self.len_time = torch.amax(timestamps).item()
        elif dset_type == "llff":
            self.len_time = 299
        if self.split == 'train':
            self.timestamps = timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        else:
            self.timestamps = timestamps

        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc} - Loaded {self.split} set from {self.datadir}: "
                 f"{self.poses.shape[0]} images of size {self.img_h}x{self.img_w} and "
                 f"{imgs.shape[-1]} channels. "
                 f"{len(torch.unique(timestamps))} timestamps up to time={self.len_time}. "
                 f"ISG={self.isg} - IST={self.ist} - "
                 f"{intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        t_s = time.time()
        gamma = 1e-3 if self.keyframes else 2e-2
        isg_weights = dynerf_isg_weight(
            self.imgs.view(-1, self.img_h, self.img_w, self.imgs.shape[-1]),
            self.median_imgs, gamma)
        # Normalize into a probability distribution, to speed up sampling
        isg_weights = (isg_weights.reshape(-1) / torch.sum(isg_weights))
        self.sampling_weights = isg_weights
        t_e = time.time()
        log.info(f"Enabled ISG weights (computed in {t_e - t_s:.2f}s).")

    def switch_isg2ist(self):
        del self.sampling_weights
        self.isg = False
        self.ist = True
        t_s = time.time()
        ist_weights = dynerf_ist_weight(
            self.imgs.view(-1, self.img_h, self.img_w, self.imgs.shape[-1]),
            num_cameras=self.median_imgs.shape[0])
        # Normalize into a probability distribution, to speed up sampling
        ist_weights = (ist_weights.reshape(-1) / torch.sum(ist_weights))
        self.sampling_weights = ist_weights
        t_e = time.time()
        log.info(f"Switched from ISG to IST weights (computed in {t_e - t_s:.2f}s).")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)
            image_id = torch.div(index, h * w, rounding_mode='floor')
            y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
            x = torch.remainder(index, h * w).remainder(w)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(w, device=dev),
                torch.arange(h, device=dev),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
        near_fars = None
        if self.per_cam_near_fars is not None:
            if self.split == 'train':
                num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
                camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # [num_rays]
                near_fars = self.per_cam_near_fars[camera_id, :]
            else:
                near_fars = self.per_cam_near_fars  # Only one test camera

        rgba = self.imgs[index] / 255.0  # (num_rays, 4)   this converts to f32
        c2w = self.poses[image_id]       # (num_rays, 3, 4)
        ts = self.timestamps[index]      # (num_rays or 1, )
        camera_dirs = gen_camera_dirs(
            x, y, self.intrinsics, True)  # [num_rays, 3]

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        if self.is_ndc:
            origins, directions = ndc_rays_blender(
                intrinsics=self.intrinsics, near=1.0, rays_o=origins, rays_d=directions)
        return {
            "rays_o": origins.reshape(-1, 3),
            "rays_d": directions.reshape(-1, 3),
            "imgs": rgba.reshape(-1, rgba.shape[-1]),
            "timestamps": ts,
            "near_far": near_fars,
        }


def parse_360_file_path(fp):
    timestamp = int(fp.split('t')[-1].split('_')[0])
    pose_id = int(fp.split('r')[-1])
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: Optional[int]) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    fpath2poseid = defaultdict(list)
    for frame in frames:
        timestamp, pose_id = parse_360_file_path(frame['file_path'])
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
    subsample_poses = int(round(len(pose_ids) / num_poses))
    pose_ids = set(pose_ids[::subsample_poses])

    num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
    subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
    timestamps = set(timestamps[::subsample_time])
    log.info(f"Selected subset of timestamps: {timestamps}")
    sub_frames = []
    for frame in frames:
        timestamp, pose_id = parse_360_file_path(frame['file_path'])
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
    # We need frames to be sorted by pose_id
    sub_frames = sorted(sub_frames, key=lambda f: fpath2poseid[f['file_path']])

    return sub_frames, meta


def calc_360_camera_medians(frames, imgs):
    """
    frames: N
    imgs: [N, H, W, C]
    :return
        median_images [num_poses, H, W, C]
    """
    # imgs are sorted by pose_id. We need to find out how many pose_ids there are,
    # and then reshape and compute medians.
    num_pose_ids = len(np.unique([parse_360_file_path(frame['file_path'])[1] for frame in frames]))
    imgs = imgs.view(num_pose_ids, -1, *imgs.shape[1:])
    median_images = torch.median(imgs, dim=1).values
    return median_images


class VideoLLFFDataset(BaseDataset):
    """This version uses normalized device coordinates, as in LLFF, for forward-facing videos
    """
    len_time: int
    timestamps: Optional[torch.Tensor]
    subsample_time: float

    def __init__(self,
                 datadir,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample=1.0,
                 keyframes=True,
                 isg=False,):
        self.keyframes = keyframes
        # self.is_contracted = True  # Use contraction rather than NDC by default
        self.isg = isg
        self.ist = False
        self.downsample = downsample

        per_cam_poses, per_cam_near_fars, intrinsics, videopaths = load_llffvideo_poses(
            datadir, downsample=self.downsample, split=split, near_scaling=1.0)
        poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
            videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics, split=split,
            keyframes=keyframes, keyframes_take_each=30)

        imgs = (imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        else:
            imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])
        isg_weights = None
        if self.isg:
            t_s = time.time()
            gamma = 1e-3 if self.keyframes else 2e-2
            isg_weights = dynerf_isg_weight(
                imgs.view(-1, intrinsics.height, intrinsics.width, imgs.shape[-1]),
                self.median_imgs,
                gamma)
            # Normalize into a probability distribution, to speed up sampling
            isg_weights = (isg_weights.reshape(-1) / torch.sum(isg_weights))
            t_e = time.time()
            log.info(f"Computed {isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=self.init_bbox(),
                         is_ndc=True,
                        #  is_contracted=self.is_contracted,
                         batch_size=batch_size,
                         imgs=imgs,
                         rays_o=None,
                         rays_d=None,
                         intrinsics=intrinsics,
                         sampling_weights=isg_weights)
        self.len_time = 300  # This is true for the 10-second sequences from DyNerf
        if self.split == 'train':
            self.timestamps = timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        else:
            self.timestamps = timestamps

        log.info(f"VideoLLFFDataset - Loaded {self.split} set from {self.datadir}: "
                 f"{len(poses)} images of shape {self.img_h}x{self.img_w} with {imgs.shape[-1]} "
                 f"channels. {len(torch.unique(timestamps))} timestamps up to "
                 f"time={torch.max(timestamps)}. {intrinsics}")

    def init_bbox(self):
        if self.is_contracted:
            return torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        else:
            return torch.tensor([[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]])

    def switch_isg2ist(self):
        del self.sampling_weights
        self.isg = False
        self.ist = True
        t_s = time.time()
        ist_weights = dynerf_ist_weight(
            self.imgs.view(-1, self.img_h, self.img_w, self.imgs.shape[-1]),
            num_cameras=self.median_imgs.shape[0])
        # Normalize into a probability distribution, to speed up sampling
        ist_weights = (ist_weights.reshape(-1) / torch.sum(ist_weights))
        self.sampling_weights = ist_weights
        t_e = time.time()
        log.info(f"Switched from ISG to IST weights (computed in {t_e - t_s:.2f}s).")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)
            image_id = torch.div(index, h * w, rounding_mode='floor')
            y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
            x = torch.remainder(index, h * w).remainder(w)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(w, device=dev),
                torch.arange(h, device=dev),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        rgba = self.imgs[index] / 255.0  # (num_rays, 4)   this converts to f32
        c2w = self.poses[image_id]       # (num_rays, 3, 4)
        ts = self.timestamps[index]      # (num_rays or 1, )
        camera_dirs = gen_camera_dirs(
            x, y, self.intrinsics, True)  # [num_rays, 3]

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        origins, directions = ndc_rays_blender(
            intrinsics=self.intrinsics, near=1.0, rays_o=origins, rays_d=directions)

        return {
            "rays_o": origins.reshape(-1, 3),
            "rays_d": directions.reshape(-1, 3),
            "imgs": rgba.reshape(-1, rgba.shape[-1]),
            "timestamps": ts,
        }


def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """

    :return:
     - poses: a list with one item per each timestamp, pose combination (note that the poses
        are the same across timestamps).
     - imgs: a tensor of shape [N, H, W, 3] where N=timestamps * num_cameras
     - intrinsics
     - timestamps: a tensor of shape [N] indicating which timestamp each frame belongs to.
     - near_fars: a numpy array of shape [num_cameras, 2]
    """

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
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    return poses, near_fars, intrinsics, videopaths


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each if keyframes else 1,
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    num_cameras, h, w, c = median_imgs.shape
    differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:,shift:,...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:,:-shift,...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff
