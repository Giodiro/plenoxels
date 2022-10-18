import glob
import json
import logging as log
import os
from typing import Optional, List, Tuple, Any

import imageio
import numpy as np
import torch
import torchvision.transforms
from PIL import Image

from .llff_dataset import load_llff_poses_helper, create_llff_rays
from .intrinsics import Intrinsics
from .ray_utils import generate_hemispherical_orbit, generate_spiral_path
from .synthetic_nerf_dataset import create_360_rays, load_360_images, load_360_intrinsics, get_360_bbox
from .patchloader import PatchLoader
from ..my_tqdm import tqdm
from .base_dataset import BaseDataset


pil2tensor = torchvision.transforms.ToTensor()
tensor2pil = torchvision.transforms.ToPILImage()


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 generator: Optional[torch.random.Generator] = None,
                 downsample: float = 1.0,
                 resolution: Optional[int] = 512,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 extra_views: bool = False,
                 patch_size: Optional[int] = 8):

        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.resolution = (resolution, resolution)
        self.patch_size = patch_size
        self.near_far = [2.0, 6.0]
        self.extra_views = split == 'train' and extra_views

        frames, transform = load_360video_frames(datadir, split, self.max_cameras, self.max_tsteps)
        imgs, poses = load_360_images(frames, datadir, split, self.downsample, self.resolution)
        intrinsics = load_360_intrinsics(transform, imgs, self.downsample)
        rays_o, rays_d, imgs = create_360_rays(
            imgs, poses, merge_all=split == 'train', intrinsics=intrinsics, is_blender_format=True)
        super().__init__(datadir=datadir,
                         split=split,
                         batch_size=batch_size,
                         is_ndc=False,
                         scene_bbox=get_360_bbox(datadir),
                         generator=generator,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         intrinsics=intrinsics,
                         images=imgs)

        timestamps = [parse_360_file_path(frame['file_path'])[0] for frame in frames]
        timestamps = torch.tensor(timestamps, dtype=torch.int32)
        self.len_time = torch.amax(timestamps).item()
        if self.split == 'train':
            self.timestamps = timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        else:
            self.timestamps = timestamps

        if self.extra_views:
            extra_poses = generate_hemispherical_orbit(poses, n_frames=120)
            extra_rays_o, extra_rays_d, _ = create_360_rays(
                imgs=None, poses=extra_poses, merge_all=False, intrinsics=intrinsics,
                is_blender_format=True)
            extra_rays_o = extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            extra_rays_d = extra_rays_d.view(-1, self.img_h, self.img_w, 3)
            self.patchloader = PatchLoader(extra_rays_o, extra_rays_d, len_time=self.len_time,
                                           batch_size=self.batch_size, patch_size=self.patch_size,
                                           generator=self.generator)

        log.info(f"Video360Dataset - Loaded {self.split} set from {self.datadir}: "
                 f"{poses.shape[0]} images of size {self.img_h}x{self.img_w} and "
                 f"{imgs.shape[-1]} channels. "
                 f"{len(torch.unique(timestamps))} timestamps up to time={torch.max(timestamps)}. "
                 f"{intrinsics}")

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if self.split == 'train':
            idxs = self.get_rand_ids(index)
            out["timestamps"] = self.timestamps[idxs]
            if self.extra_views:
                out.update(self.patchloader[index])
        else:
            out["timestamps"] = self.timestamps[index]
        return out


def parse_360_file_path(fp):
    timestamp = int(fp.split('t')[-1].split('_')[0])
    pose_id = int(fp.split('r')[-1])
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: int) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as f:
        meta = json.load(f)
        frames = meta['frames']

        timestamps = set()
        pose_ids = set()
        for frame in frames:
            timestamp, pose_id = parse_360_file_path(frame['file_path'])
            timestamps.add(timestamp)
            pose_ids.add(pose_id)
        timestamps = sorted(timestamps)
        pose_ids = sorted(pose_ids)

        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(round(len(timestamps) / num_timestamps))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        if subsample_time == 1 and subsample_poses == 1:
            return frames
        timestamps = set(timestamps[::subsample_time])
        pose_ids = set(pose_ids[::subsample_poses])

        log_txt = f"Subsampling {split}: "
        if subsample_time > 1:
            log_txt += f"time (1 every {subsample_time}) "
        if subsample_poses > 1:
            log_txt += f"poses (1 every {subsample_poses})"
        log.info(log_txt)

        sub_frames = []
        for frame in frames:
            timestamp, pose_id = parse_360_file_path(frame['file_path'])
            if timestamp in timestamps and pose_id in pose_ids:
                sub_frames.append(frame)
        return sub_frames, meta


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
                 generator: Optional[torch.random.Generator] = None,
                 downsample=1.0,
                 keyframes=True,
                 isg=False,
                 extra_views: bool = False,
                 patch_size: Optional[int] = 8,):
        """

        :param datadir:
        :param split:
        :param downsample:
        :param subsample_time:
            in [0,1] lets you use a percentage of randomly selected frames from each video
        :param extra_views:
        """
        self.keyframes = keyframes
        self.isg = isg
        self.downsample = downsample
        self.patch_size = patch_size
        self.near_far = [0.0, 1.0]
        self.extra_views = split == 'train' and extra_views
        self.patchloader = None

        per_cam_poses, per_cam_near_fars, intrinsics, videopaths = load_llffvideo_poses(
            datadir, downsample=self.downsample, split=split, near_scaling=1.0)
        poses, imgs, timestamps, median_imgs = load_llffvideo_data(
            videopaths=videopaths, poses=per_cam_poses, intrinsics=intrinsics, split=split,
            keyframes=keyframes)
        gamma = 2e-2
        if keyframes:
            gamma = 1e-3
        self.isg_weights = None
        if self.isg:
            isg_weights = dynerf_isg_weight(imgs, median_imgs, gamma)
            self.isg_weights = isg_weights.reshape(-1)
        poses = poses.float()
        rays_o, rays_d, imgs = create_llff_rays(
            imgs=imgs, poses=poses, intrinsics=intrinsics, merge_all=split == 'train')  # [-1, 3]
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=self.init_bbox(),
                         is_ndc=True,
                         generator=generator,
                         batch_size=batch_size,
                         images=imgs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         intrinsics=intrinsics)
        self.len_time = torch.amax(timestamps).item()
        if self.split == 'train':
            self.timestamps = timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        else:
            self.timestamps = timestamps

        if self.extra_views:
            extra_poses = generate_spiral_path(
                per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=120)
            extra_rays_o, extra_rays_d, _ = create_llff_rays(
                imgs=None, poses=extra_poses, intrinsics=intrinsics, merge_all=False)
            extra_rays_o = extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            extra_rays_d = extra_rays_d.view(-1, self.img_h, self.img_w, 3)
            self.patchloader = PatchLoader(extra_rays_o, extra_rays_d, len_time=self.len_time,
                                           batch_size=self.batch_size, patch_size=self.patch_size,
                                           generator=self.generator)

        log.info(f"VideoLLFFDataset - Loaded {self.split} set from {self.datadir}: "
                 f"{len(poses)} images of shape {self.img_h}x{self.img_w} with {imgs.shape[-1]} "
                 f"channels. {len(torch.unique(timestamps))} timestamps up to "
                 f"time={torch.max(timestamps)}. {intrinsics}")

    def init_bbox(self):
        return torch.tensor([[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]])

    def __getitem__(self, index):
        if self.isg_weights is not None:
            out, idxs = super().__getitem__(index, weights=self.isg_weights)
        else:
            out = super().__getitem__(index, weights=self.isg_weights)
            idxs = None
        if self.split == 'train':
            if idxs is None:
                idxs = self.get_rand_ids(index)
            out["timestamps"] = self.timestamps[idxs]
            if self.extra_views:
                out.update(self.patchloader[index])
        else:
            out["timestamps"] = self.timestamps[index]

        return out


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
                        poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def load_frame(f, out_h, out_w, already_pil=False):
        if not already_pil:
            f = tensor2pil(f)
        f = f.resize((out_w, out_h), Image.LANCZOS)  # PIL has x and y reversed from torch
        f = pil2tensor(f)  # [C, H, W]
        f = f.permute(1, 2, 0)  # [H, W, C]
        return f

    imgs, timestamps = [], []
    all_poses = []
    median_imgs = []
    for camera_id in tqdm(range(len(videopaths)), desc=f"Loading {split} data"):
        cam_video = imageio.get_reader(videopaths[camera_id], 'ffmpeg')
        cam_imgs = []
        for frame_idx, frame in enumerate(cam_video):
            # Decide whether to keep this frame or not
            if keyframes and frame_idx % 30 != 0:
                continue
            # Do any downsampling on the image
            img = load_frame(frame, intrinsics.height, intrinsics.width)
            imgs.append(img)
            cam_imgs.append(img)
            timestamps.append(frame_idx)
            all_poses.append(poses[camera_id])
        # Compute the median image from this camera
        median_imgs.append(median_image(cam_imgs))
    timestamps = torch.tensor(timestamps, dtype=torch.int32)  # [N]
    poses = torch.stack(all_poses, 0)  # [N, 3, 4]
    imgs = torch.stack(imgs, 0)  # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


def median_image(imgs):
    imgs = torch.stack(imgs, -1)
    values, _ = torch.median(imgs, dim=-1)  # [h, w, 3]
    return values


def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    num_cameras, h, w, c = median_imgs.shape
    differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    squarediff = torch.square(differences)
    psidiff = squarediff / (squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]
