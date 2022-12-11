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
from .ray_utils import gen_camera_dirs, ndc_rays_blender, generate_spherical_poses
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
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
                 contraction: bool = False,
                 ndc: bool = False):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = 0.9
        self.ndc_far = 2.6
        self.median_imgs = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        if "lego" in datadir or "dnerf" in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        if dset_type == "llff":
            per_cam_poses, per_cam_near_fars, intrinsics, videopaths = load_llffvideo_poses(
                datadir, downsample=self.downsample, split=split, near_scaling=self.near_scaling)
            if split == 'test':
                keyframes = True
            poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics, split=split,
                keyframes=keyframes, keyframes_take_each=30)
            self.poses = poses.float()
            if contraction:
                self.per_cam_near_fars = per_cam_near_fars.float()
            else:
                self.per_cam_near_fars = torch.tensor(
                    [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)
            # These values are tuned for the salmon video
            self.global_translation = torch.tensor([0, 0, 2.])
            self.global_scale = torch.tensor([0.5, 0.6, 1])
            # Normalize timestamps between -1, 1
            timestamps = (timestamps.float() / 299) * 2 - 1
        elif dset_type == "synthetic":
            assert not contraction, "Synthetic video dataset does not work with contraction."
            assert not ndc, "Synthetic video dataset does not work with NDC."
            if split == 'render':
                render_poses = torch.stack([
                    generate_spherical_poses(angle, -30.0, 4.0)
                    for angle in np.linspace(-180, 180, 40 + 1)[:-1]
                ], 0)
                imgs = None
                self.poses = render_poses
                timestamps = torch.linspace(0.0, 1.0, render_poses.shape[0])
                _, transform = load_360video_frames(
                    datadir, 'train', max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                img_h, img_w = 800, 800
            else:
                frames, transform = load_360video_frames(
                    datadir, split, max_cameras=self.max_cameras, max_tsteps=self.max_tsteps)
                imgs, self.poses = load_360_images(frames, datadir, split, self.downsample)
                timestamps = torch.tensor([
                    parse_360_file_path(frame['file_path'])[0] or float(frame['time'])
                    for frame in frames
                ], dtype=torch.float32)
                img_h, img_w = imgs[0].shape[:2]
            if ndc:
                self.per_cam_near_fars = torch.tensor([[0.0, self.ndc_far]])
            else:
                self.per_cam_near_fars = torch.tensor([[2.0, 6.0]])
            if "dnerf" in datadir:
                # dnerf time is between 0, 1. Normalize to -1, 1
                timestamps = timestamps * 2 - 1
            else:
                # lego (our vid) time is like dynerf: between 0, 30.
                timestamps = (timestamps.float() / torch.amax(timestamps)) * 2 - 1
            intrinsics = load_360_intrinsics(
                transform, img_h=img_h, img_w=img_w, downsample=self.downsample)
        else:
            raise ValueError(datadir)

        self.timestamps = timestamps
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if imgs is not None and imgs.dtype != torch.uint8:
            imgs = (imgs * 255).to(torch.uint8)
        if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
            self.median_imgs = (self.median_imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        else:
            imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])

        # ISG/IST weights are computed on 4x subsampled data.
        weights_subsampled = int(4 / downsample)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=get_bbox(datadir, is_contracted=contraction, dset_type=dset_type),
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            imgs=imgs,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )

        self.isg_weights = None
        self.ist_weights = None
        if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
            if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
                self.isg_weights = torch.load(os.path.join(datadir, f"isg_weights.pt"))
                log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")
            else:
                # Precompute ISG weights
                t_s = time.time()
                gamma = 1e-3 if self.keyframes else 2e-2
                self.isg_weights = dynerf_isg_weight(
                    imgs.view(-1, intrinsics.height, intrinsics.width, imgs.shape[-1]),
                    median_imgs=self.median_imgs, gamma=gamma)
                # Normalize into a probability distribution, to speed up sampling
                self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                torch.save(self.isg_weights, os.path.join(datadir, f"isg_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

            if os.path.exists(os.path.join(datadir, f"ist_weights.pt")):
                self.ist_weights = torch.load(os.path.join(datadir, f"ist_weights.pt"))
                log.info(f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")
            else:
                # Precompute IST weights
                t_s = time.time()
                self.ist_weights = dynerf_ist_weight(
                    imgs.view(-1, self.img_h, self.img_w, imgs.shape[-1]),
                    num_cameras=self.median_imgs.shape[0])
                # Normalize into a probability distribution, to speed up sampling
                self.ist_weights = (self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                torch.save(self.ist_weights, os.path.join(datadir, f"ist_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s.")

        if self.isg:
            self.enable_isg()

        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                 f"Images loaded: {self.imgs is not None}. "
                 f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                 f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        originalindex = index
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            if len(index) == self.batch_size:
                # Nothing special to do, either we have a weights_subsampled = 1, or we're not
                # using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                # Split each subsampled index into its 16 components in 2D.
                hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(self.weights_subsampled):
                    for aw in range(self.weights_subsampled):
                        x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(w, device=dev),
                torch.arange(h, device=dev),
                indexing="xy",
            )
            x, y = x.flatten(), y.flatten()
        out = {
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "imgs": None,
        }
        if self.split == 'train':
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])

        c2w = self.poses[image_id]                      # (num_rays, 3, 4)
        camera_dirs = gen_camera_dirs(x, y, self.intrinsics, True)  # (num_rays, 3)
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        if self.is_ndc:
            origins, directions = ndc_rays_blender(
                intrinsics=self.intrinsics, near=1.0, rays_o=origins, rays_d=directions)
        directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)
        out['rays_o'] = origins
        out['rays_d'] = directions

        if self.split != 'train':
            out['bg_color'] = torch.ones((1, 3), dtype=torch.float32, device=origins.device)
        else:
            imgs = out['imgs']
            if imgs.shape[-1] == 4:
                bg_color = torch.rand((1, 3), dtype=torch.float32, device=origins.device)
                imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
            else:
                bg_color = torch.ones((1, 3), dtype=torch.float32, device=origins.device)
            out['imgs'] = imgs
            out['bg_color'] = bg_color

        return out


def get_bbox(datadir, dset_type: str, is_contracted=False):
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def parse_360_file_path(fp):
    timestamp = None
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
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
        if timestamp is None:  # will be None for dnerf
            timestamp = float(frame['time'])
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    if max_cameras is not None:
        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        pose_ids = set(pose_ids[::subsample_poses])

    if max_tsteps is not None:
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
        timestamps = set(timestamps[::subsample_time])
        log.info(f"Selected subset of timestamps: {sorted(timestamps)} of length {len(timestamps)}")

    sub_frames = []
    for frame in frames:
        timestamp, pose_id = parse_360_file_path(frame['file_path'])
        if timestamp is None:
            timestamp = float(frame['time'])
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


def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
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
        # split_ids = np.array([1])  # Try evaluating on a train view
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


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff
