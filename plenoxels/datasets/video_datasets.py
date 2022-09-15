import glob
import json
import logging as log
import os
from typing import Optional, List, Tuple

import imageio
import numpy as np
import torch
import torchvision.transforms
from PIL import Image

from .llff_dataset import LLFFDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .ray_utils import generate_spiral_path, generate_hemispherical_orbit
from .synthetic_nerf_dataset import SyntheticNerfDataset
from ..my_tqdm import tqdm


class Video360Dataset(SyntheticNerfDataset):
    def __init__(self,
                 datadir,
                 split='train',
                 downsample=1.0,
                 resolution=512,
                 max_cameras=None,
                 max_tsteps=None,
                 extra_views: bool = False):
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.timestamps: Optional[torch.Tensor] = None
        super().__init__(datadir=datadir,
                         split=split,
                         downsample=downsample,
                         resolution=resolution,
                         max_frames=None)
        self.len_time = torch.amax(self.timestamps).item()

        if self.split == 'train' and extra_views:
            self.extra_poses = generate_hemispherical_orbit(self.poses, n_frames=30)
            _, self.extra_rays_o, self.extra_rays_d = self.init_rays(
                imgs=None, poses=self.extra_poses, merge_all=False, is_blender_format=False)
            self.extra_rays_o = self.extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            self.extra_rays_d = self.extra_rays_d.view(-1, self.img_h, self.img_w, 3)

    def fetch_data(self) -> Tuple[torch.Tensor, ...]:
        imgs, self.poses, self.intrinsics, timestamps = self.load_from_disk()
        if self.split == 'train':
            self.timestamps = timestamps[:, None, None].repeat(
                1, self.intrinsics.height, self.intrinsics.width).reshape(-1)  # [n_frames * h * w]
        else:
            self.timestamps = timestamps  # TODO: Why no repeat for test split?

        self.imgs, self.rays_o, self.rays_d = self.init_rays(
            imgs, self.poses, merge_all=self.split == 'train', is_blender_format=True)
        return self.rays_o, self.rays_d, self.imgs, self.timestamps

    def parse_file_path(self, fp):
        timestamp = int(fp.split('t')[-1].split('_')[0])
        pose_id = int(fp.split('r')[-1])
        return timestamp, pose_id

    def subsample_dataset(self, frames):
        timestamps = set()
        pose_ids = set()
        for frame in frames:
            timestamp, pose_id = self.parse_file_path(frame['file_path'])
            timestamps.add(timestamp)
            pose_ids.add(pose_id)
        timestamps = sorted(timestamps)
        pose_ids = sorted(pose_ids)

        num_poses = min(len(pose_ids), self.max_cameras or len(pose_ids))
        num_timestamps = min(len(timestamps), self.max_tsteps or len(timestamps))
        subsample_time = int(round(len(timestamps) / num_timestamps))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        if subsample_time == 1 and subsample_poses == 1:
            return frames
        timestamps = set(timestamps[::subsample_time])
        pose_ids = set(pose_ids[::subsample_poses])

        log_txt = f"Subsampling {self.split}: "
        if subsample_time > 1:
            log_txt += f"time (1 every {subsample_time}) "
        if subsample_poses > 1:
            log_txt += f"poses (1 every {subsample_poses})"
        log.info(log_txt)

        sub_frames = []
        for frame in frames:
            timestamp, pose_id = self.parse_file_path(frame['file_path'])
            if timestamp in timestamps and pose_id in pose_ids:
                sub_frames.append(frame)
        return sub_frames

    def load_from_disk(self):
        """
        This si the same as the SyntheticNerf with added loading of timestamps
        """
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            frames = self.subsample_dataset(meta['frames'])
            img_poses = parallel_load_images(
                image_iter=frames, dset_type="synthetic", data_dir=self.datadir,
                out_h=None, out_w=None, downsample=self.downsample,
                resolution=self.resolution, tqdm_title=f'Loading {self.split} data')
            imgs, poses = zip(*img_poses)
            intrinsics = self.load_intrinsics(meta, height=imgs[0].shape[0], width=imgs[0].shape[1])
            # Get timestamps as well
            timestamps = [self.parse_file_path(frame['file_path'])[0] for frame in frames]
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
        poses = torch.stack(poses, 0)  # [N, ????]
        timestamps = torch.tensor(timestamps)  # [N]
        log.info(f"Video360Dataset - Loaded {self.split} set from {self.datadir}: "
                 f"{imgs.shape[0]} images of size {imgs.shape[1]}x{imgs.shape[2]} and "
                 f"{imgs.shape[3]} channels. "
                 f"{len(torch.unique(timestamps))} timestamps up to time={torch.max(timestamps)}. "
                 f"{intrinsics}")
        return imgs, poses, intrinsics, timestamps


class VideoLLFFDataset(LLFFDataset):
    """This version uses normalized device coordinates, as in LLFF, for forward-facing videos
    """
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self,
                 datadir,
                 split='train',
                 downsample=1.0,
                 subsample_time=1,
                 extra_views: bool = False):
        """

        :param datadir:
        :param split:
        :param downsample:
        :param subsample_time:
            in [0,1] lets you use a percentage of randomly selected frames from each video
        :param extra_views:
        """
        self.subsample_time = subsample_time
        self.timestamps: Optional[torch.Tensor] = None
        super().__init__(datadir=datadir,
                         split=split,
                         downsample=downsample)
        self.len_time = torch.amax(self.timestamps).item()

        if self.split == 'train' and extra_views:
            self.extra_poses = generate_spiral_path(np.array(self.poses), self.near_fars)
            self.extra_poses = torch.from_numpy(self.extra_poses)
            self.extra_rays_o, self.extra_rays_d, _ = self.init_rays(
                imgs=None, poses=self.extra_poses, merge_all=False)
            self.extra_rays_o = self.extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            self.extra_rays_d = self.extra_rays_d.view(-1, self.img_h, self.img_w, 3)

    def fetch_data(self) -> Tuple[torch.Tensor, ...]:
        self.poses, imgs, self.intrinsics, timestamps, self.near_fars = self.load_from_disk()

        if self.split == 'train':
            self.timestamps = timestamps[:, None, None].repeat(
                1, self.intrinsics.height, self.intrinsics.width).reshape(-1)  # [n_frames * h * w]
        else:
            self.timestamps = timestamps  # TODO: Why no repeat for test split?

        self.rays_o, self.rays_d, self.imgs = self.init_rays(
            imgs, self.poses, merge_all=self.split == 'train')

        # The data returned from this function will be set as tensors of the class.
        return self.rays_o, self.rays_d, self.imgs, self.timestamps

    def init_bbox(self, datadir):
        return torch.tensor([[-2.0, -2.0, -1.0], [2.0, 2.0, 1.0]])

    def load_image(self, img, out_h, out_w, already_pil=False):
        if not already_pil:
            img = self.tensor2pil(img)
        img = img.resize((out_w, out_h), Image.LANCZOS)  # PIL has x and y reversed from torch
        img = self.pil2tensor(img)  # [C, H, W]
        img = img.permute(1, 2, 0)  # [H, W, C]
        return img

    def load_from_disk(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], Intrinsics, torch.Tensor, np.ndarray]:
        """

        :return:
         - poses: a list with one item per each timestamp, pose combination (note that the poses
            are the same across timestamps).
         - imgs: a tensor of shape [N, H, W, 3] where N=timestamps * num_cameras
         - intrinsics
         - timestamps: a tensor of shape [N] indicating which timestamp each frame belongs to.
         - near_fars: a numpy array of shape [num_cameras, 2]
        """

        poses, near_fars, intrinsics = self.load_all_poses()
        videopaths = np.array(glob.glob(os.path.join(self.datadir, '*.mp4')))  # [n_cameras]
        assert poses.shape[0] == len(videopaths), \
            'Mismatch between number of cameras and number of poses!'
        videopaths.sort()

        # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
        if self.split == 'train':
            split_ids = np.arange(1, poses.shape[0])
            # split_ids = np.array([1])
        else:
            split_ids = np.array([0])
        poses, near_fars, videopaths = poses[split_ids], near_fars[split_ids], videopaths[split_ids]

        imgs, timestamps = [], []
        all_poses = []
        for camera_id in tqdm(range(len(videopaths)), desc=f"Loading {self.split} data"):
            cam_video = imageio.get_reader(videopaths[camera_id], 'ffmpeg')
            for frame_idx, frame in enumerate(cam_video):
                # Decide whether to keep this frame or not
                if np.random.uniform() > self.subsample_time:
                    continue
                # Do any downsampling on the image
                img = self.load_image(frame, intrinsics.height, intrinsics.width)
                imgs.append(img)
                timestamps.append(frame_idx)
                all_poses.append(torch.from_numpy(poses[camera_id]).float())
        timestamps = torch.tensor(timestamps)  # [N]
        poses = all_poses  # [N, 3, 4]

        log.info(f"VideoLLFFDataset - Loaded {self.split} set from {self.datadir}: "
                 f"{len(imgs)} images of size {imgs[0].shape[0]}x{imgs[0].shape[1]} and "
                 f"{imgs[0].shape[2]} channels. "
                 f"{len(torch.unique(timestamps))} timestamps up to time={torch.max(timestamps)}. "
                 f"{intrinsics}")

        return poses, imgs, intrinsics, timestamps, near_fars
