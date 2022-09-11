import logging as log
import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from ..tqdm import tqdm
import glob
import imageio

from .intrinsics import Intrinsics
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender
from .llff_dataset import center_poses
from .base_dataset import BaseDataset


# This version uses normalized device coordinates, as in LLFF, for forward-facing videos


class VideoDataset(BaseDataset):
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self, datadir, split='train', downsample=1.0, subsample_time=1):
        super().__init__(datadir=datadir,
                         scene_bbox=torch.tensor([[-30.0, -3.0, -2.0], [3, 80, 100.0]]),
                         split=split,
                         is_ndc=True)
        # subsample_time (in [0,1]) lets you use a random percentage of the frames from each video
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.subsample_time = subsample_time

        self.len_time = None
        self.near_fars = None

        self.poses, rgbs, self.intrinsics, self.timestamps = self.load_from_disk()
        self.num_frames = len(self.poses)
        self.rays_o, self.rays_d, self.rgbs = self.init_rays(rgbs)
        # Broadcast timestamps to match rays
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(1, self.img_h, self.img_w).view(-1)
        self.set_tensors(self.rays_o, self.rays_d, self.rgbs, self.timestamps)

    def load_image(self, img, out_h, out_w):
        img = self.tensor2pil(img)
        img = img.resize((out_w, out_h), Image.LANCZOS)
        img = self.pil2tensor(img)  # [C, H, W]
        img = img.permute(1, 2, 0)  # [H, W, C]
        return img

    def load_from_disk(self):
        poses_bounds = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))  # [n_cameras, 17]
        imgs, poses, timestamps = [], [], []
        videopaths = np.array(glob.glob(os.path.join(self.datadir, '*.mp4')))  # [n_cameras]
        assert len(poses_bounds) == len(videopaths), \
            'Mismatch between number of cameras and number of poses!'
        videopaths.sort()
        len_time = 0
        # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
        if self.split == 'train':
            poses_bounds = poses_bounds[1:]
            videopaths = videopaths[1:]
        else:
            poses_bounds = poses_bounds[0].reshape(1, 17)
            videopaths = [videopaths[0]]

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # [N_cameras, 3, 5]
        self.near_fars = poses_bounds[:, -2:]  # [N_cameras, 2]

        # Step 1: rescale intrinsics according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        intrinsics = Intrinsics(width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2,
                                center_y=H / 2)
        intrinsics.scale(1 / self.downsample)

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # [N_images, 3, 4] exclude H, W, focal
        poses, pose_avg = center_poses(poses)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        poses[..., 3] /= scale_factor

        all_poses = []
        for camera_id in tqdm(range(len(videopaths))):
            cam_video = imageio.get_reader(videopaths[camera_id], 'ffmpeg')
            for frame_idx, frame in enumerate(cam_video):
                if frame_idx > len_time:
                    len_time = frame_idx + 1
                # Decide whether to keep this frame or not
                if np.random.uniform() > self.subsample_time:
                    continue
                # Only keep frame zero, for debugging
                # if frame_idx > 0:
                #     len_time = 300
                #     break
                # Do any downsampling on the image
                img = self.load_image(frame, self.intrinsics.height, self.intrinsics.width)
                imgs.append(img)
                timestamps.append(frame_idx)
                all_poses.append(torch.from_numpy(poses[camera_id]).float())
        self.len_time = len_time
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3]
        timestamps = torch.tensor(timestamps)  # [N]
        poses = all_poses  # [N, 3, 4]

        log.info(f"LLFFDataset - Loaded {self.split} set from {self.datadir}: {len(imgs)} "
                 f"images of size {imgs[0].shape}. {intrinsics}")

        return poses, imgs, intrinsics, timestamps

    def init_rays(self, imgs):
        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions_blender(self.intrinsics)  # H, W, 3

        all_rays_o, all_rays_d = [], []
        for i in range(len(self.poses)):
            rays_o, rays_d = get_rays(directions, self.poses[i])  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(
                intrinsics=self.intrinsics, near=1.0, rays_o=rays_o, rays_d=rays_d)
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        all_rgbs = (imgs
                    .reshape(-1, imgs[0].shape[-1])
                    .to(dtype=torch.float32))  # [n_frames * h * w, C]
        if self.split != 'train':
            num_pixels = self.intrinsics.height * self.intrinsics.width
            all_rays_o = all_rays_o.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rays_d = all_rays_d.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rgbs = all_rgbs.view(-1, num_pixels, all_rgbs.shape[-1])

        return all_rays_o, all_rays_d, all_rgbs
