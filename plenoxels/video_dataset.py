import json
import logging as log
import os
from typing import Optional

import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torchvision.transforms
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm
import glob
import imageio


def get_rays(H: int, W: int, focal_x, focal_y, c2w) -> torch.Tensor:
    """

    :param H:
    :param W:
    :param focal:
    :param c2w:
    :return:
        Tensor of size [2, W, H, 3] where the first dimension indexes origin and direction
        of rays
    """
    i, j = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing='xy')

    dirs = torch.stack([
        (i - W * 0.5) / focal_x,
        -(j - H * 0.5) / focal_y,
        -torch.ones_like(i)
    ], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return torch.stack((rays_o, rays_d), dim=0)

class VideoDataset(TensorDataset):
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self, datadir, split='train', downsample=1.0, subsample_time=1):
        # subsample_time (in [0,1]) lets you use a random percentage of the frames from each video
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.subsample_time = subsample_time

        self.radius = 10  # TODO: this might need to change
        self.len_time = None

        imgs, self.poses, self.timestamps, self.intrinsics = self.load_from_disk()
        imageio.imwrite(f'split{split}gt.png', imgs[0])
        self.imgs, self.rays_o, self.rays_d = self.init_rays(imgs)
        # Broadcast timestamps to match rays
        if split == 'train':
            self.timestamps = (torch.ones(len(self.timestamps), self.img_w, self.img_h) * self.timestamps[:,None,None]).reshape(-1)
        super().__init__(self.rays_o, self.rays_d, self.timestamps, self.imgs)

    def load_image(self, img, out_h, out_w):
        img = self.tensor2pil(img)
        if out_h is None:
            out_h = int(img.size[0] / self.downsample)
            self.img_h = out_h
        if out_w is None:
            out_w = int(img.size[1] / self.downsample)
            self.img_w = out_w
        img = img.resize((out_h, out_w), Image.LANCZOS)
        img = self.pil2tensor(img)  # [C, H, W]
        img = img.permute(1, 2, 0)  # [H, W, C]
        return img

    def load_pose(self, pose):
        # pose is provided as vector of 17 numbers. The first 12 are the 3x4 portion of the transform matrix.
        transform_matrix = torch.zeros(4, 4, dtype=torch.float32)
        transform_matrix[3,3] = 1
        transform_matrix[0:3, :] = torch.from_numpy(pose[0:12]).reshape(3,4)
        return transform_matrix

    def load_intrinsics(self, pose):
        # The remaining 5 numbers in pose are (in order): height, width, focal, near_bound, far_bound
        focal = pose[14]
        assert self.img_h is not None
        assert self.img_w is not None
        # intrinsics = torch.from_numpy(np.array([focal, focal, self.img_w / 2, self.img_h / 2]))
        intrinsics = torch.from_numpy(np.array([focal * 8 *self.img_w / self.img_h, focal * 8 * self.img_h / self.img_w, self.img_h / 2, self.img_w / 2]))
        return intrinsics

    def load_from_disk(self):
        poses_bounds = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))  # [n_cameras, 17]
        imgs, poses, timestamps = [], [], []
        videopaths = glob.glob(os.path.join(self.datadir, '*.mp4'))  # [n_cameras]
        videopaths.sort()
        len_time = 0
        # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
        if self.split == 'train':
            poses_bounds = poses_bounds[1:]
            videopaths = videopaths[1:]
        else:
            poses_bounds = [poses_bounds[0]]
            videopaths = [videopaths[0]]
        for camera_id in tqdm(range(len(videopaths))): 
            cam_pose = poses_bounds[camera_id]
            # Get the pose
            pose = self.load_pose(cam_pose)
            cam_video = imageio.get_reader(videopaths[camera_id], 'ffmpeg')
            for frame_idx, frame in enumerate(cam_video):
                if frame_idx > len_time:
                    len_time = frame_idx
                # Decide whether to keep this frame or not
                # if np.random.uniform() > self.subsample_time:
                #     continue
                # Only keep frame zero, for debugging
                if frame_idx > 0:
                    len_time = 300
                    break
                # Do any downsampling on the image
                img = self.load_image(frame, self.img_h, self.img_w)
                imgs.append(img)
                poses.append(pose)
                timestamps.append(frame_idx)
        intrinsics = self.load_intrinsics(poses_bounds[0])  # [4] Intrinsics are common to all cameras
        self.len_time = len_time
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3]
        poses = torch.stack(poses, 0)  # [N, 4, 4]
        timestamps = torch.from_numpy(np.array(timestamps))  # [N]
        log.info(f"Loaded {self.split}-set from {self.datadir}: {imgs.shape[0]} images of size "
                 f"{imgs.shape[1]}x{imgs.shape[2]} and {imgs.shape[3]} channels. "
                 f"Focal length {intrinsics[0]:.2f}, {intrinsics[1]:.2f}. "
                 f"Center {intrinsics[2]:.2f}, {intrinsics[3]:.2f}")
        return imgs, poses, timestamps, intrinsics

    def init_rays(self, imgs):
        assert imgs is not None and self.poses is not None and self.intrinsics is not None
        num_frames = imgs.shape[0]
        imgs = imgs.view(-1, imgs.shape[-1])  # [N*H*W, 3]

        # Rays
        rays = torch.stack(
            [get_rays(self.img_h, self.img_w, focal_x=self.intrinsics[0], focal_y=self.intrinsics[1], c2w=p)
             for p in self.poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        # Merge N, H, W dimensions
        rays_o = rays[:, 0, ...].reshape(-1, 3)  # [N*H*W, 3]
        rays_o = rays_o.to(dtype=torch.float32).contiguous()
        rays_d = rays[:, 1, ...].reshape(-1, 3)  # [N*H*W, 3]
        rays_d = rays_d.to(dtype=torch.float32).contiguous()

        if self.split != "train":
            imgs = imgs.view(num_frames, self.img_w * self.img_h, -1)  # [N, H*W, 3/4]
            rays_o = rays_o.view(num_frames, self.img_w * self.img_h, 3)  # [N, H*W, 3]
            rays_d = rays_d.view(num_frames, self.img_w * self.img_h, 3)  # [N, H*W, 3]
        return imgs, rays_o, rays_d
