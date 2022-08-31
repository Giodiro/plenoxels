import json
import logging as log
import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm


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


class SyntheticNerfDataset(TensorDataset):
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self, datadir, split='train', downsample=1.0, resolution=512, max_frames=None):
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.img_w: Optional[int] = None
        self.img_h: Optional[int] = None
        self.resolution = (resolution, resolution)
        self.max_frames = max_frames

        self.near_far = [2.0, 6.0]

        if "ship" in datadir:
            self.radius = 1.5
        else:
            self.radius = 1.3

        imgs, self.poses, self.intrinsics = self.load_from_disk()
        self.imgs, self.rays_o, self.rays_d = self.init_rays(imgs)

        super().__init__(self.rays_o, self.rays_d, self.imgs)

    def load_image_pose(self, frame, out_h, out_w):
        # Fix file-path
        f_path = os.path.join(self.datadir, frame['file_path'])
        if '.' not in os.path.basename(f_path):
            f_path += '.png'  # so silly...
        if not os.path.exists(f_path):  # there are non-exist paths in fox...
            return (None, None)
        img = Image.open(f_path)
        if out_h is None:
            out_h = int(img.size[0] / self.downsample)
        if out_w is None:
            out_w = int(img.size[1] / self.downsample)
        # Now we should downsample to out_h, out_w and low-pass filter to resolution * 2.
        # We only do the low-pass filtering if resolution * 2 is lower-res than out_h, out_w
        if out_h != out_w:
            log.warning("")
        if self.resolution[0] is not None and self.resolution[1] is not None and \
                (self.resolution[0] * 2 < out_h or self.resolution[1] * 2 < out_w):
            img = img.resize((self.resolution[0] * 2, self.resolution[1] * 2), Image.LANCZOS)
            img = img.resize((out_h, out_w), Image.LANCZOS)
        else:
            img = img.resize((out_h, out_w), Image.LANCZOS)
        img = self.pil2tensor(img)  # [C, H, W]
        img = img.permute(1, 2, 0)  # [H, W, C]

        pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

        return (img, pose)

    def load_intrinsics(self, transform):
        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.img_w / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.img_h / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.img_w / 2)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.img_h / 2)

        return np.array([fl_x, fl_y, cx, cy])

    def subsample_dataset(self, frames):
        tot_frames = len(frames)

        num_frames = min(tot_frames, self.max_frames or tot_frames)
        if self.split == 'train':
            subsample = int(round(tot_frames / num_frames))
            frame_ids = np.arange(tot_frames)[::subsample]
            if subsample > 1:
                log.info(f"Subsampling training set to 1 every {subsample} images.")
        else:
            frame_ids = np.arange(num_frames)
        return np.take(frames, frame_ids).tolist()


    def load_from_disk(self):
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            poses, imgs = [], []
            frames = self.subsample_dataset(meta['frames'])
            for frame in tqdm(frames, desc=f'Loading {self.split} data'):
                img, pose = self.load_image_pose(frame, self.img_h, self.img_w)
                imgs.append(img)
                poses.append(pose)
            self.img_h, self.img_w = imgs[0].shape[:2]
            intrinsics = self.load_intrinsics(meta)
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
        poses = torch.stack(poses, 0)  # [N, ????]
        log.info(f"Loaded {self.split}-set from {self.datadir}: {imgs.shape[0]} images of size "
                 f"{imgs.shape[1]}x{imgs.shape[2]} and {imgs.shape[3]} channels. "
                 f"Focal length {intrinsics[0]:.2f}, {intrinsics[1]:.2f}. "
                 f"Center {intrinsics[2]:.2f}, {intrinsics[3]:.2f}")
        return imgs, poses, intrinsics

    def init_rays(self, imgs):
        assert imgs is not None and self.poses is not None and self.intrinsics is not None
        # Low-pass the images at required resolution
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
