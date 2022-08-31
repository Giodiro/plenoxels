import json
import os
import random
import re
from typing import Tuple, Optional, List
import logging as log

import PIL
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
                self.resolution[0] * 2 < out_h or self.resolution[1] * 2 < out_w:
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


class MultiSyntheticNerfDataset(TensorDataset):
    def __init__(self, datadirs: List[str], split: str, low_resolution: int, high_resolution: int,
                 max_frames: Optional[int] = None):
        self.datadirs = datadirs
        self.split = split
        self.low_resolution = low_resolution
        self.high_resolution = high_resolution
        self.max_frames = max_frames

        self.white_bg = True
        self.near_far = [2.0, 6.0]

        self.scene_bbox = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]])
        self.pil2tensor = torchvision.transforms.ToTensor()
        self.tensor2pil = torchvision.transforms.ToPILImage()

        all_imgs_lr, all_imgs_hr, all_poses, all_focal, all_scene_ids = [], [], [], [], []
        for i, dd in enumerate(self.datadirs):
            imgs_lr, imgs_hr, poses, focal = self.load_from_disk(dd)
            all_imgs_lr.append(imgs_lr)
            all_imgs_hr.append(imgs_hr)
            all_poses.append(poses)
            all_focal.append(focal)
            all_scene_ids.append(torch.full((imgs_lr.shape[0], ), i, dtype=torch.int32))
        self.imgs_lr = torch.cat(all_imgs_lr, 0)
        self.imgs_hr = torch.cat(all_imgs_hr, 0)
        self.poses = torch.cat(all_poses, 0)
        self.focal = torch.cat(all_focal, 0)
        self.scene_ids = torch.cat(all_scene_ids, 0)

        self.rays_o, self.rays_d = self.init_rays()  # [N, H*W, ro+rd, 3]
        self.imgs_hr = self.imgs_hr.reshape(self.imgs_hr.shape[0], -1, 3)  # [N, H*W, 3]
        self.imgs_lr = self.imgs_lr.reshape(self.imgs_lr.shape[0], -1, 3)  # [N, H*W, 3]
        super().__init__(self.rays_o, self.rays_d, self.imgs_lr, self.imgs_hr, self.scene_ids)

    def process_img(self, img: PIL.Image):
        img = self.pil2tensor(img)
        img = img.permute(1, 2, 0)  # [h, w, 4]
        img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])  # Blend A into RGB
        return img

    def load_from_disk(self, datadir):
        with open(os.path.join(datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            poses, imgs_lr, imgs_hr = [], [], []
            num_frames = min(len(meta['frames']), self.max_frames or len(meta['frames']))
            for i in tqdm(range(num_frames), desc=f'Loading {self.split} data'):
                frame = meta['frames'][i]
                # Load pose
                pose = np.array(frame['transform_matrix'])
                poses.append(torch.tensor(pose))
                # Load image
                img_path = os.path.join(
                    datadir, self.split, f"{os.path.basename(frame['file_path'])}.png")
                img = Image.open(img_path)

                img_lr = img.resize((self.low_resolution, self.low_resolution), Image.BICUBIC)
                img_hr = img.resize((self.high_resolution, self.high_resolution), Image.BICUBIC)
                imgs_lr.append(self.process_img(img_lr))
                imgs_hr.append(self.process_img(img_hr))
            focal = 0.5 * self.low_resolution / np.tan(0.5 * meta['camera_angle_x'])
        imgs_lr = torch.stack(imgs_lr, 0)  # [N, H, W, 3]
        imgs_hr = torch.stack(imgs_hr, 0)  # [N, H, W, 3]
        poses = torch.stack(poses, 0)  # [N, ????]
        return imgs_lr, imgs_hr, poses, torch.tensor(focal).repeat(imgs_lr.shape[0])

    def init_rays(self):
        assert self.imgs_lr is not None and self.poses is not None and self.focal is not None
        # Low-pass the images at required resolution
        num_frames = self.imgs_lr.shape[0]

        # Rays
        rays = torch.stack(
            [get_rays(self.low_resolution, self.low_resolution, self.focal[i], self.poses[i, :3, :4])
             for i in range(num_frames)], 0)  # [N, ro+rd, H, W, 3]
        # Merge H, W dimensions
        rays = rays.permute(0, 2, 3, 1, 4).reshape(num_frames, -1, 2, 3)  # [N, H*W, ro+rd, 3]
        rays = rays.to(dtype=torch.float32)
        rays_o = rays[:, :, 0, :].contiguous()
        rays_d = rays[:, :, 1, :].contiguous()

        return rays_o, rays_d


class MultiSyntheticNerfDatasetv2(MultiSyntheticNerfDataset):
    def __init__(self,
                 datadirs: List[str],
                 split: str,
                 low_resolution: int,
                 high_resolution: int,
                 patch_size: int,
                 max_frames: Optional[int] = None):
        super().__init__(datadirs, split, low_resolution, high_resolution, max_frames)
        self.patch_size = patch_size
        scale_factor = high_resolution / low_resolution
        assert abs(scale_factor - int(scale_factor)) < 1e-9, f"{scale_factor} is not an integer"
        self.scale_factor = int(scale_factor)
        self.l_patch_size = self.patch_size // self.scale_factor

    def __getitem__(self, idx):
        h_img = self.imgs_hr[idx].view(self.high_resolution, self.high_resolution, 3)
        l_img = self.imgs_lr[idx].view(self.low_resolution, self.low_resolution, 3)
        rays_o = self.rays_o[idx].view(self.low_resolution, self.low_resolution, 2, 3)
        rays_d = self.rays_d[idx].view(self.low_resolution, self.low_resolution, 2, 3)
        scene_id = self.scene_ids[idx]

        if self.split == 'train':
            # Randomly crop the low-res patch
            rnd_h = random.randint(0, max(0, self.low_resolution - self.l_patch_size))
            rnd_w = random.randint(0, max(0, self.low_resolution - self.l_patch_size))
            l_img = l_img[rnd_h: rnd_h + self.l_patch_size, rnd_w: rnd_w + self.l_patch_size, :]
            rays_o = rays_o[rnd_h: rnd_h + self.l_patch_size, rnd_w: rnd_w + self.l_patch_size, ...]
            rays_d = rays_d[rnd_h: rnd_h + self.l_patch_size, rnd_w: rnd_w + self.l_patch_size, ...]
            # Crop the high-res patch correspondingly
            rnd_h_highres = int(rnd_h * self.scale_factor)
            rnd_w_highres = int(rnd_w * self.scale_factor)
            h_img = h_img[rnd_h_highres: rnd_h_highres + self.patch_size, rnd_w_highres: rnd_w_highres + self.patch_size, :]

        return {
            "scene_id": scene_id,
            "low": l_img,
            "high": h_img,
            "rays_o": rays_o,
            "rays_d": rays_d,
        }
