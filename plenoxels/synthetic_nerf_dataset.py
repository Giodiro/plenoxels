import json
import os
from typing import Tuple, Optional, List

import PIL
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm


def get_rays(H: int, W: int, focal, c2w) -> torch.Tensor:
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
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return torch.stack((rays_o, rays_d), dim=0)


class SyntheticNerfDataset(TensorDataset):
    def __init__(self, datadir, split='train', downsample=1.0, resolution=512, max_frames=None,
                 init_data=None):
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.img_w: int = int(800 // self.downsample)
        self.img_h: int = int(800 // self.downsample)
        self.resolution = resolution
        self.max_frames = max_frames

        self.white_bg = True
        self.near_far = [2.0, 6.0]

        self.scene_bbox = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]])
        self.pil2tensor = torchvision.transforms.ToTensor()
        self.tensor2pil = torchvision.transforms.ToPILImage()

        if init_data is not None:
            self.orig_imgs = init_data['orig_imgs']
            self.poses = init_data['poses']
            self.focal = init_data['focal']
        else:
            self.orig_imgs, self.poses, self.focal = self.load_from_disk()

        self.imgs, self.rays = self.init_rays()

        super().__init__(self.rays, self.imgs)

    def load_from_disk(self):
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            poses, imgs = [], []
            num_frames = min(len(meta['frames']), self.max_frames or len(meta['frames']))
            for i in tqdm(range(num_frames), desc=f'Loading {self.split} data'):
                frame = meta['frames'][i]
                # Load pose
                pose = np.array(frame['transform_matrix'])
                poses.append(torch.tensor(pose))
                # Load image
                img_path = os.path.join(
                    self.datadir, self.split, f"{os.path.basename(frame['file_path'])}.png")
                img = Image.open(img_path)
                img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
                img = self.pil2tensor(img)
                img = img.permute(1, 2, 0)  # [h, w, 4]
                img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])  # Blend A into RGB
                imgs.append(img)
            focal = 0.5 * self.img_w / np.tan(0.5 * meta['camera_angle_x'])
        orig_imgs = torch.stack(imgs, 0)  # [N, H, W, 3]
        poses = torch.stack(poses, 0)  # [N, ????]
        return orig_imgs, poses, focal

    def low_pass(self, imgs):
        out = torch.empty_like(imgs)
        for i in range(imgs.shape[0]):
            img = self.tensor2pil(imgs[i].permute(2, 0, 1))  # t2p expects C, H, W image
            if self.resolution * 2 < self.img_w:
                img = img.resize((self.resolution * 2, self.resolution * 2), Image.LANCZOS)
                img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            out[i] = self.pil2tensor(img).permute(1, 2, 0)  # [H, W, C]]
        return out

    def init_rays(self):
        assert self.orig_imgs is not None and self.poses is not None and self.focal is not None
        # Low-pass the images at required resolution
        num_frames = self.orig_imgs.shape[0]
        imgs = self.low_pass(self.orig_imgs)
        imgs = imgs.view(-1, 3)  # [N*H*W, 3]

        # Rays
        rays = torch.stack(
            [get_rays(self.img_h, self.img_w, self.focal, p)
             for p in self.poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        # Merge N, H, W dimensions
        rays = rays.permute(0, 2, 3, 1, 4).reshape(-1, 2, 3)  # [N*H*W, ro+rd, 3]
        rays = rays.to(dtype=torch.float32).contiguous()

        if self.split == "test":
            imgs = imgs.view(num_frames, self.img_w * self.img_h, 3)  # [N, H*W, 3]
            rays = rays.view(num_frames, self.img_w * self.img_h, 2, 3)  # [N, H*W, 2, 3]
        return imgs, rays

    def update_resolution(self, new_reso):
        init_data = {"orig_imgs": self.orig_imgs, "poses": self.poses, "focal": self.focal}
        return SyntheticNerfDataset(datadir=self.datadir, split=self.split,
                                    downsample=self.downsample, resolution=new_reso,
                                    max_frames=self.max_frames, init_data=init_data)


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

        self.rays = self.init_rays()
        self.imgs_hr = self.imgs_hr.reshape(self.imgs_hr.shape[0], -1, 3)  # [N, H*W, 3]
        self.imgs_lr = self.imgs_lr.reshape(self.imgs_lr.shape[0], -1, 3)  # [N, H*W, 3]
        super().__init__(self.rays, self.imgs_lr, self.imgs_hr, self.scene_ids)

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

        return rays
