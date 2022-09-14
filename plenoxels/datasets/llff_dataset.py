import glob
import os
import logging as log
from typing import Tuple

import numpy as np
import torch

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender, center_poses
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


class LLFFDataset(BaseDataset):
    def __init__(self, datadir, split='train', downsample=4, hold_every=8, resolution=512):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        super().__init__(datadir=datadir,
                         scene_bbox=self.init_bbox(datadir),
                         split=split,
                         is_ndc=True)
        self.hold_every = hold_every
        self.downsample = downsample

        self.near_far = [0.0, 1.0]

        self.poses = None
        self.intrinsics = None
        self.near_fars = None
        self.imgs = None
        self.rays_o = None
        self.rays_d = None
        tensors = self.fetch_data()
        self.set_tensors(*tensors)

    def init_bbox(self, datadir):
        return torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])

    def fetch_data(self) -> Tuple[torch.Tensor, ...]:
        self.poses, imgs, self.intrinsics, self.near_fars = self.load_from_disk()
        self.rays_o, self.rays_d, self.imgs = self.init_rays(
            imgs, self.poses, merge_all=self.split == 'train')
        return self.rays_o, self.rays_d, self.imgs

    @property
    def img_w(self):
        return self.intrinsics.width

    @property
    def img_h(self):
        return self.intrinsics.height

    @property
    def num_frames(self):
        return len(self.poses)

    def split_poses_bounds(self, poses_bounds):
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        intrinsics = Intrinsics(width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2,
                                center_y=H / 2)
        return poses[:, :, :4], near_fars, intrinsics

    def load_all_poses(self) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
        poses_bounds = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))  # (N_images, 17)
        poses, near_fars, intrinsics = self.split_poses_bounds(poses_bounds)

        # Step 1: rescale focal length according to training resolution
        intrinsics.scale(1 / self.downsample)

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        poses, pose_avg = center_poses(poses)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = near_fars.min()
        scale_factor = near_original * 1.  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        near_fars /= scale_factor
        poses[..., 3] /= scale_factor

        return poses, near_fars, intrinsics

    def load_from_disk(self):
        poses, near_fars, intrinsics = self.load_all_poses()
        if self.downsample != 4:
            raise RuntimeError(f"LLFF dataset only works with donwsample=4 but found {self.downsample}.")
        image_paths = sorted(glob.glob(os.path.join(self.datadir, 'images_4/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert poses.shape[0] == len(image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        # dists = np.sum(np.square(pose_avg[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = np.arange(0, poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(
            set(np.arange(len(poses))) - set(i_test))
        img_list = np.asarray(img_list)

        # use first N_images-1 to train, the LAST is val
        all_rgbs = parallel_load_images(
            [image_paths[i] for i in img_list],
            tqdm_title=f'Loading {self.split} data',
            dset_type='llff',
            data_dir='/',
            out_h=intrinsics.height,
            out_w=intrinsics.width,
            resolution=(None, None)
        )
        poses = [torch.from_numpy(poses[i]).float() for i in img_list]
        near_fars = near_fars[img_list]

        log.info(f"LLFFDataset - Loaded {self.split} set from {self.datadir}: {len(all_rgbs)} "
                 f"images of with {all_rgbs[0].shape[-1]} channels. {intrinsics}")

        return poses, all_rgbs, intrinsics, near_fars

    def init_rays(self, imgs, poses, merge_all: bool):
        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions_blender(self.intrinsics)  # H, W, 3

        all_rays_o, all_rays_d = [], []
        for i in range(len(poses)):
            rays_o, rays_d = get_rays(directions, poses[i])  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(
                intrinsics=self.intrinsics, near=1.0, rays_o=rays_o, rays_d=rays_d)
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
        if imgs is not None:
            imgs = (torch.cat(imgs, 0)
                    .reshape(-1, imgs[0].shape[-1])
                    .to(dtype=torch.float32))  # [n_frames * h * w, C]
        if not merge_all:
            num_pixels = self.intrinsics.height * self.intrinsics.width
            all_rays_o = all_rays_o.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            all_rays_d = all_rays_d.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
            if imgs is not None:
                imgs = imgs.view(-1, num_pixels, imgs.shape[-1])

        return all_rays_o, all_rays_d, imgs
