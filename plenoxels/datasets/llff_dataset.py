import glob
import os
import logging as log
from typing import Tuple, Optional, List

import numpy as np
import torch

from .data_loading import parallel_load_images
from .ray_utils import (
    get_ray_directions_blender, get_rays, ndc_rays_blender, center_poses,
    gen_camera_dirs
)
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset


class LLFFDataset(BaseDataset):
    def __init__(self,
                 datadir,
                 split: str,
                 dset_id: int,
                 batch_size: Optional[int] = None,
                 downsample: int = 4,
                 hold_every: int = 8):
        self.downsample = downsample
        self.hold_every = hold_every
        self.dset_id = dset_id
        use_contraction = True

        image_paths, self.poses, self.per_cam_near_fars, intrinsics = load_llff_poses(
            datadir, downsample=downsample, split=split, hold_every=hold_every, near_scaling=1.0)
        imgs = load_llff_images(image_paths, intrinsics, split)
        imgs = (imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        else:
            imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])
        if use_contraction:
            bbox = torch.tensor([[-2., -2., -2.], [2., 2., 2.]])
        else:
            bbox = torch.tensor([[-1.5, -1.67, -1.], [1.5, 1.67, 1.]])
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=bbox,
                         is_ndc=not use_contraction,
                         batch_size=batch_size,
                         imgs=imgs,
                         rays_o=None,
                         rays_d=None,
                         intrinsics=intrinsics,
                         is_contracted=use_contraction)
        log.info(f"LLFFDataset contracted {self.is_contracted} - Loaded {split} set from {datadir}: {len(self.poses)} images of "
                 f"shape {self.img_h}x{self.img_w} with {imgs.shape[-1]} channels. {intrinsics}")

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
        if self.is_ndc:
            near_fars = torch.tensor([[0.0, 1.0]])
        else:
            near_fars = self.per_cam_near_fars[image_id, :]

        rgba = self.imgs[index] / 255.0  # (num_rays, 3)   this converts to f32
        c2w = self.poses[image_id]       # (num_rays, 3, 4)
        camera_dirs = gen_camera_dirs(
            x, y, self.intrinsics, True)  # [num_rays, 3]

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        if self.is_ndc:
            origins, directions = ndc_rays_blender(
                intrinsics=self.intrinsics, near=1.0, rays_o=origins, rays_d=directions)
        else:
            directions /= torch.linalg.norm(directions, dim=-1, keepdim=True)
        return {
            "rays_o": origins.reshape(-1, 3),
            "rays_d": directions.reshape(-1, 3),
            "imgs": rgba.reshape(-1, rgba.shape[-1]),
            "dset_id": self.dset_id,
            "near_far": near_fars,
        }


def _split_poses_bounds(poses_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    intrinsics = Intrinsics(
        width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2, center_y=H / 2)
    return poses[:, :, :4], near_fars, intrinsics


def load_llff_poses_helper(datadir: str, downsample: float, near_scaling: float) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses_bounds = np.load(os.path.join(datadir, 'poses_bounds.npy'))  # (N_images, 17)
    poses, near_fars, intrinsics = _split_poses_bounds(poses_bounds)

    # Step 1: rescale focal length according to training resolution
    intrinsics.scale(1 / downsample)

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses)

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = np.min(near_fars)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    return poses, near_fars, intrinsics


def load_llff_poses(datadir: str, downsample: float, split: str, hold_every: int, near_scaling: float = 0.75):
    int_dsample = int(downsample)
    if int_dsample != downsample or int_dsample not in {4, 8}:
        raise ValueError(f"Cannot downsample LLFF dataset by {downsample}.")

    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    image_paths = sorted(glob.glob(os.path.join(datadir, f'images_{int_dsample}/*')))
    assert poses.shape[0] == len(image_paths), \
        'Mismatch between number of images and number of poses! Please rerun COLMAP!'

    # Take training or test split
    i_test = np.arange(0, poses.shape[0], hold_every)  # [np.argmin(dists)]
    img_list = i_test if split != 'train' else list(set(np.arange(len(poses))) - set(i_test))
    # If you want to visualize train results
    # img_list = list(set(np.arange(len(poses))) - set(i_test))
    # if split != 'train':
    #     img_list = img_list[0:5]  # Test on some train images
    img_list = np.asarray(img_list)

    image_paths = [image_paths[i] for i in img_list]
    poses = torch.from_numpy(poses[img_list]).float()
    near_fars = torch.from_numpy(near_fars[img_list]).float()

    return image_paths, poses, near_fars, intrinsics


def load_llff_images(image_paths: List[str], intrinsics: Intrinsics, split: str):
    all_rgbs: List[torch.Tensor] = parallel_load_images(
        tqdm_title=f'Loading {split} data',
        dset_type='llff',
        data_dir='/',  # paths from glob are absolute
        num_images=len(image_paths),
        paths=image_paths,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        resolution=(None, None),
    )
    return torch.stack(all_rgbs, 0)


def create_llff_rays(imgs: Optional[torch.Tensor],
                     poses: torch.Tensor,
                     intrinsics: Intrinsics,
                     merge_all: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ray directions for all pixels, same for all images (same H, W, focal)
    directions = get_ray_directions_blender(intrinsics)  # H, W, 3

    all_rays_o, all_rays_d = [], []
    for i in range(poses.shape[0]):
        rays_o, rays_d = get_rays(directions, poses[i])  # both (h*w, 3)
        rays_o, rays_d = ndc_rays_blender(
            intrinsics=intrinsics, near=1.0, rays_o=rays_o, rays_d=rays_d)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

    all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    if imgs is not None:
        imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)  # [n_frames * h * w, C]
    if not merge_all:
        num_pixels = intrinsics.height * intrinsics.width
        all_rays_o = all_rays_o.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
        all_rays_d = all_rays_d.view(-1, num_pixels, 3)  # [n_frames, h * w, 3]
        if imgs is not None:
            imgs = imgs.view(-1, num_pixels, imgs.shape[-1])

    return all_rays_o, all_rays_d, imgs
