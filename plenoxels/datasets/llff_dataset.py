import glob
import os
import logging as log
from typing import Tuple, Optional, List

import numpy as np
import torch

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender, center_poses, generate_spiral_path
from .intrinsics import Intrinsics
from .base_dataset import BaseDataset
from .patchloader import PatchLoader


class LLFFDataset(BaseDataset):
    patchloader: Optional[PatchLoader]

    def __init__(self,
                 datadir,
                 split: str,
                 dset_id: int,
                 batch_size: Optional[int] = None,
                 generator: Optional[torch.random.Generator] = None,
                 downsample: int = 4,
                 hold_every: int = 8,
                 patch_size: Optional[int] = 8,
                 extra_views: bool = False):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.downsample = downsample
        self.hold_every = hold_every
        self.patch_size = patch_size
        self.dset_id = dset_id
        self.near_far = [0.0, 1.0]
        self.extra_views = split == 'train' and extra_views
        self.patchloader = None

        image_paths, poses, near_fars, intrinsics = load_llff_poses(
            datadir, downsample=downsample, split=split, hold_every=hold_every, near_scaling=1.0)
        imgs = load_llff_images(image_paths, intrinsics, split)
        rays_o, rays_d, imgs = create_llff_rays(
            imgs=imgs, poses=poses, intrinsics=intrinsics, merge_all=split == 'train')
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=self.init_bbox(datadir),
                         is_ndc=True,
                         generator=generator,
                         batch_size=batch_size,
                         imgs=imgs,
                         rays_o=rays_o,
                         rays_d=rays_d,
                         intrinsics=intrinsics)

        if self.extra_views:
            extra_poses = generate_spiral_path(poses.numpy(), near_fars, n_frames=120)
            extra_rays_o, extra_rays_d, _ = create_llff_rays(
                imgs=None, poses=extra_poses, intrinsics=intrinsics, merge_all=False)
            extra_rays_o = extra_rays_o.view(-1, self.img_h, self.img_w, 3)
            extra_rays_d = extra_rays_d.view(-1, self.img_h, self.img_w, 3)
            self.patchloader = PatchLoader(extra_rays_o, extra_rays_d, len_time=None,
                                           batch_size=self.batch_size, patch_size=self.patch_size,
                                           generator=self.generator)

        log.info(f"LLFFDataset - Loaded {split} set from {datadir}: {len(poses)} images of "
                 f"shape {self.img_h}x{self.img_w} with {imgs.shape[-1]} channels. {intrinsics}")

    def init_bbox(self, datadir):
        return torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])

    def __getitem__(self, index):
        out = super().__getitem__(index)
        out["dset_id"] = self.dset_id
        if self.split == 'train' and self.extra_views:
            out.update(self.patchloader[index])
        return out


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
