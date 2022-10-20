import glob
import logging as log
import os
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .ray_utils import ndc_rays_blender, center_poses, gen_pixel_samples, gen_camera_dirs


class LLFFDataset(BaseDataset):
    OPENGL_CAMERA = True

    def __init__(self,
                 datadir,
                 split: str,
                 dset_id: int,
                 batch_size: Optional[int] = None,
                 generator: Optional[torch.random.Generator] = None,
                 downsample: int = 4,
                 hold_every: int = 8):
        self.downsample = downsample
        self.hold_every = hold_every
        self.dset_id = dset_id
        self.near = 0.0
        self.far = 1.0
        self.training = split == 'train' and batch_size is not None
        image_paths, camtoworlds, near_fars, intrinsics = load_llff_poses(
            datadir, downsample=downsample, split=split, hold_every=hold_every, near_scaling=0.75)
        images = load_llff_images(image_paths, intrinsics, split)
        super().__init__(datadir=datadir,
                         split=split,
                         scene_bbox=torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]),
                         is_ndc=True,
                         generator=generator,
                         batch_size=batch_size,
                         images=images,
                         intrinsics=intrinsics,
                         camtoworlds=camtoworlds)

        log.info(f"LLFFDataset - Loaded {split} set from {datadir}: {len(camtoworlds)} images of "
                 f"shape {self.img_h}x{self.img_w} with {images.shape[-1]} channels. {intrinsics}")

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays_o, rays_d = data["rgba"], data["rays_o"], data["rays_d"]
        return {
            "pixels": rgba,  # [n_rays, 3] or [h, w, 3]
            "rays_o": rays_o,  # [n_rays,] or [h, w]
            "rays_d": rays_d,  # [n_rays,] or [h, w]
            "color_bkgd": None,  # [3,]
            "dset_id": self.dset_id,
            **{k: v for k, v in data.items() if k not in {"rgba", "rays_o", "rays_d"}},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.batch_size
        image_id, x, y = gen_pixel_samples(
            self.training, self.images, index, num_rays, self.intrinsics)
        # generate rays
        rgba = self.images[image_id, y, x]  # (num_rays, 4)   this converts to f32
        c2w = self.camtoworlds[image_id]    # (num_rays, 3, 4)
        camera_dirs = gen_camera_dirs(
            x, y, self.intrinsics, self.OPENGL_CAMERA)  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(  # TODO: Unsure about this
            directions, dim=-1, keepdims=True
        )
        viewdirs = directions
        origins, viewdirs = ndc_rays_blender(
            intrinsics=self.intrinsics, near=1.0, rays_o=origins, rays_d=viewdirs)

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, rgba.shape[-1]))
        else:
            origins = torch.reshape(origins, (self.intrinsics.height, self.intrinsics.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.intrinsics.height, self.intrinsics.width, 3))
            rgba = torch.reshape(rgba, (self.intrinsics.height, self.intrinsics.width, rgba.shape[-1]))

        return {
            "rgba": rgba,       # [h, w, 4] or [num_rays, 4]
            "rays_o": origins,  # [h, w, 3] or [num_rays, 3]
            "rays_d": viewdirs,
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

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = np.min(near_fars)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses)

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
        image_paths,
        tqdm_title=f'Loading {split} data',
        dset_type='llff',
        data_dir='/',  # paths from glob are absolute
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        resolution=(None, None)
    )
    return torch.stack(all_rgbs, 0)
