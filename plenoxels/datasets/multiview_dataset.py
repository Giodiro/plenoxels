import os
import logging as log
import json

import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.multiprocessing import Pool

from scipy.spatial.transform import Slerp, Rotation

from torch.utils.data import Dataset, DataLoader


def load_image(data_root, frame, downscale, dset_type, scale, offset, H, W):
    # Fix file-path
    f_path = os.path.join(data_root, frame['file_path'])
    if dset_type == 'blender' and '.' not in os.path.basename(f_path):
        f_path += '.png'  # so silly...
    if not os.path.exists(f_path):  # there are non-exist paths in fox...
        return (None, None)

    pose = np.array(frame['transform_matrix'], dtype=np.float32)  # [4, 4]
    pose = nerf_matrix_to_ngp(pose, scale=scale, offset=offset)

    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
    if H is None or W is None:
        H = int(image.shape[0] / downscale)
        W = int(image.shape[1] / downscale)

    # add support for the alpha channel as a mask.
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    if image.shape[0] != H or image.shape[1] != W:
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

    image = image.astype(np.float32) / 255  # [H, W, 3/4]
    return pose, image


def parallel_load_images(args):
    torch.set_num_threads(1)
    return load_image(args['data_root'], args['frame'], args['downscale'], args['dset_type'],
                      args['scale'], args['offset'], args['H'], args['W'])


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=(0, 0, 0)):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def load_transforms(root, dset_type, split):
    if dset_type == "colmap":
        # Only one transform file (no splits provided)
        with open(os.path.join(root, "transforms.json"), "r") as fh:
            transforms = json.load(fh)
    elif dset_type == "blender":
        with open(os.path.join(root, f"transforms_{split}.json"), "r") as fh:
            transforms = json.load(fh)
    else:
        raise NotImplementedError(dset_type)

    return transforms


def load_colmap_test(frames, scale, offset, max_test_frames):
    # choose two random poses, and interpolate between.
    f0, f1 = np.random.choice(frames, 2, replace=False)
    pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32),
                               scale=scale, offset=offset)  # [4, 4]
    pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32),
                               scale=scale, offset=offset)  # [4, 4]
    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
    slerp = Slerp([0, 1], rots)

    poses = []
    for i in range(max_test_frames + 1):
        ratio = np.sin(((i / max_test_frames) - 0.5) * np.pi) * 0.5 + 0.5
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = slerp(ratio).as_matrix()
        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
        poses.append(pose)
    return poses, None


def load_intrinsics(transform, downscale, W, H):
    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = (transform['cx'] / downscale) if 'cx' in transform else (W / 2)
    cy = (transform['cy'] / downscale) if 'cy' in transform else (H / 2)

    return np.array([fl_x, fl_y, cx, cy])


@torch.cuda.amp.autocast(enabled=False)
def get_rays(pose, intrinsics, H, W, N=-1):
    """get rays
    Args:
        pose: [4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        inds: [B, N]
    """
    device = pose.device
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
                          torch.linspace(0, H-1, H, device=device),
                          indexing='ij')  # float
    i = i.t().reshape([H*W]) + 0.5
    j = j.t().reshape([H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)
        inds = torch.randint(0, H*W, size=[N], device=device)  # may duplicate
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds
    else:
        inds = torch.arange(H*W, device=device)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ pose[:3, :3].transpose(-1, -2)  # (N, 3)

    rays_o = pose[:3, 3]  # [3]
    rays_o = rays_o[None, :].expand_as(rays_d)  # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    return results


class NerfDataset(Dataset):
    def __init__(self, data_root, dataset_type: str, split: str, batch_size: int, device,
                 tr_downscale: float = 1.0, scale=0.33, offset=(0, 0, 0), bound=(-1.0, 1.0),
                 max_frames=None, subsample_frames='random', **kwargs):
        super().__init__()

        self.data_root = data_root
        self.dset_type = dataset_type
        self.tr_downscale = tr_downscale
        self.scale = scale
        self.offset = offset
        self.bound = bound
        self.max_frames = max_frames
        self.subsample_frames_type = subsample_frames
        self.batch_size = batch_size
        self.split = split
        self.num_load_workers = 4
        self.device = device

        data = self.load_data()
        self.poses = data['poses']
        self.images = data['images']
        self.radius = data['radius']
        self.intrinsics = data['intrinsics']
        self.H, self.W = data['H'], data['W']

    def get_different_split(self, split: str, device=None, max_frames=None) -> 'NerfDataset':
        if device is None:
            device = self.device
        if max_frames is None:
            max_frames = self.max_frames
        return NerfDataset(data_root=self.data_root, dataset_type=self.dset_type, split=split,
                           batch_size=self.batch_size, device=device,
                           tr_downscale=self.tr_downscale, scale=self.scale, offset=self.offset,
                           bound=self.bound, max_frames=max_frames,
                           subsample_frames=self.subsample_frames_type)

    def subsample_frames(self, frames):
        if self.max_frames is not None and self.max_frames < len(frames):
            if self.subsample_frames_type == 'random':
                frames = np.random.choice(frames, self.max_frames, replace=False)
            elif self.subsample_frames_type == 'staggered':
                take_frame_every = len(frames) // self.max_frames
                frames = frames[::take_frame_every]
            elif self.subsample_frames_type == 'sequential':
                frames = frames[:self.max_frames]
            else:
                raise NotImplementedError(self.subsample_frames_type)
        return frames

    def load_data(self):
        transform = load_transforms(self.data_root, self.dset_type, self.split)
        downscale = self.tr_downscale if self.split == 'train' else 1.0
        # load image size
        if 'h' in transform and 'w' in transform:
            H = int(int(transform['h']) / downscale)
            W = int(int(transform['w']) / downscale)
            log.info(f"Loading data with size {H}x{W}")
        else:
            # we have to actually read an image to get H and W later.
            H = W = None
        frames = transform["frames"]
        if self.dset_type == "colmap" and self.split == "test":
            max_frames = self.max_frames or 10
            poses, images = load_colmap_test(frames, self.scale, self.offset, max_frames)
        else:
            # for colmap, manually split a valid set (the first frame).
            if self.dset_type == "colmap" and self.split == "train":
                frames = frames[1:]
            elif self.dset_type == "colmap" and self.split == "val":
                frames = frames[:1]
            frames = self.subsample_frames(frames)
            p = Pool(min(self.num_load_workers, len(frames)))
            iterator = p.imap(parallel_load_images,
                              [dict(frame=frame, data_root=self.data_root, downscale=downscale,
                                    dset_type=self.dset_type, scale=self.scale, offset=self.offset,
                                    H=H, W=W) for frame in frames])
            poses, images = [], []
            for _ in tqdm(range(len(frames))):
                pose, image = next(iterator)
                if pose is not None:
                    poses.append(pose)
                if image is not None:
                    images.append(image)

        poses = torch.from_numpy(np.stack(poses, axis=0))  # [N, 4, 4]
        if images is not None:
            images = torch.from_numpy(np.stack(images, axis=0))  # [N, H, W, C]
        if images is not None and H is None:
            H, W = images.shape[1:3]  # downscaling already performed.
            log.info(f"Loading data with size {H}x{W}")

        if False:
            from .viz import visualize_poses
            visualize_poses(poses.numpy())

        radius = poses[:, :3, 3].norm(dim=-1).mean(0).item()
        log.info(f'[INFO] dataset camera poses: radius = {radius:.4f}, bound = {self.bound}')
        intrinsics = load_intrinsics(transform, downscale, W=W, H=H)
        return {
            "poses": poses,
            "images": images,
            "radius": radius,
            "intrinsics": intrinsics,
            "H": H, "W": W,
        }

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, item):
        pose = self.poses[item].to(device=self.device)
        rays = get_rays(pose, self.intrinsics, self.H, self.W, self.batch_size)
        out = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }
        if self.images is not None:
            images = self.images[item].to(self.device)  # [H, W, 3/4]
            if self.split == 'train':
                C = images.shape[-1]
                images = torch.gather(images.view(-1, C), 0, torch.stack(C * [rays['inds']], -1))  # [N, 3/4]
            out['images'] = images
        return out

    def dataloader(self):
        loader = DataLoader(self, batch_size=1, shuffle=self.split == 'train', num_workers=0)
        loader.has_gt = self.images is not None
        return loader
