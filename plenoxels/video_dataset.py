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

from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender

"""
This version can use either regular or normalized device coordinates (NDC)
NDC is as in LLFF, for forward-facing videos
We assume that if pose information is provided as a .json, then the scene is 360
and if pose is provided in a .npy, the scene is forward-facing
"""


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


class VideoDataset(TensorDataset):
    pil2tensor = torchvision.transforms.ToTensor()
    tensor2pil = torchvision.transforms.ToPILImage()

    def __init__(self, datadir, split='train', downsample=1.0, subsample=1, max_test_cameras=np.inf):
        # subsample (in [0,1]) lets you use a random percentage of the frames from each video
        self.datadir = datadir
        self.split = split
        self.downsample = downsample
        self.subsample = subsample
        self.max_test_cameras = max_test_cameras

        # Figure out if this is a forward-facing or 360 scene
        self.ndc = False
        self.aabb = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]]).cuda()
        if "ship" in self.datadir:
            self.aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]).cuda()
        pose_files = glob.glob(os.path.join(self.datadir, '*.npy'))
        if len(pose_files) > 0:
            self.ndc = True
            self.aabb = torch.tensor([[-2.0, -2.0, -1.0], [2, 2, 1.0]]).cuda() # This is made up
        self.len_time = None
        self.near_fars = None

        if self.ndc:
            self.poses, rgbs, self.intrinsics, self.timestamps = self.load_from_disk_ndc()
        else:
            self.poses, rgbs, self.intrinsics, self.timestamps = self.load_from_disk()
        self.num_frames = len(self.poses)
        self.rays_o, self.rays_d, self.rgbs = self.init_rays(rgbs)
        # Broadcast timestamps to match rays
        if split == 'train':
            self.timestamps = (torch.ones(len(self.timestamps), self.intrinsics.height, self.intrinsics.width) * self.timestamps[:,None,None]).reshape(-1)
        print(f'rays_o has shape {self.rays_o.shape}, rays_d has shape {self.rays_d.shape}, rgbs has shape {self.rgbs.shape}, timestamps has shape {self.timestamps.shape}')
        super().__init__(self.rays_o, self.rays_d, self.rgbs, self.timestamps)

    def load_image(self, img, out_h, out_w, already_pil=False):
        if not already_pil:
            img = self.tensor2pil(img)
        img = img.resize((out_w, out_h), Image.LANCZOS)  # PIL has x and y reversed from torch
        img = self.pil2tensor(img)  # [C, H, W]
        img = img.permute(1, 2, 0)  # [H, W, C]
        return img

    def load_intrinsics(self, transform):
        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downsample
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downsample
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.img_w / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.img_h / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downsample) if 'cx' in transform else (self.img_w / 2)
        cy = (transform['cy'] / self.downsample) if 'cy' in transform else (self.img_h / 2)

        intrinsics = Intrinsics(width=self.img_w, height=self.img_h, focal_x=fl_x, focal_y=fl_y, center_x=cx,
                                center_y=cy)

        return intrinsics

    def load_from_disk(self):
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            poses, imgs, timestamps = [], [], []
            len_time = 0
            for frame in tqdm(meta['frames'], desc=f'Loading {self.split} data'):
                timestamp = int(frame['file_path'].split('t')[-1].split('_')[0])
                cam_id = int(frame['file_path'].split('r')[-1])
                if timestamp > len_time:
                    len_time = timestamp
                # Decide whether to keep this frame or not
                if np.random.uniform() > self.subsample:
                    continue
                if cam_id >= self.max_test_cameras:
                    continue
                pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
                # Fix file-path
                f_path = os.path.join(self.datadir, frame['file_path'])
                if '.' not in os.path.basename(f_path):
                    f_path += '.png'  # so silly...
                img = Image.open(os.path.join(self.datadir, f_path))
                img = self.load_image(img, int(round(img.size[0] / self.downsample)), int(round(img.size[1] / self.downsample)), already_pil=True)
                imgs.append(img)
                poses.append(pose)
                timestamps.append(timestamp)
            self.img_h, self.img_w = imgs[0].shape[:2]
            intrinsics = self.load_intrinsics(meta)
        self.len_time = len_time
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3]
        poses = torch.stack(poses, 0)  # [N, 4, 4]
        timestamps = torch.from_numpy(np.array(timestamps))  # [N]
        log.info(f"360Dataset - Loaded {self.split} set from {self.datadir}: {len(imgs)} "
                 f"images of size {imgs[0].shape}. {intrinsics}")
        return poses, imgs, intrinsics, timestamps

    def load_from_disk_ndc(self):
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
            # poses_bounds = poses_bounds[0:]
            # videopaths = videopaths[0:]

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # [N_cameras, 3, 5]
        self.near_fars = poses_bounds[:, -2:]  # [N_cameras, 2]

        # Step 1: rescale focal length according to training resolution
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
        scale_factor = near_original * 0.75  # 0.75 is the default parameter, but 1 seems way better??
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
                if np.random.uniform() > self.subsample:
                    continue
                # Only keep frame zero, for debugging
                # if frame_idx > 0:
                #     len_time = 300
                #     break
                # Do any downsampling on the image
                img = self.load_image(frame, intrinsics.height, intrinsics.width)
                imgs.append(img)
                timestamps.append(frame_idx)
                all_poses.append(torch.from_numpy(poses[camera_id]).float())
        self.len_time = len_time
        imgs = torch.stack(imgs, 0)  # [N, H, W, 3]
        timestamps = torch.from_numpy(np.array(timestamps))  # [N]
        poses = all_poses  # [N, 3, 4]

        log.info(f"LLFFDataset - Loaded {self.split} set from {self.datadir}: {len(imgs)} "
                 f"images of size {imgs[0].shape}. {intrinsics}")

        return poses, imgs, intrinsics, timestamps


    # TODO: this should be refactored once I know the right thing to do for NDC vs 360
    def init_rays(self, imgs):
        if self.ndc:
            # ray directions for all pixels, same for all images (same H, W, focal)
            directions = get_ray_directions_blender(self.intrinsics)  # H, W, 3

        all_rays_o, all_rays_d = [], []
        for i in range(len(self.poses)):
            if self.ndc:
                rays_o, rays_d = get_rays(directions, self.poses[i])  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(
                    intrinsics=self.intrinsics, near=1.0, rays_o=rays_o, rays_d=rays_d)
            else:
                rays = synthetic_get_rays(self.intrinsics, self.poses[i])  # [ro+rd, H, W, 3]
                # Merge N, H, W dimensions
                rays_o = rays[0, ...].reshape(-1, 3)  # [H*W, 3]
                rays_o = rays_o.to(dtype=torch.float32).contiguous()
                rays_d = rays[1, ...].reshape(-1, 3)  # [H*W, 3]
                rays_d = rays_d.to(dtype=torch.float32).contiguous()
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

def synthetic_get_rays(intrinsics, c2w) -> torch.Tensor:
    """

    :param H:
    :param W:
    :param focal:
    :param c2w:
    :return:
        Tensor of size [2, W, H, 3] where the first dimension indexes origin and direction
        of rays
    """
    i, j = torch.meshgrid(torch.arange(intrinsics.width) + 0.5, torch.arange(intrinsics.height) + 0.5, indexing='xy')
    dirs = torch.stack([
        (i - intrinsics.width * 0.5) / intrinsics.focal_x,
        -(j - intrinsics.height * 0.5) / intrinsics.focal_y,
        -torch.ones_like(i)
    ], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return torch.stack((rays_o, rays_d), dim=0)