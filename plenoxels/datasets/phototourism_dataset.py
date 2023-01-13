import logging as log

import numpy as np
import torch
from typing import Optional, List, Tuple, Union
import os
import glob
import pandas as pd

from plenoxels.datasets.base_dataset import BaseDataset
from plenoxels.datasets.colmap_utils import read_images_binary, read_cameras_binary
from plenoxels.datasets.data_loading import parallel_load_images
from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.datasets.ray_utils import get_rays, get_ray_directions
from plenoxels.ops.bbox_colliders import intersect_with_aabb


class PhotoTourismDataset2(BaseDataset):
    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 scale_factor: float = 3.0,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6,
                 orientation_method: str = "up",
                 center_poses: bool = True,
                 auto_scale_poses: bool = True):
        # TODO: remove params and assert against not implemented stuff (e.g. ndc)
        # TODO: handle render split
        pt_data = torch.load(os.path.join(datadir, f"cache_{split}.pt"))
        intrinsics = [
            Intrinsics(width=img.shape[1],
                       height=img.shape[0],
                       center_x=img.shape[1] / 2,
                       center_y=img.shape[0] / 2,
                       focal_y=0,  # focals are unused
                       focal_x=0)
            for img in pt_data["images"]
        ]
        if split == 'train':
            images = pt_data["images"].view(-1, 3)
            rays_o = pt_data["rays_o"].view(-1, 3)
            rays_d = pt_data["rays_d"].view(-1, 3)
            near_fars = torch.cat([
                pt_data["bounds"][i].expand(intrinsics[i].width * intrinsics[i].height, 2)
                for i in range(len(intrinsics))
            ], dim=0)
            camera_ids = torch.cat([
                pt_data["camera_ids"][i].expand(intrinsics[i].width * intrinsics[i].height, 1)
                for i in range(len(intrinsics))
            ])
        elif split == 'test':
            images = pt_data["images"]
            rays_o = pt_data["rays_o"]
            rays_d = pt_data["rays_d"]
            near_fars = pt_data["bounds"]
            camera_ids = pt_data["camera_ids"]
        else:
            raise NotImplementedError(split)

        self.num_images = len(intrinsics)
        self.camera_ids = camera_ids
        self.near_fars = near_fars

        if 'trevi' in datadir:
            self.global_translation = torch.tensor([0, 0, 0.])
            self.global_scale = torch.tensor([1., 2., 1])
        elif 'sacre' in datadir:
            self.global_translation = torch.tensor([0, 0, -1])
            self.global_scale = torch.tensor([5, 5, 3])
        elif 'brandenburg' in datadir:
            self.global_translation = torch.tensor([0, 0, -1])
            self.global_scale = torch.tensor([5, 5, 3])
        else:
            raise NotImplementedError()

        if scene_bbox is None:
            raise ValueError("Must specify scene_bbox")
        scene_bbox = torch.tensor(scene_bbox)

        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=rays_o,
            rays_d=rays_d,
            intrinsics=intrinsics,
            imgs=images,
        )
        log.info(f"PhotoTourismDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of sizes between {min(self.img_h)}x{min(self.img_w)} "
                 f"and {max(self.img_h)}x{max(self.img_w)}. "
                 f"Images loaded: {self.imgs is not None}.")

    def __getitem__(self, index):
        out, index = super().__getitem__(index, return_idxs=True)
        out["bg_color"] = torch.ones((1, 3), dtype=torch.float32)
        out["timestamps"] = self.camera_ids[index]
        out["near_fars"] = self.near_fars[index]
        if self.imgs is not None:
            out["imgs"] = out["imgs"] / 255.0  # this converts to f32

        if self.split != 'train':  # gen left-image and reshape correctly
            intrinsics = self.intrinsics[index]
            img_h, img_w = intrinsics.height, intrinsics.width
            mid = img_w // 2
            if self.imgs is not None:
                out["imgs_left"] = out["imgs"][:, :mid, :].reshape(-1, 3)
                out["rays_o_left"] = out["rays_o"].view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
                out["rays_d_left"] = out["rays_d"].view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
                out["imgs"] = out["imgs"].view(-1, 3)
            out["rays_o"] = out["rays_o"].reshape(-1, 3)
            out["rays_d"] = out["rays_d"].reshape(-1, 3)
            out["timestamps"] = out["timestamps"].repeat(out["rays_o"].shape[0])
            out["near_fars"] = out["near_fars"].repeat(out["rays_o"].shape[0])
        return out


class PhotoTourismDataset(BaseDataset):
    """This version uses normalized device coordinates, as in LLFF, for forward-facing videos
    """
    len_time: int
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 scale_factor: float = 3.0,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6,
                 orientation_method: str = "up",
                 center_poses: bool = True,
                 auto_scale_poses: bool = True):
        self.orientation_method = orientation_method
        self.center_poses = center_poses
        self.auto_scale_poses = auto_scale_poses
        self.scale_factor = scale_factor
        self.ndc_far = ndc_far

        if scene_bbox is None:
            raise ValueError("Must specify scene_bbox")
        scene_bbox = torch.tensor(scene_bbox)
        self.poses, intrinsics, file_names = load_pt_metadata(
            datadir=datadir, orientation_method=orientation_method, center_poses=center_poses,
            auto_scale_poses=auto_scale_poses, scale_factor=scale_factor, split=split)
        images, intrinsics = load_pt_images(intrinsics, file_names, downsample, split)
        self.num_images = len(intrinsics)

        # Since all images are of different shapes, we cannot generate coordinates on the fly.
        # For the training set we pre-generate all rays_o, rays_d by computing them one at a time,
        # and then concatenating. For the test dataset this is not necessary since images are
        # accessed one at a time.
        directions, origins, self.camera_ids = None, None, None
        all_images: Union[torch.Tensor, List[torch.Tensor]]
        if split == 'train':
            directions, origins, all_images, camera_ids = [], [], [], []
            for i, (itr, pose, image) in enumerate(zip(intrinsics, self.poses, images)):
                _origins, _directions = pt_gen_rays(pose, itr, ndc=ndc)
                directions.append(_directions)
                origins.append(_origins)
                all_images.append(image.view(-1, 3))
                camera_ids.append(torch.full((_directions.shape[0], ), fill_value=i, dtype=torch.int32))
            directions = torch.cat(directions)
            origins = torch.cat(origins)
            all_images = torch.cat(all_images)
            self.camera_ids = torch.cat(camera_ids)  # [tot_num_rays]
            log.info(f"Generated all {directions.shape[0]} training rays.")
        else:
            all_images = images
            self.camera_ids = torch.arange(len(all_images), dtype=torch.int32)  # [num_images]

        if 'trevi' in datadir:
            self.global_translation = torch.tensor([0, 0, 0.])
            self.global_scale = torch.tensor([1., 2., 1])
        elif 'sacre' in datadir:
            self.global_translation = torch.tensor([0, 0, -1])
            self.global_scale = torch.tensor([5, 5, 3])
        elif 'brandenburg' in datadir:
            self.global_translation = torch.tensor([0, 0, -1])
            self.global_scale = torch.tensor([5, 5, 3])
        else:
            raise NotImplementedError()

        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=origins,
            rays_d=directions,
            intrinsics=intrinsics,
            imgs=all_images,
        )
        log.info(f"PhotoTourismDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of sizes between {min(self.img_h)}x{min(self.img_w)} "
                 f"and {max(self.img_h)}x{max(self.img_w)}. "
                 f"Images loaded: {self.imgs is not None}.")

    def __getitem__(self, index):
        out, index = super().__getitem__(index, return_idxs=True)
        out["bg_color"] = torch.ones((1, 3), dtype=torch.float32)
        out["timestamps"] = self.camera_ids[index]

        if self.split != 'train':
            # Generate rays
            img = out["imgs"]
            intrinsics = self.intrinsics[index]
            pose = self.poses[index]
            assert img.shape[0] == intrinsics.height and img.shape[1] == intrinsics.width
            ro, rd = pt_gen_rays(pose, intrinsics, ndc=self.is_ndc)
            # Split image in two parts
            img_h, img_w = intrinsics.height, intrinsics.width
            mid = img_w // 2

            out["imgs_left"] = img[:, :mid, :].reshape(-1, 3)
            out["rays_o_left"] = ro.view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
            out["rays_d_left"] = rd.view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
            out["rays_o"] = ro
            out["rays_d"] = rd
            out["imgs"] = img.view(-1, 3)

            out["timestamps"] = out["timestamps"].repeat(out["rays_o"].shape[0])

        if self.is_ndc:
            out["near_fars"] = torch.tensor([[0.0, self.ndc_far]]).repeat(out["rays_o"].shape[0], 1)
        else:
            out["near_fars"] = torch.stack(intersect_with_aabb(
                rays_o=out["rays_o"], rays_d=out["rays_d"], aabb=self.scene_bbox, near_plane=0.0, training=False), 1)
            out["near_fars"] *= torch.tensor([0.8, 1.5])  # random expansion
        return out


def pt_gen_rays(pose: torch.Tensor, intrinsics: Intrinsics, ndc: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics, opengl_camera=True, add_half=False)
    rays_o, rays_d = get_rays(
        directions, pose, ndc=ndc, ndc_near=1.0, intrinsics=intrinsics, normalize_rd=True)
    return rays_o, rays_d


def load_pt_images(
    intrinsics: List[Intrinsics],
    file_names: List[str],
    downsample: float,
    split: str
):
    for i in intrinsics:
        i.scale(1 / downsample)
    out_h = [i.height for i in intrinsics]
    out_w = [i.width for i in intrinsics]
    images = parallel_load_images(
        dset_type="phototourism",
        tqdm_title=f"Loading {split} data",
        num_images=len(file_names),
        paths=file_names,
        out_h=out_h,
        out_w=out_w,
    )
    return images, intrinsics


def load_pt_metadata(
    datadir: str,
    orientation_method: str,
    center_poses: bool,
    auto_scale_poses: bool,
    scale_factor: float,
    split: str,
) -> Tuple[torch.Tensor, List[Intrinsics], List[str]]:
    cams = read_cameras_binary(os.path.join(datadir, "dense/sparse/cameras.bin"))
    imgs = read_images_binary(os.path.join(datadir, "dense/sparse/images.bin"))

    poses = []
    intrinsics = []
    image_filenames = []

    for _id, cam in cams.items():
        img = imgs[_id]

        assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

        pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        poses.append(torch.linalg.inv(pose))
        intrinsics.append(Intrinsics(
            focal_x=cam.params[0], focal_y=cam.params[1],
            center_x=cam.params[2], center_y=cam.params[3],
            width=cam.params[2] * 2, height=cam.params[3] * 2))

        image_filenames.append(os.path.join(datadir, "dense/images", img.name))

    poses = torch.stack(poses).float()
    poses[..., 1:3] *= -1

    poses, transform_matrix = auto_orient_and_center_poses(
        poses, method=orientation_method, center_poses=center_poses
    )

    # Scale poses
    out_scale_factor = 1.0
    if auto_scale_poses:
        out_scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    out_scale_factor *= scale_factor
    log.info(f"Final scale factor = {out_scale_factor}")
    poses[:, :3, 3] *= out_scale_factor

    # Split
    split_ids = get_pt_split_ids(datadir, split, image_filenames)
    poses = poses[split_ids]
    intrinsics = [intrinsics[i] for i in range(len(split_ids)) if split_ids[i]]
    image_filenames = [image_filenames[i] for i in range(len(split_ids)) if split_ids[i]]
    return poses, intrinsics, image_filenames


def get_pt_split_ids(datadir: str, split: str, image_filenames: List[str]):
    # read all files in the tsv first (split to train and test later)
    tsv = glob.glob(os.path.join(datadir, '*.tsv'))[0]
    files_df = pd.read_csv(tsv, sep='\t')
    files_df = files_df[~files_df['id'].isnull()]  # remove data without id
    files_df.reset_index(inplace=True, drop=True)
    split_files_df = files_df[files_df['split'] == split]

    base_names = [os.path.basename(fp) for fp in image_filenames]

    split_ids = torch.from_numpy(np.in1d(base_names, split_files_df['filename']))
    return split_ids


def auto_orient_and_center_poses(
    poses: torch.Tensor, method: str = "up", center_poses: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orients and centers the poses. We provide two methods for orientation: pca and up.
    pca: Orient the poses so that the principal component of the points is aligned with the axes.
        This method works well when all of the cameras are in the same plane.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_poses: If True, the poses are centered around the origin.
    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    translation = poses[..., :3, 3]

    mean_translation = torch.mean(translation, dim=0)
    translation_diff = translation - mean_translation

    if center_poses:
        translation = mean_translation
    else:
        translation = torch.zeros_like(mean_translation)

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "z":
        zdir = torch.mean(poses[:, :3, 2], dim=0)
        zdir = zdir / torch.linalg.norm(zdir)

        rotation = rotation_matrix(zdir, torch.tensor([0., 0., 1.]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError("orientation method can be 'pca', 'up', 'none'")

    return oriented_poses, transform


def rotation_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the rotation matrix that rotates vector a to vector b.
    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))
