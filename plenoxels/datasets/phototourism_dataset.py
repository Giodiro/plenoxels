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
from plenoxels.datasets.ray_utils import gen_camera_dirs


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

        if scene_bbox is None:
            raise ValueError("Must specify scene_bbox")
        scene_bbox = torch.tensor(scene_bbox)
        self.poses, intrinsics, file_names = load_pt_metadata(
            datadir=datadir, orientation_method=orientation_method, center_poses=center_poses,
            auto_scale_poses=auto_scale_poses, scale_factor=scale_factor, split=split)
        images, intrinsics = load_pt_images(intrinsics, file_names, downsample, split)

        # Since all images are of different shapes, we cannot generate coordinates on the fly.
        # For the training set we pre-generate all rays_o, rays_d by computing them one at a time,
        # and then concatenating. For the test dataset this is not necessary since images are
        # accessed one at a time.
        directions, origins, self.camera_ids = None, None, None
        all_images: Union[torch.Tensor, List[torch.Tensor]]
        if split == 'train':
            directions, origins, all_images, camera_ids = [], [], [], []
            for i, (itr, pose, image) in enumerate(zip(intrinsics, self.poses, images)):
                _origins, _directions = pt_gen_rays(pose, itr)
                directions.append(_directions)
                origins.append(_origins)
                all_images.append(image.view(-1, 3))
                camera_ids.append(torch.full((_directions.shape[0], ), fill_value=i, dtype=torch.float32))
            directions = torch.cat(directions)
            origins = torch.cat(origins)
            all_images = torch.cat(all_images)
            self.camera_ids = torch.cat(camera_ids)  # [tot_num_rays]
        else:
            all_images = images
            self.camera_ids = torch.arange(len(all_images), dtype=torch.int32)  # [num_images]

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

    def __getitem__(self, index):
        out, index = super().__getitem__(index, return_idxs=True)
        out["bg_color"] = torch.ones(
            (1, 3), dtype=out['rays_o'].dtype, device=out['rays_o'].device)
        out["timestamps"] = self.camera_ids[index]

        if self.split != 'train':
            out["timestamps"] = out["timestamps"].repeat(out["rays_o"].shape[0])

            # Generate rays
            img = out["imgs"]
            intrinsics = self.intrinsics[index]
            pose = self.poses[index]
            assert img.shape[0] == intrinsics.height and img.shape[1] == intrinsics.width
            ro, rd = pt_gen_rays(pose, intrinsics)
            # Split image in two parts
            img_h, img_w = intrinsics.height, intrinsics.width
            mid = img_w // 2

            out["imgs_left"] = img[:, :mid, :].reshape(-1, 3)
            out["rays_o_left"] = ro.view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
            out["rays_d_left"] = rd.view(img_h, img_w, 3)[:, :mid, :].reshape(-1, 3)
            out["rays_o"] = ro
            out["rays_d"] = rd
            out["imgs"] = img.view(-1, 3)

        out["near_fars"] = intersect_with_aabb(
            out["rays_o"], out["rays_d"], self.scene_bbox, near_plane=None)
        return out


def pt_gen_rays(pose: torch.Tensor, intrinsics: Intrinsics) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = torch.meshgrid(
        torch.arange(intrinsics.width, device="cpu"),
        torch.arange(intrinsics.height, device="cpu"),
        indexing="xy"
    )
    x, y = x.flatten(), y.flatten()
    rays_d = gen_camera_dirs(x, y, intrinsics, True)  # (num_rays, 3)
    rays_d = (rays_d[:, None, :] * pose[None, :3, :3]).sum(dim=-1)
    rays_d /= torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    rays_o = torch.broadcast_to(pose[None, :3, -1], rays_d.shape)

    return rays_o, rays_d


def load_pt_images(
    intrinsics: List[Intrinsics],
    file_names: List[str],
    downsample: float,
    split: str
):
    scaled_intrinsics = [i.scale(1 / downsample) for i in intrinsics]
    out_h = [i.height for i in scaled_intrinsics]
    out_w = [i.width for i in scaled_intrinsics]
    images = parallel_load_images(
        dset_type="phototourism",
        tqdm_title=f"Loading {split} data",
        num_images=len(file_names),
        paths=file_names,
        out_h=out_h,
        out_w=out_w,
    )
    return images, scaled_intrinsics


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

    poses[:, :3, 3] *= out_scale_factor

    # Split
    split_ids = get_pt_split_ids(datadir, split, image_filenames)
    poses = poses[split_ids]
    intrinsics = [intrinsics[i] for i in split_ids]
    image_filenames = [image_filenames[i] for i in split_ids]
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


def intersect_with_aabb(
    rays_o: torch.Tensor, rays_d: torch.Tensor, aabb: torch.Tensor, near_plane: Optional[torch.Tensor],
) -> torch.Tensor:
    """Returns collection of valid rays within a specified near/far bounding box along with a mask
    specifying which rays are valid
    Args:
        rays_o: (num_rays, 3) ray origins
        rays_d: (num_rays, 3) ray directions
        aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
        near_plane
    """
    # avoid divide by zero
    dir_fraction = 1.0 / (rays_d + 1e-6)

    # x
    t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

    nears = torch.max(
        torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
    ).values
    fars = torch.min(
        torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
    ).values

    # clamp to near plane
    near_plane = near_plane or 0.0
    nears = torch.clamp(nears, min=near_plane)
    fars = torch.maximum(fars, nears + 1e-6)

    return torch.stack((nears, fars), 1)
