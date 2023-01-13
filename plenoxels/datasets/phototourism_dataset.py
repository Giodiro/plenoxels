import glob
import logging as log
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch

from plenoxels.datasets.base_dataset import BaseDataset
from plenoxels.datasets.intrinsics import Intrinsics
from plenoxels.ops.image.io import read_png


class PhotoTourismDataset2(BaseDataset):
    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 downsample: float = 1.0):
        # TODO: handle render split
        if ndc:
            raise NotImplementedError("PhotoTourism only handles contraction and standard.")
        if downsample != 1.0:
            raise NotImplementedError("PhotoTourism does not handle image downsampling.")
        if not os.path.isdir(datadir):
            raise ValueError(f"Directory {datadir} does not exist.")
        pt_data_file = os.path.join(datadir, f"cache_{split}.pt")
        if not os.path.isfile(pt_data_file):
            # Populate cache
            cache_data(datadir=datadir, split=split, out_fname=os.path.basename(pt_data_file))
        pt_data = torch.load(pt_data_file)

        intrinsics = [
            Intrinsics(width=img.shape[1],
                       height=img.shape[0],
                       center_x=img.shape[1] / 2,
                       center_y=img.shape[0] / 2,
                       focal_y=0,  # focals are unused, we reuse intrinsics from Matt's files.
                       focal_x=0)
            for img in pt_data["images"]
        ]
        if split == 'train':
            near_fars = torch.cat([
                pt_data["bounds"][i].expand(intrinsics[i].width * intrinsics[i].height, 2)
                for i in range(len(intrinsics))
            ], dim=0)
            camera_ids = torch.cat([
                pt_data["camera_ids"][i].expand(intrinsics[i].width * intrinsics[i].height, 1)
                for i in range(len(intrinsics))
            ])
            images = torch.cat([img.view(-1, 3) for img in pt_data["images"]], 0)
            rays_o = torch.cat([ro.view(-1, 3) for ro in pt_data["rays_o"]], 0)
            rays_d = torch.cat([rd.view(-1, 3) for rd in pt_data["rays_d"]], 0)
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
                 f"{self.num_images} images of sizes between {min(self.img_h)}x{min(self.img_w)} "
                 f"and {max(self.img_h)}x{max(self.img_w)}. "
                 f"Images loaded: {self.imgs is not None}.")
        if self.is_contracted:
            log.info(f"Contraction parameters: global_translation={self.global_translation}, "
                     f"global_scale={self.global_scale}")
        else:
            log.info(f"Bounding box: {self.scene_bbox}")

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
            out["timestamps"] = out["timestamps"].expand(out["rays_o"].shape[0], 1)
            out["near_fars"] = out["near_fars"].expand(out["rays_o"].shape[0], 2)
        return out


def get_rays_tourism(H, W, kinv, pose):
    """
    phototourism camera intrinsics are defined by H, W and kinv.
    Args:
        H: image height
        W: image width
        kinv (3, 3): inverse of camera intrinsic
        pose (4, 4): camera extrinsic
    Returns:
        rays_o (H, W, 3): ray origins
        rays_d (H, W, 3): ray directions
    """
    yy, xx = torch.meshgrid(torch.arange(0., H, device=kinv.device),
                            torch.arange(0., W, device=kinv.device),
                            indexing='ij')
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

    directions = torch.matmul(pixco, kinv.T)  # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)

    rays_o = pose[:3, -1].expand_as(rays_d)  # (H, W, 3)

    return rays_o, rays_d


def cache_data(datadir: str, split: str, out_fname: str):
    log.info(f"Preparing cached rays for dataset at {datadir} - {split=}.")
    # read all files in the tsv first (split to train and test later)
    tsv = glob.glob(os.path.join(datadir, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()]  # remove data without id
    files.reset_index(inplace=True, drop=True)

    scale = 0.05
    files = files[files["split"] == split]

    imagepaths = sorted((Path(datadir) / "dense" / "images").glob("*.jpg"))
    imkey = np.array([os.path.basename(im) for im in imagepaths])
    idx = np.in1d(imkey, files["filename"])

    imagepaths = np.array(imagepaths)[idx]
    try:
        poses = np.load(str(Path(datadir) / "c2w_mats.npy"))[idx]
        kinvs = np.load(str(Path(datadir) / "kinv_mats.npy"))[idx]
        bounds = np.load(str(Path(datadir) / "bds.npy"))[idx]
        res = np.load(str(Path(datadir) / "res_mats.npy"))[idx]
    except FileNotFoundError as e:
        error_msg = (
            f"One of the needed Phototourism files does not exist ({e.filename}). "
            f"They can be downloaded from "
            f"https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW"
        )
        log.error(error_msg)
        raise e
    img_w = res[:, 0]
    img_h = res[:, 1]
    size = int(np.sum(img_w * img_h))
    log.info(f"Loading dataset from {datadir}. Num images={len(imagepaths)}. Total rays={size}.")

    all_images, all_rays_o, all_rays_d, all_bounds, all_camera_ids = [], [], [], [], []
    for idx, impath in enumerate(imagepaths):
        image = read_png(impath)

        pose = torch.from_numpy(poses[idx]).float()
        pose[:3, 3:4] *= scale
        kinv = torch.from_numpy(kinvs[idx]).float()
        bound = torch.from_numpy(bounds[idx]).float()
        bound = bound * torch.tensor([0.9, 1.2]) * scale

        rays_o, rays_d = get_rays_tourism(image.shape[0], image.shape[1], kinv, pose)

        camera_id = torch.tensor(idx)

        all_images.append(image.mul(255).to(torch.uint8))
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_bounds.append(bound)
        all_camera_ids.append(camera_id)

    torch.save({
        "images": all_images,
        "rays_o": all_rays_o,
        "rays_d": all_rays_d,
        "bounds": all_bounds,
        "camera_ids": all_camera_ids,
    }, os.path.join(datadir, out_fname))
