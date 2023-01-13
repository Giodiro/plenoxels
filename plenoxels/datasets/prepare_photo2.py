import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from plenoxels.ops.image.io import read_png


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


def prepare(datadir, split):
    dset_name = os.path.basename(datadir)

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

    poses = np.load(Path(datadir) / "c2w_mats.npy")[idx]
    kinvs = np.load(Path(datadir) / "kinv_mats.npy")[idx]
    bounds = np.load(Path(datadir) / "bds.npy")[idx]
    res = np.load(Path(datadir) / "res_mats.npy")[idx]

    img_w = res[:, 0]
    img_h = res[:, 1]
    size = int(np.sum(img_w * img_h))
    print(f"Loading dataset {dset_name}. Num images={len(imagepaths)}. Total rays={size}.")

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
    }, os.path.join(datadir, f"cache_{split}.pt"))


if __name__ == "__main__":
    datadir = "/data/DATASETS/phototourism/sacre_coeur"
    for split in ["train", "test"]:
        prepare(datadir, split)
