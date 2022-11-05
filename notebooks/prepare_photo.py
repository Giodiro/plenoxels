import numpy as np
import torch
from pathlib import Path
import imageio
from typing import Optional
import os
import glob
import pandas as pd
import h5py

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
                            torch.arange(0., W, device=kinv.device))
    pixco = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

    directions = torch.matmul(pixco, kinv.T) # (H, W, 3)

    rays_d = torch.matmul(directions, pose[:3, :3].T)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # (H, W, 3)
    
    rays_o = pose[:3, -1].expand_as(rays_d) # (H, W, 3)

    return rays_o, rays_d


train_images = {"sacre" : 1179,
            "trevi" : 1689}

test_images = {"sacre" : 21,
            "trevi" : 19}

datadir = "/work3/frwa/data/phototourism/trevi"
name = os.path.basename(datadir)
split = "train"


# read all files in the tsv first (split to train and test later)
tsv = glob.glob(os.path.join(datadir, '*.tsv'))[0]
scene_name = os.path.basename(tsv)[:-4]
files = pd.read_csv(tsv, sep='\t')
files = files[~files['id'].isnull()] # remove data without id
files.reset_index(inplace=True, drop=True)

files = files[files["split"]==split]

# the first N idx are for training, the rest are for testing
# we therefore need to know how many training frames there are
# such that the test frames can get unique time/appearance indices
n_train_images = train_images[name]
        
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


size = 0
for w,h in res:
    size += w*h
    
#with h5py.File("mytestfile.hdf5", "w") as f:
    #dset = f.create_dataset("mydataset", (size,3+3+3+2+1), dtype='f')
dset = np.zeros((size,3+3+3+2+1), dtype=np.float32)
print(dset.shape, dset.dtype)
count = 0
for idx, impath in enumerate(imagepaths):
    image = imageio.imread(impath)
    image = image[..., :3]/255.
    image = torch.as_tensor(image, dtype=torch.float)
            
    scale = 0.05
    pose = poses[idx]
    pose = torch.as_tensor(pose, dtype=torch.float)
    pose = torch.cat([pose[:3, :3], pose[:3, 3:4]*scale], dim=-1)

    kinv = kinvs[idx]
    kinv = torch.as_tensor(kinv, dtype=torch.float)
    
    bound = bounds[idx]
    bound = torch.as_tensor(bound, dtype=torch.float)
    bound = bound * torch.as_tensor([0.9, 1.2]) * scale
    #TODO: histoggram of image resolutions 
    rays_o, rays_d = get_rays_tourism(image.shape[0], image.shape[1], kinv, pose)
        
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    image = image.reshape(-1, 3)

    if split == "test":
        timestamp = torch.tensor(idx + n_train_images)
    else:
        timestamp = torch.tensor(idx)
        
    timestamp = timestamp.expand(len(image), 1)
    bound = bound.expand(len(image), 2)
    import pdb; pdb.set_trace()
    rows = np.concatenate([image, rays_o, rays_d, bound, timestamp], axis=1)
    if count < 10:
        print(rows.shape, rows.dtype)
    
    dset[count : count + len(rows),:] = rows
    count += len(rows)
    

import os
print(os.path.join(datadir, "my_cache" "data.npy"))
os.makedirs(os.path.join(datadir, "my_cache"), exist_ok=True)
print(dset.shape, dset.dtype)
np.save(os.path.join(datadir, "my_cache" "data.npy"), dset)
