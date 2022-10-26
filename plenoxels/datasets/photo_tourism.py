import numpy as np
import torch
from pathlib import Path
import imageio
from typing import Optional
import os

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

class PhotoTourismDataset(torch.utils.data.Dataset):
    """This version uses normalized device coordinates, as in LLFF, for forward-facing videos
    """
    len_time: int
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0):

        self.isg = False
        self.ist = False
        self.downsample = downsample
        self.near_far = [0.0, 1.0]
        self.keyframes = False
        self.is_ndc = False
        self.is_contracted = True
        self.lookup_time = True
        self.scene_bbox = torch.tensor([[-2.0,-2.0,-2.0], [2.0,2.0,2.0]])
        self.batch_size = batch_size
        self.training = split == 'train'
        self.name = os.path.basename(datadir)

        self.imagepaths = sorted((Path(datadir) / "dense" / "images").glob("*.jpg"))
        self.poses = np.load(Path(datadir) / "c2w_mats.npy")
        self.kinvs = np.load(Path(datadir) / "kinv_mats.npy")
        self.bounds = np.load(Path(datadir) / "bds.npy")
        res = np.load(Path(datadir) / "res_mats.npy")
        
        # first 20 images are test, next 5 for validation and the rest for training.
        # https://github.com/tancik/learnit/issues/3
        splits = {
            "test": (self.imagepaths[:20], self.poses[:20], self.kinvs[:20], self.bounds[:20], res[:20]),
            #"val": (all_imagepaths[20:25], all_poses[20:25], all_kinvs[20:25], all_bounds[20:25]),
            "train": (self.imagepaths, self.poses, self.kinvs, self.bounds, res)
        }
        self.imagepaths, self.poses, self.kinvs, self.bounds, res = splits[split]

        self.img_w = res[:, 0]
        self.img_h = res[:, 1]

        self.len_time = len(self.imagepaths)
        # self.len_time = 300  # This is true for the 10-second sequences from DyNerf

    def __len__(self):
        return len(self.imagepaths)
    
    def reset_iter(self):
        pass 
    
    def __getitem__(self, idx: int):
        
        image = imageio.imread(self.imagepaths[idx])
        image = image[..., :3]/255.
        image = torch.as_tensor(image, dtype=torch.float)

        scale = 0.05
        pose = self.poses[idx]
        pose = torch.as_tensor(pose, dtype=torch.float)
        pose = torch.cat([pose[:3, :3], pose[:3, 3:4]*scale], dim=-1)

        kinv = self.kinvs[idx]
        kinv = torch.as_tensor(kinv, dtype=torch.float)
        
        bound = self.bounds[idx]
        bound = torch.as_tensor(bound, dtype=torch.float)
        bound = bound * torch.as_tensor([0.9, 1.2]) * scale
        
        rays_o, rays_d = get_rays_tourism(image.shape[0], image.shape[1], kinv, pose)
        
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        image = image.reshape(-1, 3)
        
        if self.training:            
            indices = torch.randint(0, len(rays_o), size = [self.batch_size])
            rays_o = rays_o[indices, : ]
            rays_d = rays_d[indices, : ]
            image = image[indices, : ]
            
            timestamp = torch.tensor(idx).expand(len(rays_o))
            bound = bound.expand(len(rays_o), 2)
        else:
            timestamp = torch.tensor(idx)
            bound = bound.expand(1, 2)
            
        return {"rays_o" : rays_o, "rays_d": rays_d, "imgs": image, "near_far" : bound, "timestamps": timestamp}
    
