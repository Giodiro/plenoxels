import numpy as np
import torch
from pathlib import Path
import imageio
from typing import Optional
import os
import glob
import pandas as pd
import h5py

train_images = {"sacre" : 1179,
            "trevi" : 1689}

test_images = {"sacre" : 21,
            "trevi" : 19}

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


def get_rays_tourism_single(xx, yy, kinv, pose):
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
        if self.is_contracted:
            self.scene_bbox = torch.tensor([[-2.0,-2.0,-2.0], [2.0,2.0,2.0]])
        else:
            self.scene_bbox = torch.tensor([[-1.0,-1.0,-1.0], [1.0,1.0,1.0]])
        self.batch_size = batch_size
        self.training = split == 'train'
        self.name = os.path.basename(datadir)
        self.datadir = datadir
        # TODO: tune these for each dataset
        self.global_translation = torch.tensor([0, 0, -1])
        self.global_scale = torch.tensor([2.2, 2.2, 3])

        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.datadir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)
        
        
        # the first N idx are for training, the rest are for testing
        # we therefore need to know how many training frames there are
        # such that the test frames can get unique time/appearance indices
        self.n_train_images = train_images[self.name]
             
        self.files = self.files[self.files["split"]==split]
    
        self.imagepaths = sorted((Path(datadir) / "dense" / "images").glob("*.jpg"))
        imkey = np.array([os.path.basename(im) for im in self.imagepaths])
        idx = np.in1d(imkey, self.files["filename"])
    
        self.imagepaths = np.array(self.imagepaths)[idx]
            
        self.poses = np.load(Path(datadir) / "c2w_mats.npy")[idx]
        self.kinvs = np.load(Path(datadir) / "kinv_mats.npy")[idx]
        self.bounds = np.load(Path(datadir) / "bds.npy")[idx]
        res = np.load(Path(datadir) / "res_mats.npy")[idx]
        
        self.img_w = res[:, 0]
        self.img_h = res[:, 1]
        
        if self.training:
            data = np.load(os.path.join(datadir, f'my_cachedata.npy'))
            self.data = torch.from_numpy(data)
            print(data.shape, data.dtype)
        
        self.size = np.sum(self.img_w * self.img_h)
        self.len_time = train_images[self.name] + test_images[self.name]
        print(f"==> in total there are {self.len_time} images")

    def __len__(self):
        if self.training:
            return len(self.data)

        return len(self.imagepaths)
    
    def reset_iter(self):
        pass

    def __getitem__(self, idx: int):
        
        # use data in the buffers
        if self.training:
            data = self.data[idx, :]

            return {'imgs': data[:3],
                    'rays_o': data[3:6], 
                      'rays_d': data[6:9], 
                      'near_far': data[9:11],
                      'timestamps': data[11],
                      }

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
        #TODO: histoggram of image resolutions 
        rays_o, rays_d = get_rays_tourism(image.shape[0], image.shape[1], kinv, pose)
        
        mid = image.shape[1]//2
        
        image_left = image[:, :mid, :].reshape(-1, 3)
        rays_o_left = rays_o[:, :mid, :].reshape(-1, 3)
        rays_d_left = rays_d[:, :mid, :].reshape(-1, 3)
    
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        image = image.reshape(-1, 3)

        timestamp = torch.tensor(idx + self.n_train_images)
        bound = bound.expand(1, 2)
        
        out = {"rays_o" : rays_o, "rays_d": rays_d, "imgs": image, "near_far" : bound, "timestamps": timestamp}
    
        out["rays_o_left"] = rays_o_left
        out["rays_d_left"] = rays_d_left
        out["imgs_left"] =  image_left
    
        return out
    


