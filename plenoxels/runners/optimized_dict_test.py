# Given a trained plenoxel grid, optimize a dictionary of patches of desired size
from plenoxels.configs import optimize_dict_config, parse_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from plenoxels.models.grid_plenoxel import RegularGrid
from plenoxels.runners.utils import *
from plenoxels.tc_harmonics import plenoxel_sh_encoder
from plenoxels.synthetic_nerf_dataset import SyntheticNerfDataset

np.random.seed(0)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

logdir = './logs'
datadir = '/data/datasets/nerf/data/nerf_synthetic/lego'
plenoxel_grid_dir = 'plenoxel_lego_256_v2'
dictionary_dir = 'dict256_4.128'

# Reload an optimized plenoxel grid
# Load the pretrained grid
checkpoint_data = torch.load(os.path.join(logdir, plenoxel_grid_dir, "model.pt"), map_location='cpu')
plenoxel_grid = checkpoint_data["model"]["data"].squeeze()  # [data_dim, reso, reso, reso]

# Reload an optimized patch dictionary
checkpoint_data = torch.load(os.path.join(logdir, dictionary_dir, "model.pt"), map_location='cpu')
dictionary = checkpoint_data["model"]["dictionary"].squeeze()  # [data_dim, patch_reso, patch_reso, patch_reso, num_atoms]
patch_reso = dictionary.shape[1]
coarse_reso = plenoxel_grid.shape[1] // patch_reso
data_dim = dictionary.shape[0]
reso = coarse_reso * patch_reso

def encode_patches(dictionary, patches):
    # Compute the inverse dictionary
    atoms = dictionary.reshape(-1, dictionary.shape[-1])  # [patch_size, num_atoms]
    # print(atoms.device)
    # pinv = torch.linalg.pinv(atoms) # [num_atoms, patch_size]
    # print(pinv.device)
    # Apply to the patches
    vectorized_patches = patches.reshape(patches.size(0), -1) # [batch_size, patch_size]
    # weights = vectorized_patches @ pinv.T  # [batch_size, num_atoms]
    weights = torch.linalg.lstsq(atoms, vectorized_patches.T, driver="gels").solution.T
    return weights @ atoms.T

# Fill out a coarse grid using the patch dictionary to match the optimized plenoxel grid
all_patches = plenoxel_grid.unfold(1, patch_reso, patch_reso) \
         .unfold(2, patch_reso, patch_reso) \
         .unfold(3, patch_reso, patch_reso) \
         .reshape(plenoxel_grid.shape[0], -1, patch_reso, patch_reso, patch_reso)  # [data_dim, num_patches, patch_reso, patch_reso, patch_reso]
all_patches = all_patches.transpose(0,1)  # [num_patches, data_dim, patch_reso, patch_reso, patch_reso]
encoded_patches = encode_patches(dictionary, all_patches)  # [num_patches, patch_size] where num_patches = coarse_reso**3 and patch_size is patch_reso**3 * data_dim
grid = encoded_patches.reshape(coarse_reso, coarse_reso, coarse_reso, data_dim, patch_reso, patch_reso, patch_reso)   # [coarse_reso, coarse_reso, coarse_reso, data_dim, patch_reso, patch_reso, patch_reso]
grid = grid.permute(3, 0, 4, 1, 5, 2, 6).reshape(data_dim, reso, reso, reso)  # [data_dim, reso, reso, reso]

# Measure PSNR
sh_encoder = plenoxel_sh_encoder(0)
renderer = RegularGrid(reso, 1.3, 0, sh_encoder)
renderer.data.data = grid.unsqueeze(0)

test_dset = SyntheticNerfDataset(datadir, split='test', downsample=1, resolution=800, max_frames=10)
test_model(renderer, test_dset, log_dir=None, batch_size=4000, render_fn=lambda ro, rd: renderer(ro, rd), plot_type="imageio", num_test_imgs=1)
