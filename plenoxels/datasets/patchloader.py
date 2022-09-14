import torch
torch.manual_seed(0)
import numpy as np

# Take extra rays (from unsupervised views) and produce random batches of patches, for regularization
# PatchLoader should work for either static or dynamic scenes; for dynamic scenes each ray is associated with a timestamp

# Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py#L277
class PatchLoader():
    def __init__(self, rays_o, rays_d, len_time=None, batch_size=2000, patch_size=8):  # batch size in RegNerf defaults to 4096
        """
        rays_o: [n_frames, height, width, 3]
        rays_d: [n_frames, height, width, 3]
        timestamps (optional): [n_frames, height, width]
        """
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.len_time = len_time
        self.n_patches = batch_size // (patch_size ** 2)
        self.patch_size = patch_size
        self.n_frames = self.rays_o.shape[0]
        self.height = self.rays_o.shape[1]
        self.width = self.rays_o.shape[2]
        self.i_grid, self.j_grid = torch.meshgrid(torch.arange(self.patch_size), torch.arange(self.patch_size), indexing='xy')

    # Get the next batch of patches randomly
    def next(self):
        # Choose which frame to draw each patch from
        frame_idxs = torch.randint(low=0, high=self.n_frames, size=(self.n_patches,))

        # Choose a random i, j coordinate from each random frame
        i_coords = torch.randint(low=0, high=self.height, size=(self.n_patches,))
        j_coords = torch.randint(low=0, high=self.width, size=(self.n_patches,))

        # Get a patch-sized box around each random i, j coordinate
        i_coords = torch.clamp(i_coords[:, None, None] + self.i_grid[None, ...], min=0, max=self.height - 1).long()
        j_coords = torch.clamp(j_coords[:, None, None] + self.j_grid[None, ...], min=0, max=self.width - 1).long()
        coords = torch.stack([i_coords, j_coords], dim=-1).reshape(self.n_patches, -1, 2)

        # Extract the batch
        rays_o = self.rays_o[frame_idxs[:,None], coords[...,0], coords[...,1]].reshape(self.n_patches, self.patch_size, self.patch_size, -1)
        rays_d = self.rays_d[frame_idxs[:,None], coords[...,0], coords[...,1]].reshape(self.n_patches, self.patch_size, self.patch_size, -1)
        
        # Choose which timestep to draw each patch from
        if self.len_time is not None:
            timestamps = torch.randint(low=0, high=self.len_time, size=(self.n_patches,))
            return (rays_o, rays_d, timestamps)
        return (rays_o, rays_d)

