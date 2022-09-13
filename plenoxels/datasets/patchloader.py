import torch
torch.manual_seed(0)
import numpy as np

# TODO: finish debugging/integrating with run_video

class PatchLoader():
    def __init__(self, dset, batch_size):
        self.dataset = dset
        self.batch_size = np.sqrt(batch_size)  # patch side length
        self.n_frames = self.dataset.rgbs.shape[0]
        self.height = self.dataset.rgbs.shape[1]
        self.width = self.dataset.rgbs.shape[2]
        print(f'n_frames is {self.n_frames}, height is {self.height}, width is {self.width}')
        self.i_grid, self.j_grid = torch.meshgrid(torch.arange(self.batch_size), torch.arange(self.batch_size), indexing='xy')

    def next(self):
        # Get the next batch randomly
        # Choose a random i, j coordinate from a random frame
        frame_idx = torch.randint(low=0, high=self.n_frames, size=(1,)).long()
        i_coord = torch.randint(low=0, high=self.height, size=(1,))
        j_coord = torch.randint(low=0, high=self.width, size=(1,))

        # Get a batch-sized box around the random i, j coordinate
        i_coords = torch.clamp(i_coord + self.i_grid, min=0, max=self.height - 1).long()
        j_coords = torch.clamp(j_coord + self.j_grid, min=0, max=self.width - 1).long()

        # Exract the batch
        rays_o = self.dataset.rays_o[frame_idx, i_coords, j_coords, ...]
        rays_d = self.dataset.rays_d[frame_idx, i_coords, j_coords, ...]
        rgbs = self.dataset.rgbs[frame_idx, i_coords, j_coords, ...]
        timestamps = self.dataset.timestamps[frame_idx, i_coords, j_coords]
        return (rays_o, rays_d, rgbs, timestamps)
