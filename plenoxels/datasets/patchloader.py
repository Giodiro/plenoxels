import torch
import logging as log

# Take extra rays (from unsupervised views) and produce random batches of patches, for regularization
# PatchLoader should work for either static or dynamic scenes; for dynamic scenes each ray is associated with a timestamp


# Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py#L277
class PatchLoader():
    def __init__(self, rays_o, rays_d, len_time=None, batch_size=2000, patch_size=8, generator=None):  # batch size in RegNerf defaults to 4096
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

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

        log.info(f"Initialized PatchLoader with batch-size of {self.n_patches} patches of size {self.patch_size}")

    def __getitem__(self, item):
        """Get the next batch of patches randomly.

        This generates an infinite iterator (`item` is ignored)
        """
        generator = self.generator
        # Choose which frame to draw each patch from
        frame_idxs = torch.randint(low=0, high=self.n_frames, size=(self.n_patches, 1), generator=generator)

        # Sample start locations
        x0 = torch.randint(0, self.width - self.patch_size + 1, size=(self.n_patches, 1, 1), generator=generator)
        y0 = torch.randint(0, self.height - self.patch_size + 1, size=(self.n_patches, 1, 1), generator=generator)
        xy0 = torch.cat((x0, y0), dim=-1)  # [n_patches, 1, 2]
        patch_ids = xy0 + torch.stack(
            torch.meshgrid(torch.arange(self.patch_size), torch.arange(self.patch_size), indexing='xy'),
            dim=-1).reshape(1, -1, 2)  # [n_patches, patch_size^2, 2]

        # Extract the patches from rayso and raysd
        patch_rays_o = self.rays_o[frame_idxs, patch_ids[..., 1], patch_ids[..., 0], :].view(-1, self.patch_size, self.patch_size, 3)
        patch_rays_d = self.rays_d[frame_idxs, patch_ids[..., 1], patch_ids[..., 0], :].view(-1, self.patch_size, self.patch_size, 3)

        if self.len_time is not None:
            timestamps = torch.randint(0, self.len_time, size=(self.n_patches, ), generator=generator)
            return dict(patch_rays_o=patch_rays_o, patch_rays_d=patch_rays_d, patch_timestamps=timestamps)
        return dict(patch_rays_o=patch_rays_o, patch_rays_d=patch_rays_d)

