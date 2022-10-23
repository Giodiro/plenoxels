from abc import ABC
import os
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset

from .intrinsics import Intrinsics


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 scene_bbox: torch.Tensor,
                 split: str,
                 is_ndc: bool,
                 is_contracted: bool,
                 rays_o: torch.Tensor,
                 rays_d: torch.Tensor,
                 intrinsics: Intrinsics,
                 batch_size: Optional[int] = None,
                 imgs: Optional[torch.Tensor] = None,
                 sampling_weights: Optional[torch.Tensor] = None,
                 ):
        self.datadir = datadir
        self.name = os.path.basename(self.datadir)
        self.scene_bbox = scene_bbox
        self.split = split
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.batch_size = batch_size
        if self.split == 'train':
            assert self.batch_size is not None
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.imgs = imgs
        self.num_samples = self.imgs.shape[0]
        self.intrinsics = intrinsics
        self.sampling_weights = sampling_weights
        if self.sampling_weights is not None:
            assert len(self.sampling_weights) == self.num_samples, (
                f"Expected {self.num_samples} sampling weights but given {len(self.sampling_weights)}."
            )
        self.sampling_batch_size = 80_000_000
        self.perm = None

    @property
    def img_h(self) -> int:
        return self.intrinsics.height

    @property
    def img_w(self) -> int:
        return self.intrinsics.width

    def reset_iter(self):
        if self.sampling_weights is None:
            self.perm = torch.randperm(self.num_samples)

    def get_rand_ids(self, index):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if self.sampling_weights is not None:
            num_weights = len(self.sampling_weights)
            if num_weights > self.sampling_batch_size:
                # Take a uniform random sample first, then according to the weights
                subset = torch.randint(0, num_weights, size=(self.sampling_batch_size,), dtype=torch.int64)
                samples = torch.multinomial(
                    input=self.sampling_weights[subset], num_samples=self.batch_size)
                return subset[samples]
            return torch.multinomial(
                input=self.sampling_weights, num_samples=self.batch_size)
        return self.perm[index * self.batch_size: (index + 1) * self.batch_size]

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        else:
            return self.num_samples

    def __getitem__(self, index, return_idxs: bool = False):
        if self.split == 'train':
            index = self.get_rand_ids(index)
        out = {
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
        }
        if self.imgs is not None:
            out["imgs"] = self.imgs[index]
        if return_idxs:
            return out, index
        return out
