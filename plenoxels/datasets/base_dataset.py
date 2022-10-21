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
                 generator: Optional[torch.random.Generator],
                 batch_size: Optional[int] = None,
                 imgs: Optional[torch.Tensor] = None,
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

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

        self.rays_o = rays_o
        self.rays_d = rays_d
        self.intrinsics = intrinsics
        self.imgs = imgs
        self.num_batches = self.rays_o.shape[0]

        self.perm = None

    @property
    def img_h(self) -> int:
        return self.intrinsics.height

    @property
    def img_w(self) -> int:
        return self.intrinsics.width

    def reset_iter(self):
        self.perm = torch.randperm(self.num_batches)

    def get_rand_ids(self, index, weights=None):
        assert self.batch_size is not None, "Can't get rand_ids for test split"
        if weights is not None:
            # if len(weights) >= 16777216:  # 2^24 is the max for torch.multinomial
            if len(weights) > 8000000:  # 2^24 is the max for torch.multinomial
                subset = torch.from_numpy(np.random.choice(len(weights), size=8000000))
                samples = torch.multinomial(input=weights[subset], num_samples=self.batch_size, generator=self.generator)
                return subset[samples]
            return torch.multinomial(input=weights, num_samples=self.batch_size, generator=self.generator)
        assert self.perm is not None, "Call reset_iter"
        return self.perm[index * self.batch_size: (index + 1) * self.batch_size]

    def __len__(self):
        if self.split == 'train':
            return (self.num_batches + self.batch_size - 1) // self.batch_size
        else:
            return self.num_batches

    def __getitem__(self, index, weights=None):
        if self.split == 'train':
            idxs = self.get_rand_ids(index, weights)
            out = {
                "rays_o": self.rays_o[idxs].contiguous(),
                "rays_d": self.rays_d[idxs].contiguous(),
            }
            if self.imgs is not None:
                out["imgs"] = self.imgs[idxs].contiguous()
            if weights is not None:
                return out, idxs
            return out
        else:
            out = {
                "rays_o": self.rays_o[index],
                "rays_d": self.rays_d[index],
            }
            if self.imgs is not None:
                out["imgs"] = self.imgs[index]
            return out
