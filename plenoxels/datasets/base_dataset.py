from abc import ABC
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from .intrinsics import Intrinsics


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 scene_bbox: torch.Tensor,
                 split: str,
                 is_ndc: bool,
                 poses: torch.Tensor,
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
        self.batch_size = batch_size
        if self.split == 'train':
            assert self.batch_size is not None

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = generator

        self.poses = poses
        self.intrinsics = intrinsics
        self.imgs = imgs  # [N, H, W, 3/4]

    @property
    def img_h(self) -> int:
        return self.intrinsics.height

    @property
    def img_w(self) -> int:
        return self.intrinsics.width

    @property
    def num_images(self) -> int:
        return self.imgs.shape[0]

    @property
    def num_samples(self) -> int:
        if self.split == 'train':
            return self.img_h * self.img_w * self.num_images
        else:
            return self.num_images

    def update_num_rays(self, num_rays):
        self.batch_size = num_rays

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        else:
            return self.num_samples
