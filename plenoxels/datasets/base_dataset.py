import abc
from abc import ABC
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from .intrinsics import Intrinsics


class MultiSceneDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets):
        super(MultiSceneDataset, self).__init__()
        self.datasets = list(datasets)
        self.num_datasets = len(self.datasets)
        assert self.num_datasets > 0, 'datasets should not be an empty iterable'

    def __iter__(self):
        idx = 0
        while True:
            dset_idx = idx % self.num_datasets
            batch_idx = (idx // self.num_datasets) % len(self.datasets[dset_idx])
            yield self.datasets[dset_idx][batch_idx]
            idx += 1


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 datadir: str,
                 scene_bbox: torch.Tensor,
                 split: str,
                 is_ndc: bool,
                 camtoworlds: torch.Tensor,
                 intrinsics: Intrinsics,
                 generator: Optional[torch.random.Generator],
                 batch_size: Optional[int] = None,
                 images: Optional[torch.Tensor] = None,
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

        self.camtoworlds = camtoworlds
        self.intrinsics = intrinsics
        self.images = images  # [N, H, W, 3/4]

        assert self.images.shape[0] == self.camtoworlds.shape[0]
        assert self.images.shape[1:3] == (self.intrinsics.height, self.intrinsics.width)

    @property
    def img_h(self) -> int:
        return self.intrinsics.height

    @property
    def img_w(self) -> int:
        return self.intrinsics.width

    @property
    def num_images(self) -> int:
        return self.images.shape[0]

    def update_num_rays(self, num_rays):
        self.batch_size = num_rays

    def __len__(self):
        return self.images.shape[0]

    def to(self, device):
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)

    @torch.no_grad()
    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()

        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    @abc.abstractmethod
    def fetch_data(self, index: int):
        pass

    @abc.abstractmethod
    def preprocess(self, data):
        pass
