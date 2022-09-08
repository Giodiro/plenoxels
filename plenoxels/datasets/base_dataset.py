from typing import Optional, Tuple
import os

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, datadir: str, scene_bbox: torch.Tensor, split: str, is_ndc: bool):
        self.datadir = datadir
        self.name = os.path.basename(self.datadir)
        self.scene_bbox = scene_bbox
        self.split = split
        self.is_ndc = is_ndc

        self.tensors: Optional[Tuple[torch.Tensor, ...]] = None

    def set_tensors(self, *tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
