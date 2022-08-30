# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from typing import Callable, Tuple, Optional

import torch
from torch.utils.data import Dataset

from .standard_nerf_loader import load_nerf_standard_data
from ..core import Rays


class MultiviewDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit a static 3D volume from
    multiview images.

    TODO(ttakikawa): Support single-camera dynamic temporal scenes, and multi-camera dynamic temporal scenes.
    TODO(ttakikawa): Currently this class only supports sampling per image, not sampling across the entire
                     dataset. This is due to practical reasons. Not sure if it matters...
    """

    def __init__(self,
                 dataset_path: str,
                 multiview_dataset_format: str = 'standard',
                 data_resize_shape: Optional[Tuple[int, int]] = None,
                 bg_color: str = None,
                 dataset_num_workers: int = -1,
                 transform: Callable = None,
                 **kwargs
                 ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want
        to load the images unless we have to. This might change later.

        Args:
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.multiview_dataset_format = multiview_dataset_format
        self.resize_shape = data_resize_shape
        self.bg_color = bg_color
        self.dataset_num_workers = dataset_num_workers
        self.transform = transform

        self.data = None
        self.val_data = None
        self.test_data = None
        self.img_shape = None
        self.num_imgs = None

    def init(self):
        """Initializes the dataset.
        """
        self.data = self.get_images()
        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]

        self.data["imgs"] = self.data["imgs"].view(-1, self.data["imgs"].shape[-1])
        self.data["rays"] = self.data["rays"].reshape(-1, 3)

        if "depths" in self.data:
            self.data["depths"] = self.data["depths"].reshape(-1, self.data["depths"].shape[-1])
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(-1, self.data["masks"].shape[-1])

    def get_images(self, split='train', resize_shape=None):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            resize_shape (tuple(int, int)): If specified, will rescale the image to this size.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        if resize_shape is None:
            resize_shape = self.resize_shape

        if self.multiview_dataset_format == "standard":
            data = load_nerf_standard_data(self.root, split,
                                           bg_color=self.bg_color,
                                           num_workers=self.dataset_num_workers,
                                           shape=resize_shape)
        else:
            raise ValueError(self.multiview_dataset_format)

        return data

    def get_full_val_set(self):
        if self.val_data is None:
            self.val_data = self.get_images(split='val', resize_shape=None)
        return self.val_data

    def get_full_test_set(self):
        if self.test_data is None:
            self.test_data = self.get_images(split='test', resize_shape=None)
        return self.test_data

    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx: int):
        """Returns rays and color-data for an image.
        If transform is the ray-sampler this is the same as calling `get_img_samples`
        """
        out = {'rays': self.data["rays"][idx], 'imgs': self.data["imgs"][idx]}

        if self.transform is not None:
            out = self.transform(out)

        return out

    def get_img_samples(self, idx, batch_size):
        """Returns a batch of samples from an image, indexed by idx.
        """
        ray_idx = torch.randperm(self.data["imgs"].shape[1])[:batch_size]
        return {
            'rays': Rays(origins=self.data["rays"].origins[idx, ray_idx],
                         dirs=self.data["rays"].dirs[idx, ray_idx],
                         dist_min=self.data["rays"].dist_min,
                         dist_max=self.data["rays"].dist_max),
            'imgs': self.data["imgs"][idx, ray_idx]
        }
