# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import glob
import os

import cv2
from PIL import Image
import logging as log
import numpy as np

""" A module for reading / writing various image formats. """


def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.

    Returns:
        (void): Writes to path.
    """
    Image.fromarray(data).save(path)


def glob_imgs(path, exts=None):
    """Utility to find images in some path.

    Args:
        path (str): Path to search images in.
        exts (list of str): List of extensions to try.

    Returns:
        (list of str): List of paths that were found.
    """
    if exts is None:
        exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


def write_video_to_file(file_name, frames):
    log.info(f"Saving video ({len(frames)} frames) to {file_name}")
    # Photo tourism image sizes differ
    sizes = np.array([frame.shape[:2] for frame in frames])
    same_size_frames = np.unique(sizes, axis=0).shape[0] == 1
    if same_size_frames:
        height, width = frames[0].shape[:2]
        video = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for img in frames:
            video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()
    else:
        height = sizes[:, 0].max()
        width = sizes[:, 1].max()
        video = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for img in frames:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            image[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w, :] = img
            video.write(image[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()
