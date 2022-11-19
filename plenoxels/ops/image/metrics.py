# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import math

import skimage.metrics
import torch
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure

""" A module for image based metrics """


def psnr(rgb, gts):
    """Calculate the PSNR metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.Tensor): Image tensor of shape [H,W3]

    Returns:
        (float): The PSNR score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    mse = torch.mean((rgb[..., :3] - gts[..., :3]) ** 2).item()
    return 10 * math.log10(1.0 / mse)


def ssim(rgb, gts):
    """Calculate the SSIM metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.Tensor): Image tensor of shape [H,W,3]
        gts (torch.Tensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The SSIM score
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return skimage.metrics.structural_similarity(
        rgb[..., :3].cpu().numpy(),
        gts[..., :3].cpu().numpy(),
        channel_axis=2,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5)


def msssim(rgb, gts):
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    return ms_ssim(torch.permute(rgb[None, ...], (0, 3, 1, 2)),
                   torch.permute(gts[None, ...], (0, 3, 1, 2))).item()
