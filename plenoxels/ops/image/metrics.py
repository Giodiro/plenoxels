import math

import skimage.metrics
import torch
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import lpips

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

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
        gaussian_weights=False)


def msssim(rgb, gts):
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(torch.permute(rgb[None, ...], (0, 3, 1, 2)),
                   torch.permute(gts[None, ...], (0, 3, 1, 2))).item()


__LPIPS__ = {}


def init_lpips(net_name, device):
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(rgb, gts, net_name='alex', device='cpu'):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gts = gts.permute([2, 0, 1]).contiguous().to(device)
    rgb = rgb.permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gts, rgb, normalize=True).item()


def jod(rgb_video, gt_video, fps=30):
    import pyfvvdp
    fv = pyfvvdp.fvvdp(display_name='standard_fhd', heatmap="threshold")
    # inputs are numpy(uint8)
    rgb_video = torch.from_numpy(rgb_video).float() / 255
    gt_video = torch.from_numpy(gt_video).float() / 255
    q_jod, stats = fv.predict(
        rgb_video, gt_video, dim_order="FHWC", frames_per_second=fps)
    heatmap = stats.get('heatmap', None)
    return q_jod.item(), heatmap
