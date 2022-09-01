from typing import Optional

import torch
import torch.nn.functional as F

from plenoxels.ops.interpolation import grid_sample_4d, grid_sample_1d



def interp_regular(grid, pts, align_corners=True, padding_mode='border'):
    """Interpolate data on a regular grid at the given points.

    :param grid:
        Tensor of size [1, ch, res, res, res].
    :param pts:
        Tensor of size [1, n, 1, 1, 3]. Points must be normalized between -1 and 1.
    :return:
        Tensor of size [ch, n] or [n] if the `ch` dimension is of size 1.
    """
    pts = pts.to(dtype=grid.dtype, copy=False)
    interp_data = F.grid_sample(
        grid, pts, mode='bilinear', align_corners=align_corners, padding_mode=padding_mode)  # [1, ch, n, 1, 1]
    interp_data = interp_data.squeeze()  # [ch, n] or [n] if ch is 1
    return interp_data


def positional_encoding(pts, dirs, num_freqs_p: int, num_freqs_d: Optional[int] = None):
    """
    pts : N, 3
    dirs : N, 3
    returns: N, 3 * 2 * (num_freqs_p + num_freqs_d)
    """
    if num_freqs_d is None:
        num_freqs_d = num_freqs_p
    freq_bands_d = 2 ** torch.arange(num_freqs_d, device=dirs.device)
    freq_bands_p = 2 ** torch.arange(num_freqs_p, device=pts.device)
    out_p = pts[..., None] * freq_bands_p * torch.pi
    out_d = dirs[..., None] * freq_bands_d * torch.pi
    out_p = out_p.view(-1, num_freqs_p * 3)
    out_d = out_d.view(-1, num_freqs_d * 3)

    return torch.cat((torch.sin(out_p), torch.cos(out_p), torch.sin(out_d), torch.cos(out_d)), dim=-1)


def pos_encode(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    bands = 2 ** torch.arange(num_freqs, device=x.device)
    out = x[..., None] * bands * torch.pi
    out = out.view(-1, num_freqs * 3)

    return torch.cat((torch.sin(out), torch.cos(out)), dim=-1)


def ensure_list(el, expand_size: Optional[int] = None) -> list:
    if isinstance(el, list):
        return el
    elif isinstance(el, tuple):
        return list(el)
    else:
        if expand_size:
            return [el] * expand_size
        return [el]


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 1:
        interp = grid_sample_1d(
            grid,  # [B, feature_dim, reso]
            coords,  # [B, n, 1]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
    elif grid_dim == 2:
        interp = F.grid_sample(
            grid,  # [B, feature_dim, reso, reso]
            coords[:, None, ...],  # [B, 1, n, 2]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
    elif grid_dim == 3:
        interp = F.grid_sample(
            grid,  # [B, feature_dim, reso, reso, reso]
            coords[:, None, None, ...],  # [B, 1, 1, n, 3]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
    elif grid_dim == 4:
        interp = grid_sample_4d(
            grid,  # [B, feature_dim, reso, reso, reso, reso]
            coords[:, None, None, None, ...],  # [B, 1, 1, 1, n, 4]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
    else:
        raise ValueError("grid_dim can be 1, 2, 3 or 4.")
    return interp
