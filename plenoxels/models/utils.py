from typing import Optional

import torch
import torch.nn.functional as F

from plenoxels.ops.interpolation import grid_sample_4d, grid_sample_1d, grid_sample_nd


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


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 1:
        grid_sampler = grid_sample_1d
    elif grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    elif grid_dim == 4:
        grid_sampler = grid_sample_4d
    else:
        grid_sampler = grid_sample_nd

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


# Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/math.py#L237
def compute_tv_norm(depths, losstype='l2', weighting=None):
    # depths [n_patches, h, w]
    v00 = depths[:, :-1, :-1]
    v01 = depths[:, :-1, 1:]
    v10 = depths[:, 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)  # In RegNerf it's actually square l2
    elif losstype == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')

    if weighting is not None:
        loss = loss * weighting[:, :-1, :-1]

    return torch.mean(loss)
