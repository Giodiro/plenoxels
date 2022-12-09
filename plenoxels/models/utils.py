import collections.abc
import itertools
import math
from dataclasses import dataclass
from typing import Optional, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.ops.interpolation import grid_sample_4d, grid_sample_1d, grid_sample_nd
from ..ops.activations import trunc_exp


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
    # TODO: figure out why this fails for 2^d interpolation
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

    # Do the weighting before computing TV, because otherwise there are off-by-one issues that can cause large errors
    if weighting is not None:
        depths = depths * weighting

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

    # if weighting is not None:
    #     loss = loss * weighting[:, :-1, :-1]

    return torch.mean(loss)


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg

    # v00 = t[..., :-1, :-1]
    # v01 = t[..., :-1, 1:]
    # v10 = t[..., 1:, :-1]
    # loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    # return torch.mean(loss)


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def compute_line_tv(t):
    batch_size, w = t.shape
    count_w = batch_size * (w - 1)
    w_tv = torch.square(t[..., 1:] - t[..., :w-1]).sum()
    return 2 * (w_tv / count_w)


def raw2alpha(sigma, dist):
    alpha = 1 - torch.exp(-sigma * dist)
    T = torch.cat((torch.ones(alpha.shape[0], 1, device=alpha.device),
                   torch.cumprod(1.0 - alpha, dim=-1)), dim=-1)
    # T = torch.cat((torch.ones(alpha.shape[0], 1, device=alpha.device),
    #                torch.cumprod(1.0 - alpha[:, :-1] + 1e-10, dim=-1)), dim=-1)
    #T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, device=alpha.device), 1 - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]
    return alpha, weights, T#[:, -1:]  # Return full-length T so we can use the last one for background


def init_density_activation(activation_type: str):
    if activation_type == 'trunc_exp':
        return lambda x: trunc_exp(x - 1)
    elif activation_type == 'relu':
        return F.relu
    else:
        raise ValueError(activation_type)


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int]):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=0.1, b=0.5)
        grid_coefs.append(new_grid_coef)


def basis_vector(n, k, dense=True):
    vector = torch.zeros(n)
    vector[k] = 1
    if dense:
        vector[-1] = 10
    return vector


def to_list(el, list_len, name: Optional[str] = None) -> Sequence:
    if not isinstance(el, collections.abc.Sequence):
        return [el] * list_len
    if len(el) != list_len:
        raise ValueError(f"Length of {name} is incorrect. Expected {list_len} but found {len(el)}")
    return el
