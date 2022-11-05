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


@dataclass
class GridParamDescription:
    grid_coefs: nn.ParameterList
    reso: torch.Tensor
    time_reso: int = None
    time_coef: nn.Parameter = None


def init_features_param(grid_config, sh: bool) -> torch.nn.Parameter:
    assert "feature_dim" in grid_config

    reso: List[int] = grid_config["resolution"]
    try:
        in_dim = len(reso)
    except AttributeError:
        raise ValueError("Configuration incorrect: resolution must be a list.")
    assert in_dim == grid_config["input_coordinate_dim"]
    features = nn.Parameter(
        torch.zeros([grid_config["feature_dim"]] + reso[::-1]))
    if sh:
        if reso[0] > 2:
            nn.init.zeros_(features)
            features[-1].data.fill_(grid_config["init_std"])  # here init_std is repurposed as the sigma initialization
        elif reso[0] == 2:
            # Make each feature a standard basis vector
            # Feature shape is [feature_dim] + [2]*d
            nn.init.uniform_(features, a=0, b=0.1)
            feats = features.data.view(grid_config["feature_dim"], -1).permute(1, 0)  # [feature_dim, num_features]
            for i in range(grid_config["feature_dim"]-1):
                feats[i] = basis_vector(grid_config["feature_dim"], i, dense=False)
            # For trying a fixed/nonlearnable F
            # nn.init.uniform_(features, a=0, b=0)  # for learnable, works well to have a=0, b=0.1
            # feats = features.data.view(grid_config["feature_dim"], -1).permute(1, 0)  # [feature_dim, num_features]
            # for i in range(grid_config["feature_dim"]-1):
            #     feats[i] = basis_vector(grid_config["feature_dim"], i, dense=True)
            # extra_sigma_vals = [-100, 100, 1000, -1000]
            # k = 0
            # for j in range(grid_config["feature_dim"], len(feats)):
            #     feats[j] = basis_vector(grid_config["feature_dim"], i) * extra_sigma_vals[k]
            #     k = k + 1
            # feats[grid_config["feature_dim"]]
            print(feats)
            features.data = feats.permute(0, 1).reshape([grid_config["feature_dim"]] + reso[::-1])
    else:
        nn.init.normal_(features, mean=0.0, std=grid_config["init_std"])
    return features


def init_grid_param(grid_config, is_video: bool, is_appearance: bool, grid_level: int, use_F: bool = True) -> GridParamDescription:
    out_dim: int = grid_config["output_coordinate_dim"]
    grid_nd: int = grid_config["grid_dimensions"]

    reso: List[int] = grid_config["resolution"]
    try:
        in_dim = len(reso)
    except AttributeError:
        raise ValueError("Configuration incorrect: resolution must be a list.")
    pt_reso = torch.tensor(reso, dtype=torch.long)
    num_comp = math.comb(in_dim, grid_nd)
    rank: Sequence[int] = to_list(grid_config["rank"], num_comp, "rank")
    grid_config["rank"] = rank
    # Configuration correctness checks
    assert in_dim == grid_config["input_coordinate_dim"]
    if grid_level == 0:
        if is_video:
            assert in_dim == 4
        else:
            assert in_dim == 3 or in_dim == 4
    if use_F:
        assert out_dim in {1, 2, 3, 4, 5, 6, 7}
    assert grid_nd <= in_dim
    if grid_nd == in_dim:
        assert all(r == 1 for r in rank)
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        if use_F:
            # if appearance and time plane, then init as ones (static).
            if is_appearance and 3 in coo_comb:
                grid_coefs.append(
                    nn.Parameter(nn.init.ones_(torch.empty(
                        [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                    ))))
            else:
                grid_coefs.append(
                    nn.Parameter(nn.init.uniform_(torch.empty(
                        [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                    ), a=-1.0, b=1.0)))
        else:
            if is_appearance and 3 in coo_comb:
                grid_coefs.append(
                    nn.Parameter(nn.init.ones_(torch.empty(
                        [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
                    ))))
            else:
                grid_coefs.append(
                    nn.Parameter(nn.init.uniform_(torch.empty(
                        [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                    ), a=0.1, b=0.5)))
    """
    if is_appearance:
        time_reso = int(grid_config["time_reso"])

        if use_F:
            time_coef = nn.Parameter(nn.init.uniform_(
                torch.empty([out_dim * rank[0], time_reso]),
                a=-1.0, b=1.0))  # if time init is fixed at 1, then it learns a static video
        else:

            # if sh + density in grid, then we do not want appearance code to influence density
            if out_dim == 28:
                out_dim = out_dim - 1

            time_coef = nn.Parameter(nn.init.ones_(torch.empty([out_dim * rank[0], time_reso])))  # no time dependence
        return GridParamDescription(
            grid_coefs=grid_coefs, reso=pt_reso, time_reso=time_reso, time_coef=time_coef)
    """
    return GridParamDescription(
        grid_coefs=grid_coefs, reso=pt_reso)


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
