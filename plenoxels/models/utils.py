from typing import Optional
from importlib.machinery import PathFinder
from pathlib import Path

import torch
import torch.nn.functional as F
import kaolin.render.spc as spc_render

from plenoxels.ops.interpolation import grid_sample_4d, grid_sample_1d


@torch.no_grad()
def get_intersections(rays_o, rays_d, radius: float, n_intersections: int, perturb: bool = False):
    """
    Produces ray-grid intersections in world-coordinates (between -radius, +radius)
    :param rays_o:
    :param rays_d:
    :param radius:
    :param n_intersections:
    :return:
    """
    dev, dt = rays_o.device, rays_o.dtype
    n_rays = rays_o.shape[0]
    inv_rays_d = torch.reciprocal(torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d))
    offsets_pos = ( radius - rays_o) * inv_rays_d  # [batch, 3]
    offsets_neg = (-radius - rays_o) * inv_rays_d  # [batch, 3]
    offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
    offsets_out = torch.maximum(offsets_pos, offsets_neg)
    start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
    end = torch.amin(offsets_out, dim=-1, keepdim=True)

    steps = (torch.linspace(0, 1.0, n_intersections, device=dev)[None])  # [1, num_samples]
    steps = steps.expand((n_rays, n_intersections))  # [num_rays, num_samples]
    intersections = start + (end - start) * steps

    #step_size = 0.01
    #steps = torch.arange(n_intersections, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
    #steps = steps.repeat(rays_d.shape[0], 1)   # [batch, n_intrs]
    #intersections = start + steps * step_size  # [batch, n_intrs]

    deltas = intersections.diff(dim=-1, prepend=torch.zeros(intersections.shape[0], 1, device=dev) + start)

    if perturb:
        sample_dist = (end - start) / n_intersections
        intersections += (torch.rand_like(intersections) - 0.5) * sample_dist

    intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections[..., None]  # [batch, n_intrs, 3]
    mask = ((-radius <= intrs_pts) & (intrs_pts <= radius)).all(dim=-1)

    ridx = torch.arange(0, n_rays, device=dev)
    ridx = ridx[..., None].repeat(1, n_intersections)[mask]
    boundary = spc_render.mark_pack_boundaries(ridx)
    deltas = deltas[mask]
    intrs_pts = intrs_pts[mask]

    return intrs_pts, ridx, boundary, deltas


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
