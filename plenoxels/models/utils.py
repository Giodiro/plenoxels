from typing import Optional
from importlib.machinery import PathFinder
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.no_grad()
def get_intersections_wisp(rays, n_intersections: int, near: float, far: float):
    rays_o, rays_d = rays
    # Sample points along 1D line
    depth = torch.linspace(0, 1.0, n_intersections, device=rays_o.device)[None] + \
            (torch.rand(rays_o.shape[0], n_intersections, device=rays_o.device) / n_intersections)
    depth = depth ** 2

    # Normalize between near and far plane
    depth *= far - near
    depth += near

    # Batched generation of samples
    samples = rays_o[:, None] + rays_d[:, None] * depth[..., None]
    deltas = depth.diff(dim=-1, prepend=torch.full((depth.shape[0], 1), near, device=depth.device))
    # Hack together pidx, mask, ridx, boundaries, etc
    pidx = self.query(samples.reshape(-1, 3), level=level).reshape(-1, num_samples)
    mask = (pidx > -1)
    ridx = torch.arange(0, pidx.shape[0], device=pidx.device)
    ridx = ridx[...,None].repeat(1, num_samples)[mask]
    boundary = spc_render.mark_pack_boundaries(ridx)
    pidx = pidx[mask]
    #depth_samples = depth[None].repeat(rays.origins.shape[0], 1)[mask][..., None]
    depth_samples = depth[mask][..., None]

    #deltas = spc_render.diff(depth_samples, boundary).reshape(-1, 1)
    deltas = deltas[mask].reshape(-1, 1)

    samples = samples[mask][:,None]


@torch.no_grad()
def get_intersections(rays_o, rays_d, radius: float, n_intersections: int, step_size: float):
    """
    Produces ray-grid intersections in world-coordinates (between -radius, +radius)
    :param rays_o:
    :param rays_d:
    :param radius:
    :param n_intersections:
    :param step_size:
    :return:
    """
    dev, dt = rays_o.device, rays_o.dtype
    rays_d_nodiv0 = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    offsets_pos = (radius - rays_o) / rays_d_nodiv0  # [batch, 3]
    offsets_neg = (-radius - rays_o) / rays_d_nodiv0  # [batch, 3]
    offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
    start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]

    steps = torch.arange(n_intersections, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
    steps = steps.repeat(rays_d.shape[0], 1)   # [batch, n_intrs]
    intersections = start + steps * step_size  # [batch, n_intrs]
    intersections_trunc = intersections[:, :-1]
    intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections_trunc[..., None]  # [batch, n_intrs, 3]
    # noinspection PyUnresolvedReferences
    mask = ((-radius <= intrs_pts) & (intrs_pts <= radius)).all(dim=-1)
    return intrs_pts, intersections, mask


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
    if grid_dim == 2:
        interp = F.grid_sample(
            grid[None, ...],  # [1, feature_dim, reso, reso]
            coords[None, None, ...],  # [1, 1, n, 2]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, feature_dim]
    elif grid_dim == 3:
        interp = F.grid_sample(
            grid[None, ...],  # [1, feature_dim, reso, reso, reso]
            coords[None, None, None, ...],  # [1, 1, 1, n, 3]
            align_corners=True,
            mode='bilinear', padding_mode='border').squeeze().permute(1, 0)  # [n, feature_dim]
    elif grid_dim == 4:
        if not hasattr(torch.ops, 'plenoxels'):
            spec = PathFinder().find_spec("c_ext", [str(Path(__file__).resolve().parents[1])])
            torch.ops.load_library(spec.origin)
        interp = torch.ops.plenoxels.grid_sample_4d(
            grid[None, ...],  # [1, feature_dim, reso, reso, reso, reso]
            coords[None, None, None, None, ...],  # [1, 1, 1, n, 4]
            0,  # interpolation_mode
            1,  # padding_mode
            True,  # align_corners
        ).squeeze().permute(1, 0)  # [n, feature_dim]
    else:
        raise ValueError("grid_dim can be 2, 3 or 4.")
    return interp
