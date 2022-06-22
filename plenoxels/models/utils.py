from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def get_intersections(rays_o, rays_d, radius: float, n_intersections: int, step_size: float):
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
    intrs_pts = rays_o[..., None, :] + rays_d[..., None, :] * intersections_trunc[..., None]
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
