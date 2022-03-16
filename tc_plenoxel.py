from dataclasses import dataclass
import time

from typing import Tuple

import torch

import tc_harmonics


@dataclass
class Grid:
    indices: torch.Tensor
    grid: torch.Tensor


# @torch.jit.script
def tensor_linspace(start: torch.Tensor, end: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246

    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


@torch.jit.script
def safe_ceil(vector):
    return torch.ceil(vector - 1e-5)


@torch.jit.script
def safe_floor(vector):
    return torch.floor(vector + 1e-5)


# @torch.jit.script
def grid_lookup(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, grid_idx: torch.Tensor,
                grid_data: torch.Tensor) -> torch.Tensor:
    return grid_data[grid_idx[x, y, z]]


@torch.jit.script
def get_interp_weights(xs, ys, zs):
    # xs: n_pts, 1
    # ys: n_pts, 1
    # zs: n_pts, 1
    # out: n_pts, 8
    weight000 = (1 - xs) * (1 - ys) * (1 - zs)  # [n_pts]
    weight001 = (1 - xs) * (1 - ys) * zs  # [n_pts]
    weight010 = (1 - xs) * ys * (1 - zs)  # [n_pts]
    weight011 = (1 - xs) * ys * zs  # [n_pts]
    weight100 = xs * (1 - ys) * (1 - zs)  # [n_pts]
    weight101 = xs * (1 - ys) * zs  # [n_pts]
    weight110 = xs * ys * (1 - zs)  # [n_pts]
    weight111 = xs * ys * zs  # [n_pts]
    return torch.stack(
        (weight000, weight001, weight010, weight011,
         weight100, weight101, weight110, weight111), dim=1)  # [n_pts, 8]


# noinspection PyAbstractClass,PyMethodOverriding
class TrilinearInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid_data: torch.Tensor, neighbor_ids: torch.Tensor, offsets: torch.Tensor):
        # grid_data:    [n_voxels, n_channels]
        # neighbor_ids: [batch, n_intrs, 8] (long)
        # offsets:      [batch, n_intrs, 3]

        # Fetch the data from voxels we care about
        neighbor_data = grid_data[neighbor_ids]  # [batch, n_intrs, 8, channels]

        # Coalesce all intersections into the 'batch' dimension. Call batch * n_intrs == n_pts
        batch, nintrs = offsets.shape[:2]
        neighbor_data = neighbor_data.view(batch * nintrs, 8, -1)  # [n_pts, 8, channels]
        offsets = offsets.view(batch * nintrs, 3)  # [n_pts, 3]

        weights = get_interp_weights(xs=offsets[:, 0:1], ys=offsets[:, 1:2], zs=offsets[:, 2:3])

        out = torch.sum(neighbor_data * weights, dim=1)  # sum over the 8 neighbors => [n_pts, channels]
        out = out.view(batch, nintrs, -1)  # [batch, n_intersections, channels]

        ctx.weights = weights
        ctx.batch, ctx.nintrs = batch, nintrs
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # data: [batch * n_intersections, 8, channels]
        # weights: [n_pts, 8, 1]
        # grad_output: [batch, n_intersections, channels]
        batch, nintrs = ctx.batch, ctx.nintrs
        weights = ctx.weights.reshape(batch, nintrs, 8, 1)
        return grad_output.unsqueeze(2) * weights, None

    @staticmethod
    def test_autograd():
        data = torch.randn(5, 4, 8, 5).to(dtype=torch.float64).requires_grad_()
        weights = torch.randn(5, 4, 3).to(dtype=torch.float64)

        torch.autograd.gradcheck(lambda d: TrilinearInterpolate.apply(d, weights), inputs=data)


if __name__ == "__main__":
    TrilinearInterpolate.test_autograd()


@torch.jit.script
def get_intersection_ids(intersections: torch.Tensor,  # [batch, n_intersections]
                         rays_o: torch.Tensor,         # [batch, 3]
                         rays_d: torch.Tensor,         # [batch, 3]
                         grid_idx: torch.Tensor,       # [res, res, res]
                         voxel_len: float,
                         radius: float,
                         eps: float,
                         resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets_3d = torch.tensor(
        [[-1, -1, -1],
         [-1, -1, 1],
         [-1, 1, -1],
         [-1, 1, 1],
         [1, -1, -1],
         [1, -1, 1],
         [1, 1, -1],
         [1, 1, 1]], dtype=intersections.dtype, device=intersections.device)
    # Points at which the rays intersect the grid-lines.
    intrs_pts = rays_o.unsqueeze(1) + intersections.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intersections, 3]
    # Offsets
    offsets = offsets_3d.mul_(voxel_len / 2)  # [8, 3]

    # Radius: the radius of the 'world' grid

    # Go from an intersection point to its 8 neighboring voxels
    # GIAC: Here we remove clamp(-radius, radius)
    neighbors = intrs_pts.unsqueeze(2) + offsets[None, None, :, :]  # [batch, n_intersections, 8, 3]

    # Dividing one of the points in neighbors by voxel_len, gives us the grid coordinates (i.e. integers)
    # TODO: why + eps?
    neighbors_grid_coords = safe_floor(neighbors / voxel_len)

    # The actual voxel (at the center?)
    neighbor_centers = torch.clamp(
        (neighbors_grid_coords + 0.5) * voxel_len,
        min=-(radius - voxel_len / 2),
        max=radius - voxel_len / 2)  # [batch, n_intersections, 8, 3]

    neighbor_ids = torch.clamp(
        (neighbors_grid_coords + resolution / 2).to(torch.long),
        min=0, max=resolution - 1)  # [batch, n_intersections, 8, 3]

    xyzs = (intrs_pts - neighbor_centers[:, :, 0, :]) / voxel_len  # [batch, n_intersections, 3]
    neighbor_idxs = grid_idx[neighbor_ids[..., 0], neighbor_ids[..., 1], neighbor_ids[..., 2]]  # [batch, n_intersections, 8]
    return xyzs, neighbor_idxs


def values_rays(intersections: torch.Tensor,  # [batch, n_intersections]
                rays_o: torch.Tensor,  # [batch, 3]
                rays_d: torch.Tensor,  # [batch, 3]
                resolution: int,
                radius: float,
                jitter: bool,
                eps: float,
                interpolation: str,
                grid_idx: torch.Tensor,
                grid_data: torch.Tensor) -> torch.Tensor:  # [batch, n_intersections - 1, channels]
    voxel_len = radius * 2 / resolution
    if not jitter:
        with torch.autograd.no_grad():
            t_s = time.time()
            xyzs, neighbor_idxs = get_intersection_ids(intersections, rays_o, rays_d, grid_idx,
                                                       voxel_len, radius, eps, resolution)
            t_idx = time.time() - t_s
        if interpolation == 'trilinear':
            t_s = time.time()
            neighbor_data = grid_data[neighbor_idxs]  # [batch, n_intersections, 8, channels]
            intr_data = TrilinearInterpolate.apply(neighbor_data, xyzs)  # [batch, n_intersections, channels]
            # NOTE: Here we ignore the last intersection! (TODO: Why?)
            intr_data = intr_data[:, :-1, :]  # [batch, n_intersections - 1, channels]
            t_int = time.time() - t_s
            print(f"Time get_intersection_ids: {t_idx:.5f}s trilinear_interpolate: {t_int:.5f}s")
        else:
            raise NotImplementedError("%s interpolation not implemented" % (interpolation))
        return intr_data
    else:
        raise NotImplementedError("jitter")


# @torch.jit.script
def volumetric_rendering(rgb: torch.Tensor,  # [batch, n_intersections-1, 3]
                         sigma: torch.Tensor,  # [batch, n_intersections-1]
                         z_vals: torch.Tensor,  # [batch, n_intersections]
                         dirs: torch.Tensor,  # [batch, 3]
                         white_bkgd: bool = True):
    """Volumetric Rendering Function.
    Args:
      rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
      sigma: jnp.ndarray(float32), density, [batch_size, num_samples].
      z_vals: jnp.ndarray(float32), [batch_size, num_samples].
      dirs: jnp.ndarray(float32), [batch_size, 3].
      white_bkgd: bool.
    Returns:
      comp_rgb: jnp.ndarray(float32), [batch_size, 3].
      disp: jnp.ndarray(float32), [batch_size].
      acc: jnp.ndarray(float32), [batch_size].
      weights: jnp.ndarray(float32), [batch_size, num_samples]
    # Based on https://github.com/google-research/google-research/blob/d0a9b1dad5c760a9cfab2a7e5e487be00886803c/jaxnerf/nerf/model_utils.py#L166
    """
    eps = 1e-10
    batch, n_intrs = z_vals.shape
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    # dists: [batch, n_intersections - 1]
    dists = torch.diff(z_vals, n=1, dim=1) * torch.linalg.norm(dirs, ord=2, dim=-1, keepdim=True)
    omalpha = torch.exp(-torch.relu(sigma) * dists)  # [batch, n_intersections - 1]
    alpha = 1.0 - omalpha
    accum_prod = torch.cumprod(omalpha[:, :-1] + eps, dim=-1)  # [batch, n_intersections - 2]
    accum_prod = torch.cat(
        (torch.ones(batch, 1, dtype=rgb.dtype, device=rgb.device), accum_prod),
        dim=-1)  # [batch, n_intersections - 1]
    # the absolute amount of light that gets stuck in each voxel
    weights = alpha * accum_prod  # [batch, n_intersections - 1]
    # Accumulated color over the samples, ignoring background
    comp_rgb = (weights.unsqueeze(-1) * torch.sigmoid(rgb)).sum(dim=-2)  # [batch, 3]
    # Weighted average of depths by contribution to final color
    depth = (weights * z_vals[:, :-1]).sum(dim=-1)  # [batch]
    # Total amount of light absorbed along the ray
    acc = weights.sum(-1)  # [batch]
    # disparity: inverse depth.
    # equivalent to (but slightly more efficient and stable than):
    #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
    disp = torch.clamp(acc / depth, min=eps, max=1 / eps)  # TODO: not sure if it's equivalent
    # inv_eps = torch.tensor(1 / eps, dtype=rgb.dtype, device=rgb.device)
    # disp = acc / depth  # [batch]
    # disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    if white_bkgd:
        # Including the white background in the final color
        comp_rgb = comp_rgb + (1. - acc.unsqueeze(1))
    return comp_rgb, disp, acc, weights


# @torch.jit.script
def intersection_distances(
        start: torch.Tensor,  # [batch, 1]
        stop: torch.Tensor,  # [batch, 1]
        offset: torch.Tensor,  # [batch, 3] (only needed for uniform = 0)
        interval: torch.Tensor,  # [batch, 3] (only needed for uniform = 0)
        rays_o: torch.Tensor,  # [batch, 3]
        rays_d: torch.Tensor,  # [batch, 3]
        uniform: float,
        interpolation: str,
        jitter: bool,
        radius: float,
        resolution: int,
        grid_idx: torch.Tensor,
        grid_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
         - intersections [batch, n_intersections]
         - interpolated [batch, n_intersections, channels]
    """

    if uniform == 0:
        raise NotImplementedError("uniform == 0 not implemented")
    else:
        voxel_len = radius * 2 / resolution
        count = int(resolution * 3 / uniform)
        intersections = tensor_linspace(
            start=start.squeeze() + uniform * voxel_len,
            end=stop.squeeze() + uniform * voxel_len * count,
            # difference from plenoxel.py since here endpoint=True
            steps=count)
    intersections = torch.clamp(intersections, min=None, max=stop)
    interpolated = values_rays(intersections, rays_o, rays_d, resolution, radius, jitter,
                               eps=1e-5, interpolation=interpolation, grid_idx=grid_idx,
                               grid_data=grid_data)
    return intersections, interpolated


def render_rays(grid_idx: torch.Tensor,
                grid_data: torch.Tensor,
                rays: Tuple[torch.Tensor, torch.Tensor],
                resolution: int,
                radius: float = 1.3,
                harmonic_degree: int = 0,
                jitter: bool = False,
                uniform: float = 0.5,
                interpolation: str = 'trilinear'):
    voxel_len = radius * 2.0 / resolution
    assert resolution % 2 == 0  # Renderer assumes resolution is a multiple of 2
    rays_o, rays_d = rays
    # Compute when the rays enter and leave the grid
    with torch.autograd.no_grad():
        offsets_pos = (radius - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (-radius - rays_o) / rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        offsets_out = torch.maximum(offsets_pos, offsets_neg)  # [batch, 3]
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        stop = torch.amin(offsets_out, dim=-1, keepdim=True)  # [batch, 1]
        first_intersection = rays_o + start * rays_d  # [batch, 3]

        # Compute locations of ray-voxel intersections along each dimension
        interval = voxel_len / torch.abs(rays_d)
        offset_bigger = (safe_ceil(
            first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d
        offset_smaller = (safe_floor(
            first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d
        offset = torch.minimum(offset_bigger, offset_smaller)

    t_s = time.time()
    # intersections: [batch, n_intersections]
    # voxel_data:    [batch, n_intersections - 1, channels]
    intersections, voxel_data = intersection_distances(
        start=start, stop=stop, offset=offset, interval=interval, rays_o=rays_o, rays_d=rays_d,
        uniform=uniform, interpolation=interpolation, jitter=jitter, radius=radius,
        resolution=resolution, grid_idx=grid_idx, grid_data=grid_data)
    t_d = time.time() - t_s
    # Exclude density data from the eval_sh call
    t_s = time.time()
    voxel_rgb = tc_harmonics.eval_sh(
        harmonic_degree, voxel_data[..., :-1], rays_d)  # [batch, n_intersections - 1, 3]
    t_sh = time.time() - t_s
    t_s = time.time()
    rgb, disp, acc, weights = volumetric_rendering(
        voxel_rgb, voxel_data[..., -1], intersections, rays_d)
    t_v = time.time() - t_s

    intrs_pts = rays_o.unsqueeze(1) + intersections.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intersections, 3]
    neighbor_ids = torch.clamp(
        (safe_floor(intrs_pts / voxel_len) + resolution / 2).to(torch.int32),
        min=0, max=resolution - 1)  # [batch, n_intersections, 3]
    print(f"dist {t_d:.4f}s sh {t_sh:.4f}s vol {t_v:.4f}s")
    return rgb, disp, acc, weights, neighbor_ids
