from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

import tc_harmonics


@dataclass
class Grid:
    indices: torch.Tensor
    grid: torch.Tensor


@torch.jit.script
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
def get_intersection_ids(intersections: torch.Tensor,  # [batch, n_intersections]
                         rays_o: torch.Tensor,  # [batch, 3]
                         rays_d: torch.Tensor,  # [batch, 3]
                         grid_idx: torch.Tensor,  # [res, res, res]
                         voxel_len: float,
                         radius: float,
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
    # TODO: why + eps?0
    neighbors_grid_coords = torch.floor_(neighbors.div_(voxel_len).add_(1e-5))

    # The actual voxel (at the center?)
    neighbor_centers = torch.clamp(
        (neighbors_grid_coords + 0.5) * voxel_len,
        min=-(radius - voxel_len / 2),
        max=radius - voxel_len / 2)  # [batch, n_intersections, 8, 3]

    neighbor_ids = torch.clamp_(
        neighbors_grid_coords.add_(resolution / 2).to(torch.long),
        min=0, max=resolution - 1)  # [batch, n_intersections, 8, 3]

    xyzs = intrs_pts.sub_(neighbor_centers[:, :, 0, :]).div_(voxel_len)
    neighbor_idxs = grid_idx[
        neighbor_ids[..., 0],
        neighbor_ids[..., 1],
        neighbor_ids[..., 2]
    ]  # [batch, n_intersections, 8]
    return xyzs, neighbor_idxs


@torch.jit.script
def get_intersections(rays_o: torch.Tensor,
                      rays_d: torch.Tensor,
                      resolution: int,
                      radius: float = 1.3,
                      uniform: float = 0.5,):
    """
    :param rays_o:
        Ray origin. Tensor of shape [batch, 3]
    :param rays_d
        Ray direction. Tensor of shape [batch, 3]
    :param resolution:
        Resolution of the voxel grid (maximum length of the grid in any direction)
    :param radius:
        Half the resolution, but in the real-world (not in integer coordinates)
    :param uniform:
        No idea what this is for
    :return:
        intersections: the intersection points [batch, n_intersections]
    """
    with torch.autograd.no_grad():
        voxel_len = radius * 2 / resolution
        count = int(resolution * 3 / uniform)

        offsets_pos = (radius - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (-radius - rays_o) / rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        offsets_out = torch.maximum(offsets_pos, offsets_neg)  # [batch, 3]
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        stop = torch.amin(offsets_out, dim=-1, keepdim=True)  # [batch, 1]

        intersections = tensor_linspace(
            start=start.squeeze() + uniform * voxel_len,
            end=start.squeeze() + uniform * voxel_len * count,
            # difference from plenoxel.py since here endpoint=True
            steps=count)  # [batch, n_intersections]
        intersections = torch.clamp_(intersections, min=None, max=stop)
    return intersections


@torch.jit.script
def get_interp_weights(xs, ys, zs):
    # xs: n_pts, 1
    # ys: n_pts, 1
    # zs: n_pts, 1
    # out: n_pts, 8
    weights = torch.empty(xs.shape[0], 8, dtype=xs.dtype, device=xs.device)
    weights[:, 0] = (1 - xs) * (1 - ys) * (1 - zs)  # [n_pts]
    weights[:, 1] = (1 - xs) * (1 - ys) * zs  # [n_pts]
    weights[:, 2] = (1 - xs) * ys * (1 - zs)  # [n_pts]
    weights[:, 3] = (1 - xs) * ys * zs  # [n_pts]
    weights[:, 4] = xs * (1 - ys) * (1 - zs)  # [n_pts]
    weights[:, 5] = xs * (1 - ys) * zs  # [n_pts]
    weights[:, 6] = xs * ys * (1 - zs)  # [n_pts]
    weights[:, 7] = xs * ys * zs  # [n_pts]
    return weights


# noinspection PyAbstractClass,PyMethodOverriding
class TrilinearInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, neighbor_data: torch.Tensor, offsets: torch.Tensor):
        # neighbor_data: [batch, n_intrs, 8, channels]
        # offsets:       [batch, n_intrs, 3]
        # out:           [batch, n_intrs, channels]

        # Coalesce all intersections into the 'batch' dimension. Call batch * n_intrs == n_pts
        batch, nintrs = offsets.shape[:2]
        neighbor_data = neighbor_data.view(batch * nintrs, 8, -1)  # [n_pts, 8, channels]
        offsets = offsets.view(batch * nintrs, 3)  # [n_pts, 3]

        weights = get_interp_weights(xs=offsets[:, 0], ys=offsets[:, 1], zs=offsets[:, 2]).unsqueeze(-1)  # [n_pts, 8, 1]

        out = torch.einsum('bik, bik -> bk', neighbor_data, weights)
        out = out.view(batch, nintrs, -1)  # [batch, n_intrs, n_ch]

        ctx.weights = weights
        ctx.batch, ctx.nintrs = batch, nintrs
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # weights:      [n_pts, 8, 1]
        # grad_output:  [batch, n_intrs, n_channels]
        # out:          [batch, n_intrs, 8, n_channels]
        batch, nintrs = ctx.batch, ctx.nintrs
        weights = ctx.weights.reshape(batch, nintrs, 8, 1)
        return (grad_output.unsqueeze(2) * weights), None, None

    @staticmethod
    def test_autograd():
        data = torch.randn(5, 4, 8, 6).to(dtype=torch.float64).requires_grad_()
        weights = torch.randn(5, 4, 3).to(dtype=torch.float64)

        torch.autograd.gradcheck(lambda d: TrilinearInterpolate.apply(d, weights),
                                 inputs=data)


if __name__ == "__main__":
    TrilinearInterpolate.test_autograd()


# noinspection PyAbstractClass,PyMethodOverriding
class CumProdVolRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha: torch.Tensor):
        """
        :param ctx:
            Autograd context
        :param alpha:
            [batch, n_intrs - 1] parameter
        :return:
            abs_light: [batch, n_intrs - 1] tensor
        """
        dt, dev = alpha.dtype, alpha.device
        batch, nintrs = alpha.shape

        # Optimized memory copies here
        cum_memc = torch.ones(batch, nintrs + 1, dtype=dt, device=dev)  # [batch, n_intrs]
        torch.cumprod(1 - alpha + 1e-10, dim=-1, out=cum_memc[:, 1:])  # [batch, n_intrs - 1]
        cum_light_ex = cum_memc[:, :-1]  # [batch, n_intrs - 1]
        # the absolute amount of light that gets stuck in each voxel
        abs_light = alpha * cum_light_ex  # [batch, n_intersections - 1]

        ctx.cum_memc = cum_memc
        ctx.save_for_backward(alpha)

        return abs_light

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors

        bwa_cont = torch.zeros_like(ctx.cum_memc)               # [batch, n_intrs]
        torch.mul(alpha, grad_output, out=bwa_cont[:, :-1])     # [batch, n_intrs-1]
        b_cum_light_ex = bwa_cont[:, 1:]                        # [batch, n_intrs-1]

        # CumProd (assume no zeros!) - cum_light -> alpha
        w = b_cum_light_ex.mul_(ctx.cum_memc[:, 1:])            # [batch, n_intrs-1]
        b_alpha = torch.div(
            w.flip(-1).cumsum(-1).flip(-1),
            (1 - alpha).add_(1e-10),
        ).neg_()  # [batch, n_intrs - 1]
        b_alpha.add_(grad_output.mul_(ctx.cum_memc[:, :-1]))

        return b_alpha,

    @staticmethod
    def test_autograd():
        alpha = torch.randn(16, 8).to(dtype=torch.float64).requires_grad_()
        torch.autograd.gradcheck(lambda d: CumProdVolRender.apply(d), inputs=alpha)


if __name__ == "__main__":
    CumProdVolRender.test_autograd()


@torch.jit.script
def volumetric_rendering(rgb: torch.Tensor,  # [batch, n_intersections-1, 3]
                         sigma: torch.Tensor,  # [batch, n_intersections-1, 1]
                         intersections: torch.Tensor,  # [batch, n_intersections]
                         rays_d: torch.Tensor,  # [batch, 3]
                         white_bkgd: bool = True):
    # Volumetric rendering
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    alpha = 1 - torch.exp(-torch.relu(sigma) * dists)     # alpha: [batch, n_intrs-1]
    # the absolute amount of light that gets stuck in each voxel
    # This quantity can be used to threshold the intersections which must be processed (only if
    # abs_light > threshold). Often the variable is called 'weights'
    abs_light = CumProdVolRender.apply(alpha)
    acc_map = abs_light.sum(-1)  # [batch]

    # Accumulated color over the samples, ignoring background
    rgb = torch.sigmoid(rgb)  # [batch, n_intrs-1, 3]
    rgb_map = (abs_light.unsqueeze(-1) * rgb).sum(dim=-2)  # [batch, 3]

    if white_bkgd:
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map


def compute_irregular_grid(grid_data: torch.Tensor,
                           grid_idx: torch.Tensor,
                           rays_d: torch.Tensor,
                           rays_o: torch.Tensor,
                           radius: float,
                           resolution: int,
                           uniform: float,
                           harmonic_degree: int,
                           sh_encoder,
                           white_bkgd: bool) -> torch.Tensor:
    with torch.autograd.no_grad():
        voxel_len = radius * 2 / resolution
        intersections = get_intersections(rays_o=rays_o, rays_d=rays_d, resolution=resolution,
                                          radius=radius, uniform=uniform)  # [batch, n_intrs]
        intersections_trunc = intersections[:, :-1]
        interp_dirs, neighbor_ids = get_intersection_ids(
            intersections_trunc, rays_o, rays_d, grid_idx, voxel_len, radius, resolution)

    batch, nintrs = interp_dirs.shape[:2]
    n_ch = grid_data.shape[-1]

    # Fetch neighbors
    neighbor_data = torch.gather(grid_data, dim=0, index=neighbor_ids.view(-1, 1).expand(-1, n_ch))
    neighbor_data = neighbor_data.view(batch, nintrs, 8, n_ch)  # [batch, n_intrs, 8, n_ch]

    # Interpolation
    interp_data = TrilinearInterpolate.apply(neighbor_data, interp_dirs)  # [batch, n_intrs, n_ch]

    # Spherical harmonics
    sh_mult = sh_encoder(rays_d)  # [batch, ch/3]
    # sh_mult : [batch, ch/3] => [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
    sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]
    interp_rgb = interp_data[..., :-1].view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
    rgb_data = torch.sum(sh_mult * interp_rgb, dim=-1)  # [batch, nintrs, 3]

    sigma_data = interp_data[..., -1]  # [batch, n_intrs-1, 1]

    rgb_map = volumetric_rendering(rgb_data, sigma_data, intersections, rays_d, white_bkgd)
    return rgb_map


def compute_with_hashgrid(hg, rays_d: torch.Tensor, rays_o: torch.Tensor, radius: float,
                          resolution: int, uniform: float, harmonic_degree: int, sh_encoder,
                          white_bkgd: bool):
    with torch.autograd.no_grad():
        intersections = get_intersections(rays_o=rays_o, rays_d=rays_d, resolution=resolution,
                                          radius=radius, uniform=uniform)  # [batch, n_intrs]
        intersections_trunc = intersections[:, :-1]  # [batch, n_intrs - 1]

        # Intersections in the real world
        intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intrs - 1, 3]
        batch, nintrs = intrs_pts.shape[:2]
        # Normalize to -1, 1 range (this can probably be done without computing minmax) TODO: This is wrong, and it's unclear what HG wants
        intrs_pts = (intrs_pts - intrs_pts.min()) / (radius) - 1

    interp_data = hg(intrs_pts.view(-1, 3))  # [batch * n_intrs - 1, n_ch]
    interp_data = interp_data.view(batch, nintrs, -1)

    # Split the channels in density (sigma) and RGB. Here we ignore any extra channels which
    # may be present in interp_data
    sh_dim = ((harmonic_degree + 1) ** 2) * 3
    interp_rgb_data = interp_data[..., :sh_dim]
    sigma_data = interp_data[..., sh_dim]

    # Deal with RGB data
    sh_mult = sh_encoder(rays_d)
    # sh_mult : [batch, ch/3] => [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
    sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]
    rgb_sh = interp_rgb_data.view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
    rgb_data = torch.sum(sh_mult * rgb_sh, dim=-1)  # [batch, nintrs, 3]

    rgb_map = volumetric_rendering(rgb_data, sigma_data, intersections, rays_d, white_bkgd)
    return rgb_map


def compute_grid(grid_data: torch.Tensor,
                 rays_d: torch.Tensor,
                 rays_o: torch.Tensor,
                 radius: float,
                 resolution: int,
                 uniform: float,
                 harmonic_degree: int,
                 sh_encoder,
                 white_bkgd: bool) -> torch.Tensor:
    """
    :param grid_data:
        [res*res*res, ch].
    :param rays_d:
        Ray direction. Tensor of shape [batch, 3]
    :param rays_o:
        Ray origin. [batch, 3]
    :param radius:
        Real-world grid radius
    :param resolution:
    :param uniform
    :param harmonic_degree:
        Determines how many spherical harmonics are being used.
    :param white_bkgd:

    :return:
        rgb     : Computed color for each ray [batch, 3]
    """
    with torch.autograd.no_grad():
        intersections = get_intersections(rays_o=rays_o, rays_d=rays_d, resolution=resolution,
                                          radius=radius, uniform=uniform)  # [batch, n_intrs]
        intersections_trunc = intersections[:, :-1]  # [batch, n_intrs - 1]

        # Intersections in the real world
        intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intrs - 1, 3]
        # Normalize to -1, 1 range (this can probably be done without computing minmax) TODO: This is wrong
        intrs_pts = (intrs_pts - intrs_pts.min()) / (radius) - 1

    # Interpolate grid-data at intersection points (trilinear)
    intrs_pts = intrs_pts.unsqueeze(0).unsqueeze(0)  # [1, 1, batch, n_intrs - 1, 3]
    grid_data = grid_data.view(resolution, resolution, resolution, -1).permute(3, 0, 1, 2)  # ch, res, res, res
    # interp_data : [1, ch, 1, batch, n_intrs - 1]
    interp_data = F.grid_sample(
        grid_data.unsqueeze(0), intrs_pts,
        mode='bilinear', align_corners=True)  # [1, ch, 1, batch, n_intrs - 1]
    interp_data = interp_data.permute(0, 2, 3, 4, 1).squeeze()  # [batch, n_intrs - 1, ch]

    batch, nintrs = interp_data.shape[:2]

    # Split the channels in density (sigma) and RGB.
    sigma_data = interp_data[..., -1]
    interp_rgb_data = interp_data[..., :-1]

    # Deal with RGB data
    sh_mult = sh_encoder(rays_d)
    # sh_mult : [batch, ch/3] => [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
    sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]
    rgb_sh = interp_rgb_data.view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
    rgb_data = torch.sum(sh_mult * rgb_sh, dim=-1)  # [batch, nintrs, 3]

    rgb_map = volumetric_rendering(rgb_data, sigma_data, intersections, rays_d, white_bkgd)
    return rgb_map


def plenoxel_shell_encoder(harmonic_degree):
    num_sh = (harmonic_degree + 1) ** 2

    def encode(rays_d):
        out = torch.empty(rays_d.shape[0], num_sh, dtype=rays_d.dtype, device=rays_d.device)
        out = tc_harmonics.eval_sh_bases(harmonic_degree, rays_d, out)
        return out
    return encode
