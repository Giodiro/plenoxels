from dataclasses import dataclass
import time

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
def safe_floor(vector):
    return torch.floor(vector + 1e-5)


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

    #xyzs = (intrs_pts - neighbor_centers[:, :, 0, :]) / voxel_len  # [batch, n_intersections, 3]
    xyzs = intrs_pts.sub_(neighbor_centers[:, :, 0, :]).div_(voxel_len)
    neighbor_idxs = grid_idx[
        neighbor_ids[..., 0],
        neighbor_ids[..., 1],
        neighbor_ids[..., 2]
    ]  # [batch, n_intersections, 8]
    return xyzs, neighbor_idxs


@torch.jit.script
def fetch_intersections(grid_idx: torch.Tensor,
                        rays_o: torch.Tensor,
                        rays_d: torch.Tensor,
                        resolution: int,
                        radius: float = 1.3,
                        uniform: float = 0.5,
                        interpolation: str = 'trilinear'):
    """
    :param grid_idx:
        Integer indices in the 3D voxel grid. The size will depend on splitting and pruning,
        but this will always be a 3-dimensional long tensor.
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
    :param interpolation:
        Always equal to 'trilinear'
    :return:
        weights: trilinear interpolation weights for each intersection [batch, n_intersections, 3]
        neighbor_ids: grid-coordinate IDs of each intersections' neighbors [batch, n_intersections, 8]
        intersections: the intersection points [batch, n_intersections]
    """
    if uniform == 0:
        raise NotImplementedError(f"uniform: {uniform}")
    if interpolation != "trilinear":
        raise NotImplementedError(f"interpolation: {interpolation}")
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
    xyzs, neighbor_ids = get_intersection_ids(intersections, rays_o, rays_d, grid_idx,
                                              voxel_len, radius, resolution)

    return xyzs, neighbor_ids, intersections


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
        #out = torch.sum(neighbor_data * weights, dim=1)  # sum over the 8 neighbors => [n_pts, channels]
        out = out.view(batch, nintrs, -1)  # [batch, n_intersections, channels]

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

        # TODO: The gradient is essentially sparse. There is no point in computing the gradient for all of data, it should be much
        #       quicker to return the gradient with respect to only the data contained in neighbor_ids. Then we can essentially avoid
        #       the call to index_put!

    @staticmethod
    def test_autograd():
        data = torch.randn(5, 4, 8, 6).to(dtype=torch.float64).requires_grad_()
        weights = torch.randn(5, 4, 3).to(dtype=torch.float64)

        torch.autograd.gradcheck(lambda d: TrilinearInterpolate.apply(d, weights),
                                 inputs=data)


if __name__ == "__main__":
    TrilinearInterpolate.test_autograd()


# noinspection PyAbstractClass,PyMethodOverriding
class ComputeIntersection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid_data: torch.Tensor, neighbor_ids: torch.Tensor, offsets: torch.Tensor, rays_d: torch.Tensor, intersections: torch.Tensor, sh_deg: int):
        # neighbor_data: [batch, n_intrs - 1, 8, channels]
        # offsets:       [batch, n_intrs - 1, 3]
        # rays_d:        [batch, 3]
        # intersections: [batch, n_intrs]
        dt = offsets.dtype
        dev = offsets.device
        print("fwd start %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # Remove last intersection
        neighbor_ids = neighbor_ids[:, :-1, :]
        offsets = offsets[:, :-1, :].contiguous()
        batch, nintrs = offsets.shape[:2]

        # Fetch neighbors
        neighbor_data = grid_data[neighbor_ids]
        print("neighbors fetched %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        ######################################
        # ########## Interpolation ######### #
        ######################################
        neighbor_data = neighbor_data.view(batch * nintrs, 8, -1)  # [n_pts, 8, channels]
        offsets = offsets.view(batch * nintrs, 3)  # [n_pts, 3]
        weights = get_interp_weights(xs=offsets[:, 0], ys=offsets[:, 1], zs=offsets[:, 2]).unsqueeze(-1)  # [n_pts, 8, 1]
        interp_data = torch.einsum('bik, bik -> bk', neighbor_data, weights)

        interp_data = interp_data.view(batch, nintrs, -1)  # [batch, n_intersections, channels]
        interp_datal = interp_data.split(3, dim=-1)   # Seq[batch, n_intrs, 3 or 1]
        print("interp %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        ######################################
        # ###### Spherical harmonics  ###### #
        ######################################
        rgb_data = torch.zeros(batch, nintrs, 3, dtype=dt, device=dev)  # [batch, n_intrs-1, 3]
        rgb_data = tc_harmonics.sh_fwd_apply_list(interp_datal[:-1], rays_d, out=rgb_data, deg=sh_deg)
        sigma_data = interp_datal[-1]  # [batch, n_intrs-1, 1]
        print("sh %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        ######################################
        # ###### Volumetric rendering ###### #
        ######################################
        # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
        dists = torch.diff(intersections, n=1, dim=1) \
                     .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
        sigma_data = sigma_data.squeeze()  # SAVED for bwd so relu can't be inplace
        alpha = 1 - torch.exp_(torch.relu(sigma_data).neg_().mul_(dists))     # alpha: [batch, n_intrs-1]
        # Optimized memory copies here
        cum_memc = torch.ones(batch, nintrs + 1, dtype=dt, device=dev)  # [batch, n_intrs]
        cum_light = torch.cumprod(1 - alpha + 1e-10, dim=-1, out=cum_memc[:, 1:])  # [batch, n_intrs - 1]
        cum_light_ex = cum_memc[:, :-1]  # [batch, n_intrs - 1]

        # the absolute amount of light that gets stuck in each voxel
        abs_light = alpha * cum_light_ex  # [batch, n_intersections - 1]

        # # Accumulated color over the samples, ignoring background
        rgb_data = torch.sigmoid_(rgb_data)
        # rgb_data  : [batch, n_intrs-1, 3]
        # abs_light : [batch, n_intrs-1]
        # The following 3 lines compute the same thing. TODO: test which one is fastest
        # [batch, 3, n_intrs-1] * [batch, n_intrs-1, 1] = [batch, 3, 1]
        comp_rgb = torch.bmm(rgb_data.permute(0, 2, 1), abs_light.unsqueeze(-1)).squeeze()  # [batch, 3]
        # comp_rgb = torch.einsum('bik, bik->bk', rgb_data, abs_light.unsqueeze(-1))  # [batch, 3]
        # comp_rgb = (weights.unsqueeze(-1) * torch.sigmoid(rgb)).sum(dim=-2)  # [batch, 3]
        print("out %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        ctx.batch, ctx.n_intrs, ctx.n_ch = batch, nintrs, neighbor_data.shape[-1]
        ctx.sh_deg = sh_deg

        ctx.cum_light = cum_light
        ctx.rgb_data = rgb_data
        ctx.sigma_data = sigma_data
        ctx.cum_light_ex = cum_light_ex
        ctx.alpha = alpha
        ctx.dists = dists
        ctx.weights = weights
        ctx.rays_d = rays_d
        ctx.neighbor_data = neighbor_data  # Only saved for reusing the buffer
        ctx.neighbor_ids = neighbor_ids
        ctx.grid_data_size = grid_data.shape
        return comp_rgb

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output:  [batch, n_intrs - 1]
        # out:          [batch, n_intrs - 1, 8, channels]
        batch, n_intrs, n_ch = ctx.batch, ctx.n_intrs, ctx.n_ch
        dt, dev = grad_output.dtype, grad_output.device

        # out_data = torch.zeros(batch, n_intrs, n_ch, dtype=dt, device=dev)
        # out_datal = torch.split(out_data, 3, dim=-1)
        print("BWD start %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # bmm
        abs_light = ctx.alpha * ctx.cum_light_ex  # batch, n_intrs-1

        grad_output = grad_output.unsqueeze(-1)  # [batch, 3, 1]
        # goes into first element of out_datal. This operation is an outer product.
        # [batch, n_intrs-1, 1] * [batch, 1, 3] => [batch, n_intrs-1, 3]
        b_crgb_rgbdata = torch.bmm(abs_light.unsqueeze(-1), grad_output.transpose(1, 2))
        # b_crgb_rgbdata = torch.bmm(grad_output, abs_light.unsqueeze(1)).transpose(1, 2)
        # [batch, n_intrs-1, 3] * [batch, 3, 1] => [batch, n_intrs-1]
        b_crgb_alight = torch.bmm(ctx.rgb_data, grad_output).squeeze()

        # Sigmoid on b_crgb_rgb_data (NOTE: aten has sigmoid_backward)
        b_crgb_rgbdata.mul_((1 - ctx.rgb_data) * ctx.rgb_data)

        # We can now create the `out_data` tensor [batch, n_intrs, n_ch]
        out_data = b_crgb_rgbdata.tile((1, 1, (n_ch // 3) + 1))[:, :, :-2]
        out_datal = torch.split(out_data, 3, dim=-1)

        # Multiply with both operands needing gradient. It looks weird because of mergin this op with random reshapings
        bwa_cont = torch.zeros(batch, n_intrs + 1, dtype=dt, device=dev)
        torch.mul(ctx.alpha, b_crgb_alight, out=bwa_cont[:, :-1])
        b_weights_clight = ctx.cum_light_ex * b_crgb_alight

        # Random reshapings
        b_cum_light_ex = bwa_cont[:, 1:]  # [batch, n_intrs - 1]

        # 1. CumProd (Assume no zeros!) - cum_light -> alpha
        w = b_cum_light_ex.mul_(ctx.cum_light)  # [batch, n_intrs - 1]
        b_sigma_data = torch.div(
            w.flip(-1).cumsum(-1).flip(-1),
            (1 - ctx.alpha).add_(1e-10),
            out=out_datal[-1].squeeze()
        )  # [batch, n_intrs - 1]

        # 2. alpha -> sigma_data
        b_sigma_data.sub_(b_weights_clight).mul_(1 - ctx.alpha).mul_(ctx.dists).neg_()
        mask = ctx.sigma_data <= 0
        b_sigma_data[mask] = 0.
        b_sigma_data.unsqueeze_(2)  # [batch, n_intrs - 1, 1]
        print("rendering %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # 3. Spherical harmonics (from b_crgb_rgbdata: [batch, n_intrs-1, 3])
        tc_harmonics.sh_bwd_apply_list(out_datal[:-1], dirs=ctx.rays_d, deg=ctx.sh_deg)
        print("sh %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # 4. Interpolation
        weights = ctx.weights.view(batch, n_intrs, 8, 1)
        # [batch, n_intrs, 1, n_ch] * [batch, n_intrs, 8, 1]
        neighbor_data = ctx.neighbor_data.view(batch, n_intrs, 8, n_ch)
        torch.mul(out_data.unsqueeze(2), weights, out=neighbor_data)
        print("interp %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # [n, n_ch]
        grid_data_grad = torch.zeros(*ctx.grid_data_size, dtype=dt, device=dev)
        print("before fail %.2fGB" % (torch.cuda.memory_allocated() / 2**30))
        grid_data_grad.index_put_((ctx.neighbor_ids, ), neighbor_data, accumulate=True)

        return grid_data_grad, None, None, None, None, None

    @staticmethod
    def test_autograd():
        batch = 6
        data = torch.randn(batch, 2, 8, 4).to(dtype=torch.float64).requires_grad_()
        offsets = torch.randn(batch, 2, 3).to(dtype=torch.float64)
        rays_d = torch.randn(batch, 3).to(dtype=torch.float64)
        intersections = torch.randn(batch, 3).to(dtype=torch.float64)
        deg = 0

        torch.autograd.gradcheck(lambda d: ComputeIntersection.apply(d, offsets, rays_d, intersections, deg),
                                 inputs=data)


if __name__ == "__main__":
    ComputeIntersection.test_autograd()


@torch.jit.script
def volumetric_rendering(rgb: torch.Tensor,  # [batch, n_intersections-1, 3]
                         sigma: torch.Tensor,  # [batch, n_intersections-1, 1]
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
    sigma = sigma.squeeze()  # [batch, n_intrs]
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    # dists: [batch, n_intersections - 1]
    dists = torch.diff(z_vals, n=1, dim=1).mul_(torch.linalg.norm(dirs, ord=2, dim=-1, keepdim=True))
    omalpha = torch.exp(-torch.relu(sigma) * dists)  # [batch, n_intersections - 1]
    alpha = 1.0 - omalpha
    accum_prod = torch.cumprod(omalpha[:, :-1] + eps, dim=-1)  # [batch, n_intersections - 2]
    accum_prod = torch.cat(
        (torch.ones(batch, 1, dtype=sigma.dtype, device=sigma.device), accum_prod),
        dim=-1)  # [batch, n_intersections - 1]
    # the absolute amount of light that gets stuck in each voxel
    weights = alpha * accum_prod  # [batch, n_intersections - 1]
    # Accumulated color over the samples, ignoring background
    comp_rgb = torch.einsum('bik, bik->bk', torch.sigmoid(rgb), weights.unsqueeze(-1))  # [batch, 3]
    # comp_rgb = (weights.unsqueeze(-1) * torch.sigmoid(rgb)).sum(dim=-2)  # [batch, 3]
    # Weighted average of depths by contribution to final color
    depth = (weights * z_vals[:, :-1]).sum(dim=-1)  # [batch]
    # Total amount of light absorbed along the ray
    acc = weights.sum(-1)  # [batch]
    # disparity: inverse depth.
    # equivalent to (but slightly more efficient and stable than):
    #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
    # disp = torch.clamp(acc / depth, min=eps, max=1 / eps)  # TODO: not sure if it's equivalent
    inv_eps = torch.tensor(1 / eps, dtype=sigma.dtype, device=sigma.device)
    disp = acc / depth  # [batch]
    disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    if white_bkgd:
        # Including the white background in the final color
        comp_rgb = comp_rgb + (1. - acc.unsqueeze(1))
    return comp_rgb, disp, acc, weights


def compute_intersection_results(interp_weights: torch.Tensor,
                                 neighbor_data: torch.Tensor,
                                 rays_d: torch.Tensor,
                                 intersections: torch.Tensor,
                                 harmonic_degree: int,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param interp_weights:
        Weights for trilinear interpolation [batch, n_intrs, 3].
    :param neighbor_data:
        Data (rgb and density) at all neighborhoods of the intersections [batch, n_intrs, 8, n_ch]
    :param rays_d:
        Ray direction. Tensor of shape [batch, 3]
    :param intersections:
        Intersection points of the rays with the grid. [batch, n_intrs]
    :param harmonic_degree:
        Determines how many spherical harmonics are being used.

    :return:
        rgb     : Computed color for each ray [batch, 3]
        disp    : disparity (inverse depth)   [batch]
        acc     : total amount of ligght absorbed by ray [batch]
        weights : absolute amount of light that gets stuck in each voxel [batch, n_intrs]
    """
    t_s = time.time()
    interp_data = TrilinearInterpolate.apply(
        neighbor_data, interp_weights)  # [batch, n_intersections, channels]
    # TODO: Why ignore the last intersection?
    interp_data = interp_data[:, :-1, :]  # [batch, n_intersections - 1, channels]
    interp_data = interp_data.split(3, dim=2)  # Tuple[tensors]
    t_int = time.time() - t_s

    # Exclude density data from the eval_sh call
    t_s = time.time()
    voxel_rgb = tc_harmonics.SphericalHarmonics.apply(
        interp_data[:-1], rays_d, harmonic_degree)  # [batch, n_intersections - 1, 3]
    # voxel_rgb = tc_harmonics.eval_sh(
    #     harmonic_degree, interp_data[..., :-1], rays_d)  # [batch, n_intersections - 1, 3]
    t_sh = time.time() - t_s
    t_s = time.time()
    rgb, disp, acc, weights = volumetric_rendering(
        voxel_rgb, interp_data[-1], intersections, rays_d)
    t_v = time.time() - t_s

    print(f"Compute results: interpolate {t_int*1000:.2f}ms    harmonics {t_sh*1000:.2f}ms    "
          f"volumetric {t_v*1000:.2f}ms")
    return rgb, disp, acc, weights

