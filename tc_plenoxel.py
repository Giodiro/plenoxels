from dataclasses import dataclass
import time

from typing import Tuple, List

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


@torch.jit.script
def interp_bwd(weights: torch.Tensor, neighbor_data: torch.Tensor, sh_data: List[torch.Tensor]) -> torch.Tensor:
    # [batch, n_intrs, 1, n_ch] * [batch, n_intrs, 8, 1]
    j = 0
    for i in range(len(sh_data)):
        torch.mul(sh_data[i].unsqueeze(2), weights, out=neighbor_data.narrow(3, j, sh_data[i].shape[-1]))
        j += sh_data[i].shape[-1]
    return neighbor_data


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
    dt, dev = interp_dirs.dtype, interp_dirs.device
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

    # Volumetric rendering
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    sigma_data = sigma_data.squeeze()
    alpha = 1 - torch.exp(-torch.relu(sigma_data) * dists)     # alpha: [batch, n_intrs-1]
    cum_light = CumProdVolRender.apply(alpha)
    # cum_light = torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)  # [batch, n_intrs - 2]
    # cum_light = torch.cat(
    #     (torch.ones(batch, 1, dtype=dt, device=dev), cum_light), dim=-1)  # [batch, n_intrs - 1]

    # the absolute amount of light that gets stuck in each voxel
    abs_light = alpha * cum_light  # [batch, n_intersections - 1]
    acc_map = abs_light.sum(-1)  # [batch]

    # Accumulated color over the samples, ignoring background
    rgb_data = torch.sigmoid(rgb_data)  # [batch, n_intrs-1, 3]
    rgb_map = (abs_light.unsqueeze(-1) * rgb_data).sum(dim=-2)  # [batch, 3]

    if white_bkgd:
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map


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
        if torch.cuda.is_available():
            print("fwd start %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # Remove last intersection
        neighbor_ids = neighbor_ids[:, :-1, :].contiguous()
        offsets = offsets[:, :-1, :].contiguous()
        batch, nintrs = offsets.shape[:2]
        n_ch = grid_data.shape[-1]

        # Fetch neighbors
        # neighbor_data = grid_data[neighbor_ids]
        neighbor_data = torch.gather(grid_data, dim=0, index=neighbor_ids.view(-1, 1).expand(-1, n_ch))
        neighbor_data = neighbor_data.view(batch, nintrs, 8, n_ch)
        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
            print("interp %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        ######################################
        # ###### Spherical harmonics  ###### #
        ######################################
        rgb_data = torch.zeros(batch, nintrs, 3, dtype=dt, device=dev)  # [batch, n_intrs-1, 3]
        rgb_data = tc_harmonics.sh_fwd_apply_list(interp_datal[:-1], rays_d, out=rgb_data, deg=sh_deg)
        sigma_data = interp_datal[-1]  # [batch, n_intrs-1, 1]
        if torch.cuda.is_available():
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
        #comp_rgb = torch.bmm(rgb_data.permute(0, 2, 1), abs_light.unsqueeze(-1)).squeeze()  # [batch, 3]
        comp_rgb = torch.einsum('bik, bik->bk', rgb_data, abs_light.unsqueeze(-1))  # [batch, 3]
        # comp_rgb = (weights.unsqueeze(-1) * torch.sigmoid(rgb)).sum(dim=-2)  # [batch, 3]
        if torch.cuda.is_available():
            print("out %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        ctx.batch, ctx.n_intrs, ctx.n_ch = batch, nintrs, neighbor_data.shape[-1]
        ctx.sh_deg = sh_deg

        ctx.cum_light = cum_light
        ctx.rgb_data = rgb_data
        ctx.sigma_data_mask = sigma_data <= 0
        ctx.cum_light_ex = cum_light_ex
        ctx.alpha = alpha
        ctx.dists = dists
        ctx.weights = weights
        ctx.rays_d = rays_d
        ctx.neighbor_data = neighbor_data  # Only saved for reusing the buffer
        ctx.interp_datal = interp_datal    # Only for the buffer
        ctx.interp_data = interp_data      # Only for the buffer
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
        if torch.cuda.is_available():
            print("BWD start %.2fGB" % (torch.cuda.memory_allocated() / 2**30))
        # s1 = torch.cuda.Stream()
        # s2 = torch.cuda.Stream()

        # bmm
        abs_light = ctx.alpha * ctx.cum_light_ex  # batch, n_intrs-1

        grad_output = grad_output.unsqueeze(-1)  # [batch, 3, 1]
        # goes into first element of out_datal. This operation is an outer product.
        # [batch, n_intrs-1, 1] * [batch, 1, 3] => [batch, n_intrs-1, 3]
        b_crgb_rgbdata = ctx.interp_datal[0]
        b_crgb_rgbdata.copy_(abs_light.unsqueeze(-1).expand_as(b_crgb_rgbdata))
        b_crgb_rgbdata.mul_(grad_output.transpose(1, 2))
        # b_crgb_rgbdata = torch.bmm(abs_light.unsqueeze(-1), grad_output.transpose(1, 2), out=ctx.interp_datal[0])
        # b_crgb_rgbdata = torch.bmm(grad_output, abs_light.unsqueeze(1)).transpose(1, 2)
        # [batch, n_intrs-1, 3] * [batch, 3, 1] => [batch, n_intrs-1]
        b_crgb_alight = torch.bmm(ctx.rgb_data, grad_output).squeeze()

        # Sigmoid on b_crgb_rgb_data (NOTE: aten has sigmoid_backward)
        b_crgb_rgbdata.mul_((1 - ctx.rgb_data) * ctx.rgb_data)

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
            out=ctx.interp_datal[-1].squeeze()
        )  # [batch, n_intrs - 1]

        # 2. alpha -> sigma_data
        b_sigma_data.sub_(b_weights_clight).mul_(1 - ctx.alpha).mul_(ctx.dists).neg_()
        b_sigma_data[ctx.sigma_data_mask] = 0.
        b_sigma_data.unsqueeze_(2)  # [batch, n_intrs - 1, 1]
        if torch.cuda.is_available():
            print("rendering %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # 3. Spherical harmonics (from b_crgb_rgbdata: [batch, n_intrs-1, 3])
        # sh_data = tc_harmonics.sh_bwd_apply_singleinput(b_crgb_rgbdata, dirs=ctx.rays_d, deg=ctx.sh_deg)
        # sh_data.append(b_sigma_data)
        # sh_data = torch.cat(sh_data, dim=-1)
        tc_harmonics.sh_bwd_apply_listinput(ctx.interp_datal[:-1], dirs=ctx.rays_d, deg=ctx.sh_deg)
        if torch.cuda.is_available():
            print("sh %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # 4. Interpolation
        # neighbor_ids = ctx.neighbor_ids.reshape(-1, 1)  # [batch*(n_intrs-1)*8, 1]
        # weights = ctx.weights.view(batch*n_intrs*8, 1)
        weights = ctx.weights.view(batch, n_intrs, 8, 1)
        neighbor_data = ctx.neighbor_data.view(batch, n_intrs, 8, n_ch)  # This is an empty buffer
        torch.mul(ctx.interp_data.unsqueeze(2), weights, out=neighbor_data)
        if torch.cuda.is_available():
            print("interp %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        # 5. Neighborhood data -> grid data
        grid_data_grad = torch.zeros(*ctx.grid_data_size, dtype=dt, device=dev)
        neighbor_data = neighbor_data.view(-1, neighbor_data.shape[-1])  # [n_pts, n_ch]
        neighbor_ids = ctx.neighbor_ids.view(-1, 1).expand(-1, neighbor_data.shape[-1])  # [n_pts, n_ch]
        grid_data_grad.scatter_add_(0, neighbor_ids, neighbor_data)

        # [n, n_ch]
        if torch.cuda.is_available():
            print("before fail %.2fGB" % (torch.cuda.memory_allocated() / 2**30))

        return grid_data_grad, None, None, None, None, None

    @staticmethod
    def test_autograd():
        batch = 8
        grid_data = torch.randn(30, 4).to(dtype=torch.float64).requires_grad_()
        neighbor_ids = torch.randint(0, 29, (batch, 3, 8)).long()
        offsets = torch.randn(batch, 3, 3).to(dtype=torch.float64)
        rays_d = torch.randn(batch, 3).to(dtype=torch.float64)
        intersections = torch.randn(batch, 3).to(dtype=torch.float64)
        deg = 0

        torch.autograd.gradcheck(lambda d: ComputeIntersection.apply(d, neighbor_ids, offsets, rays_d, intersections, deg),
                                 inputs=grid_data)


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
        dt, dev = rays_d.dtype, rays_d.device
        # Normalize to -1, 1 range (this can probably be done without computing minmax) TODO: This is wrong, and it's unclear what HG wants
        intrs_pts = (intrs_pts - intrs_pts.min()) / (radius) - 1

    interp_data = hg(intrs_pts.view(-1, 3))  # [batch * n_intrs - 1, n_ch]
    interp_data = interp_data.view(batch, nintrs, -1)

    # Split the channels in density (sigma) and RGB. Here we ignore any extra channels which
    # may be present in interp_data
    sh_dim = ((harmonic_degree + 1) ** 2) * 3
    interp_rgb_data = interp_data[..., :sh_dim]
    interp_density_data = interp_data[..., sh_dim]

    # Deal with density data
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    alpha = 1 - torch.exp(-torch.relu(interp_density_data) * dists)  # alpha: [batch, n_intrs-1]
    cum_light = torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)  # [batch, n_intrs - 2]
    cum_light = torch.cat(
        (torch.ones(batch, 1, dtype=dt, device=dev), cum_light), dim=-1)  # [batch, n_intrs - 1]
    # the absolute amount of light that gets stuck in each voxel
    # This quantity can be used to threshold the intersections which must be processed (only if
    # abs_light > threshold). Often the variable is called 'weights'
    abs_light = alpha * cum_light  # [batch, n_intersections - 1]
    acc_map = abs_light.sum(-1)  # [batch]

    # Deal with RGB data
    sh_mult = sh_encoder(rays_d)
    # sh_mult = torch.empty(batch, (harmonic_degree + 1) ** 2, dtype=dt, device=dev)   # [batch, ch/3]
    # sh_mult = tc_harmonics.eval_sh_bases(harmonic_degree, rays_d, sh_mult)           # [batch, ch/3]
    # sh_mult : [batch, ch/3] => [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
    sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]

    rgb_sh = interp_rgb_data.view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
    rgb = torch.sigmoid(torch.sum(sh_mult * rgb_sh, dim=-1))  # [batch, nintrs, 3]
    rgb_map = torch.sum(rgb * abs_light.unsqueeze(-1), dim=-2)  # [batch, 3]
    print("RGB Map", rgb_map.shape)

    if white_bkgd:
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

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
    dt, dev = interp_data.dtype, interp_data.device

    # Split the channels in density (sigma) and RGB.
    interp_density_data = interp_data[..., -1]
    interp_rgb_data = interp_data[..., :-1]

    # Deal with density data
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    alpha = 1 - torch.exp(-torch.relu(interp_density_data) * dists)  # alpha: [batch, n_intrs-1]
    cum_light = torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)  # [batch, n_intrs - 2]
    cum_light = torch.cat(
        (torch.ones(batch, 1, dtype=dt, device=dev), cum_light), dim=-1)  # [batch, n_intrs - 1]
    # the absolute amount of light that gets stuck in each voxel
    # This quantity can be used to threshold the intersections which must be processed (only if
    # abs_light > threshold). Often the variable is called 'weights'
    abs_light = alpha * cum_light  # [batch, n_intersections - 1]
    acc_map = abs_light.sum(-1)  # [batch]

    # Deal with RGB data
    sh_mult = sh_encoder(rays_d)
    # sh_mult = torch.empty(batch, (harmonic_degree + 1) ** 2, dtype=dt, device=dev)   # [batch, ch/3]
    # sh_mult = tc_harmonics.eval_sh_bases(harmonic_degree, rays_d, sh_mult)           # [batch, ch/3]
    # sh_mult : [batch, ch/3] => [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
    sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]

    rgb_sh = interp_rgb_data.view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
    rgb = torch.sigmoid(torch.sum(sh_mult * rgb_sh, dim=-1))  # [batch, nintrs, 3]
    rgb_map = torch.sum(rgb * abs_light.unsqueeze(-1), dim=-2)  # [batch, 3]

    if white_bkgd:
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map


if __name__ == "__main__":
    batch = 8
    grid_data = torch.randn(32**3, 13).requires_grad_()
    rays_d = torch.randn(batch, 3)
    rays_o = torch.randn(batch, 3)
    radius = 1.3
    resolution = 32
    uniform = 0.5
    harmonic_degree = 1
    white_bkgd = True

    # out = compute_grid(grid_data, rays_d, rays_o, radius, resolution, uniform, harmonic_degree, white_bkgd)
