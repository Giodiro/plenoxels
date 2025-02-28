import random
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Union

import torch
import torch.nn.functional as F

import tc_harmonics
from tc_interpolate import trilinear_upsampling_weights


@dataclass
class Grid:
    indices: Optional[torch.Tensor]
    grid: Union[torch.Tensor, Tuple[torch.Tensor]]


def initialize_grid(resolution: torch.Tensor,
                    ini_rgb: float = 0.0,
                    ini_sigma: float = 0.1,
                    harmonic_degree: int = 0,
                    device=None,
                    dtype=torch.float32,
                    init_indices=True,
                    separate_grids=False,) -> Grid:
    sh_dim = (harmonic_degree + 1) ** 2
    total_data_channels = sh_dim * 3 + 1

    if separate_grids:
        data_rgb = torch.full((torch.prod(resolution).item(), total_data_channels - 1),
                              ini_rgb, dtype=dtype, device=device)
        # torch.nn.init.normal_(data_rgb, std=0.1)
        #torch.nn.init.uniform_(data_rgb, -1e-4, 1e-4)
        data_sigma = torch.full((torch.prod(resolution).item(), 1),
                                ini_sigma, dtype=dtype, device=device)
        #torch.nn.init.uniform_(data_sigma, 1e-4, 1e-1)
        # torch.nn.init.normal_(data_sigma, std=0.1)
        data = (data_rgb, data_sigma)
    else:
        data = torch.full((torch.prod(resolution).item(), total_data_channels), ini_rgb, dtype=dtype, device=device)
        data[:, -1].fill_(ini_sigma)

    if not init_indices:
        return Grid(grid=data, indices=None)

    indices = torch.arange(torch.prod(resolution).item(), dtype=torch.long, device=device).reshape(
        (resolution[0], resolution[1], resolution[2]))
    return Grid(grid=data, indices=indices)


def plenoxel_sh_encoder(harmonic_degree: int) -> Callable[[torch.Tensor], torch.Tensor]:
    num_sh = (harmonic_degree + 1) ** 2

    def encode(rays_d: torch.Tensor) -> torch.Tensor:
        out = torch.empty(rays_d.shape[0], num_sh, dtype=rays_d.dtype, device=rays_d.device)
        out = tc_harmonics.eval_sh_bases(harmonic_degree, rays_d, out)
        return out
    return encode


@torch.jit.script
def get_intersection_ids(intersections: torch.Tensor,  # [batch, n_intersections]
                         rays_o: torch.Tensor,  # [batch, 3]
                         rays_d: torch.Tensor,  # [batch, 3]
                         grid_idx: torch.Tensor,  # [res, res, res]
                         voxel_len: float,
                         aabb: torch.Tensor,  # [2, 3]
                         resolution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        min=aabb[0] + voxel_len / 2,
        max=aabb[1] - voxel_len / 2)  # [batch, n_intersections, 8, 3]

    neighbor_ids = torch.clamp_(
        neighbors_grid_coords.add_(resolution / 2).to(torch.long),
        min=torch.tensor(0, dtype=resolution.dtype, device=resolution.device), max=resolution - 1)  # [batch, n_intersections, 8, 3]

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
                      aabb: torch.Tensor,
                      step_size: float,
                      n_samples: int,
                         near: float,
                      far: float) -> torch.Tensor:
    with torch.autograd.no_grad():
        dev, dt = rays_o.device, rays_o.dtype
        #offsets_pos = (aabb[1].flip(0) - rays_o) / rays_d  # [batch, 3]
        #offsets_neg = (aabb[0].flip(0) - rays_o) / rays_d  # [batch, 3]
        offsets_pos = (aabb[1] - rays_o) / rays_d  # [batch, 3]
        offsets_neg = (aabb[0] - rays_o) / rays_d  # [batch, 3]
        offsets_in = torch.minimum(offsets_pos, offsets_neg)  # [batch, 3]
        # offsets_out = torch.maximum(offsets_pos, offsets_neg)  # [batch, 3]
        start = torch.amax(offsets_in, dim=-1, keepdim=True)  # [batch, 1]
        # stop = torch.amin(offsets_out, dim=-1, keepdim=True)  # [batch, 1]
        start.clamp_(min=near, max=far)  # [batch, 1]

        steps = torch.arange(n_samples, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
        steps = steps.repeat(rays_d.shape[0], 1)   # [batch, n_intrs]
        intersections = start + steps * step_size  # [batch, n_intrs]
        # intersections = torch.clamp_(intersections, min=None, max=stop)
    return intersections


def get_intersections_tensorrf(rays_o: torch.Tensor,
                               rays_d: torch.Tensor,
                               aabb: torch.Tensor,  # same role as radius
                               n_samples: int,
                               step_size: float,    # same as voxel_len
                               near: float,
                               far: float,
                               is_train: bool) -> torch.Tensor:
    with torch.autograd.no_grad():
        dev, dt = rays_o.device, rays_o.dtype
        offsets_pos = (aabb[1] - rays_o) / rays_d
        offsets_neg = (aabb[0] - rays_o) / rays_d
        offsets_in = torch.minimum(offsets_pos, offsets_neg)
        start = torch.amax(offsets_in, dim=-1, keepdim=True)
        start.clamp_(min=near, max=far)  # [batch, 1]

        steps = torch.arange(n_samples, dtype=dt, device=dev).unsqueeze(0)  # [1, n_intrs]
        if is_train:  # Add some randomization
            steps = steps.repeat(rays_d.shape[0], 1)  # [batch, n_intrs]
            steps += torch.rand_like(steps[:, [0]])   # Random is in [0, 1) range of shape [batch, 1]

        intersections = start + step_size * steps  # [batch, n_intrs]
    return intersections


@torch.jit.script
def normalize_coord(intersections: torch.Tensor,
                    aabb: torch.Tensor,
                    inverse_aabb_size: torch.Tensor) -> torch.Tensor:
    """Returns intersection coordinates between -1 and +1"""
    #out = (intersections - aabb[0].flip(0)) * inverse_aabb_size.flip(0) - 1
    out = (intersections - aabb[0]) * inverse_aabb_size - 1
    return out


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
        # neighbor_data: [batch, channels, 8]
        # offsets:       [batch, 3]
        # out:           [batch, channels]
        offsets = offsets.to(dtype=neighbor_data.dtype)  # [batch, 3]
        weights = get_interp_weights(xs=offsets[:, 0], ys=offsets[:, 1], zs=offsets[:, 2]).unsqueeze(1)  # [batch, 1, 8]
        if neighbor_data.dtype == torch.bfloat16:
            out = neighbor_data.mul_(weights).sum(-1)  # [batch, ch]
        else:
            out = torch.einsum('bki, bki -> bk', neighbor_data, weights)  # [batch, ch]

        ctx.weights = weights
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # weights:      [batch, 1, 8]
        # grad_output:  [batch, n_channels]
        # out:          [batch, n_channels, 8]
        return grad_output[..., None] * ctx.weights, None, None

    @staticmethod
    def test_autograd():
        data = torch.randn(5, 6, 8).to(dtype=torch.float64).requires_grad_()
        weights = torch.randn(5, 3).to(dtype=torch.float64)

        torch.autograd.gradcheck(lambda d: TrilinearInterpolate.apply(d, weights),
                                 inputs=data)


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


@torch.jit.script
def sigma2alpha(sigma: torch.Tensor, intersections: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    alpha: torch.Tensor = 1 - torch.exp(-sigma * dists)            # alpha: [batch, n_intrs-1]

    # the absolute amount of light that gets stuck in each voxel
    # This quantity can be used to threshold the intersections which must be processed (only if
    # abs_light > threshold). Often the variable is called 'weights'
    cum_light = torch.cat((torch.ones(sigma.shape[0], 1, dtype=sigma.dtype, device=sigma.device),
                           torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)), dim=-1)  # [batch, n_intrs-1]
    abs_light = alpha * cum_light  # [batch, n_intersections - 1]
    return alpha, abs_light


@torch.jit.script
def shrgb2rgb(sh_rgb: torch.Tensor, abs_light: torch.Tensor, white_bkgd: bool) -> torch.Tensor:
    # Accumulated color over the samples, ignoring background
    rgb = torch.sigmoid(sh_rgb)  # [batch, n_intrs-1, 3]
    #rgb = torch.relu(sh_rgb)
    rgb_map: torch.Tensor = (abs_light.unsqueeze(-1) * rgb).sum(dim=-2)  # [batch, 3]

    if white_bkgd:
        acc_map = abs_light.sum(-1)    # [batch]
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map


@torch.jit.script
def depth_map(abs_light: torch.Tensor, intersections: torch.Tensor) -> torch.Tensor:
    with torch.autograd.no_grad():  # Depth & Inverse Depth-map
        # Weighted average of depths by contribution to final color
        depth: torch.Tensor = (abs_light * intersections[..., :-1]).sum(dim=-1)
        return depth


@torch.jit.script
def volumetric_rendering(rgb: torch.Tensor,            # [batch, n_intersections-1, 3]
                         sigma: torch.Tensor,          # [batch, n_intersections-1, 1]
                         intersections: torch.Tensor,  # [batch, n_intersections]
                         rays_d: torch.Tensor,         # [batch, 3]
                         white_bkgd: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Volumetric rendering
    # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    dists = torch.diff(intersections, n=1, dim=1) \
                 .mul(torch.linalg.norm(rays_d, ord=2, dim=-1, keepdim=True))  # dists: [batch, n_intrs-1]
    alpha: torch.Tensor = 1 - torch.exp(-torch.relu(sigma) * dists)            # alpha: [batch, n_intrs-1]
    # the absolute amount of light that gets stuck in each voxel
    # This quantity can be used to threshold the intersections which must be processed (only if
    # abs_light > threshold). Often the variable is called 'weights'
    cum_light = torch.cat((torch.ones(rgb.shape[0], 1, dtype=rgb.dtype, device=rgb.device),
                           torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1)), dim=-1)  # [batch, n_intrs-1]
    abs_light = alpha * cum_light  # [batch, n_intersections - 1]
    acc_map = abs_light.sum(-1)    # [batch]

    # Accumulated color over the samples, ignoring background
    rgb = torch.sigmoid(rgb)  # [batch, n_intrs-1, 3]
    rgb_map: torch.Tensor = (abs_light.unsqueeze(-1) * rgb).sum(dim=-2)  # [batch, 3]

    with torch.autograd.no_grad():  # Depth & Inverse Depth-map
        # Weighted average of depths by contribution to final color
        depth: torch.Tensor = (abs_light * intersections[..., :-1]).sum(dim=-1)

    if white_bkgd:
        # Including the white background in the final color
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map, alpha, depth


@torch.jit.script
def shifted_softplus(data: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return F.softplus(data + offset)


# noinspection PyAttributeOutsideInit
class AbstractNerF(torch.nn.Module):
    def __init__(self,
                 resolution: torch.Tensor,
                 aabb: torch.Tensor,
                 uniform_rays: float,
                 count_intersections: str,
                 voxel_mul: float):
        super().__init__()

        self.uniform_rays = uniform_rays
        self.count_intersections = count_intersections
        self.voxel_mul = voxel_mul
        self.n_intersections = None

        self.s_res = (resolution[0], resolution[1], resolution[2])
        self.register_buffer("resolution_", resolution)
        self.register_buffer("aabb_", aabb)

        self.register_buffer("aabb_size", None)
        self.register_buffer("inv_aabb_size", None)
        self.register_buffer("voxel_len", None)
        self.update_step_sizes()

    @property
    def resolution(self):
        return self.resolution_

    @resolution.setter
    def resolution(self, val):
        self.resolution_ = val.to(dtype=self.resolution_.dtype, device=self.resolution_.device)
        self.s_res = (val[0], val[1], val[2])
        self.update_step_sizes()

    @property
    def aabb(self) -> torch.Tensor:
        return self.aabb_

    @aabb.setter
    def aabb(self, val):
        self.aabb_ = val.to(dtype=self.aabb_.dtype, device=self.aabb_.device)
        self.update_step_sizes()

    def update_step_sizes(self):
        self.aabb_size = self.aabb_[1] - self.aabb_[0]
        self.inv_aabb_size = 2 / self.aabb_size
        units = self.aabb_size / (self.resolution_ - 1)
        aabb_diag = torch.linalg.norm(self.aabb_size)

        if isinstance(self.count_intersections, int):
            self.n_intersections = self.count_intersections
        elif self.count_intersections == "plenoxels":
            self.n_intersections = int(torch.mean(self.resolution * 3 * self.voxel_mul).item())
        elif self.count_intersections == "tensorrf":
            self.n_intersections = int(aabb_diag / torch.mean(units).item() * self.voxel_mul) + 1
        self.voxel_len = aabb_diag / self.n_intersections
        print(f"Voxel length {self.voxel_len:.3e} - Num intersections {self.n_intersections}")


class IrregularGrid(AbstractNerF):
    def __init__(self, resolution: torch.Tensor, aabb: torch.Tensor, deg: int,
                 ini_sigma: float, ini_rgb: float, sh_encoder,
                 white_bkgd: bool, uniform_rays: float,
                 prune_threshold: float, count_intersections: str,
                 near_far: Tuple[float, float]):
        super().__init__(resolution, aabb, count_intersections=count_intersections,
                         uniform_rays=uniform_rays)

        grid = initialize_grid(self.resolution, harmonic_degree=deg, device=None,
                               dtype=torch.bfloat16, init_indices=True,
                               ini_sigma=ini_sigma, ini_rgb=ini_rgb)
        self.grid_data = torch.nn.Parameter(grid.grid, requires_grad=True)
        self.grid_idx_ = grid.indices
        self.register_buffer("grid_idx", self.grid_idx_)

        self.sh_encoder = sh_encoder
        self.white_bkgd = white_bkgd
        self.prune_threshold = prune_threshold
        self.near = near_far[0]
        self.far = near_far[1]

        split_offsets = torch.tensor(
            [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
             [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
             [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.long)  # [27, 3]
        self.register_buffer("split_offsets", split_offsets)

        prune_offsets = torch.tensor(  # Not sure why but here 6 offsets are used
            [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0],
             [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=torch.long)
        self.register_buffer("prune_offsets", prune_offsets)

        split_weights = trilinear_upsampling_weights()  # [8, 27]
        self.register_buffer("split_weights", split_weights)

    def forward(self, rays_d: torch.Tensor, rays_o: torch.Tensor):
        with torch.autograd.no_grad():
            #intersections = get_intersections_tensorrf(
            #    rays_o=rays_o, rays_d=rays_d, aabb=self.aabb, step_size=self.voxel_len, n_samples=self.n_intersections,
            #    near=self.near, far=self.far, is_train=self.training)
            # noinspection PyTypeChecker
            intersections = get_intersections(
                rays_o=rays_o, rays_d=rays_d, aabb=self.aabb, step_size=self.voxel_len, n_samples=self.n_intersections,
                uniform=self.uniform_rays)  # [batch, n_intrs]
            intersections_trunc = intersections[:, :-1]
            # noinspection PyTypeChecker
            interp_dirs, neighbor_ids = get_intersection_ids(
                intersections_trunc, rays_o, rays_d, self.grid_idx,
                voxel_len=self.voxel_len, aabb=self.aabb, resolution=self.resolution)

        batch, nintrs = interp_dirs.shape[:2]
        n_ch = self.grid_data.shape[-1]

        # Fetch neighbors
        neighbor_data = torch.gather(self.grid_data, dim=0, index=neighbor_ids.view(-1, 1).expand(-1, n_ch))
        neighbor_data = neighbor_data.view(batch, nintrs, 8, n_ch)  # [batch, n_intrs, 8, n_ch]

        # Interpolation
        interp_data = TrilinearInterpolate.apply(neighbor_data, interp_dirs)  # [batch, n_intrs, n_ch]

        # Spherical harmonics
        sh_mult = self.sh_encoder(rays_d)  # [batch, ch/3]
        # sh_mult : [batch, ch/3] => [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
        sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)  # [batch, nintrs, 1, ch/3]
        interp_rgb = interp_data[..., :-1].view(batch, nintrs, 3, sh_mult.shape[-1])  # [batch, nintrs, 3, ch/3]
        rgb_data = torch.sum(sh_mult * interp_rgb, dim=-1)  # [batch, nintrs, 3]

        sigma_data = interp_data[..., -1]  # [batch, n_intrs-1, 1]

        rgb_map, alpha, depth = volumetric_rendering(rgb_data, sigma_data, intersections, rays_d, self.white_bkgd)
        return rgb_map, alpha, depth

    def approx_density_tv_reg(self):
        return torch.mean(torch.relu(self.grid_data[..., -1]))

    def density_l1_reg(self):
        return torch.mean(torch.abs(self.grid_data[..., -1]))

    @torch.autograd.no_grad()
    def split_grid(self):
        # Expand the indices, respecting sparsity
        new_resolution = self.resolution * 2
        new_grid_idx = torch.ones(new_resolution, dtype=self.grid_idx.dtype, device=self.grid_idx.device).mul_(-1)
        # Expand the data, with trilinear interpolation
        valid_idx: torch.Tensor = torch.nonzero(self.grid_idx >= 0)  # [n_keep, 3]
        neighbor_ids = valid_idx.unsqueeze(1) + self.split_offsets.unsqueeze(0)  # [n_keep, 27, 3]
        neighbor_ids = neighbor_ids.clamp_(min=0, max=self.resolution - 1)
        neighbor_idxs = self.grid_idx[
            neighbor_ids[..., 0],
            neighbor_ids[..., 1],
            neighbor_ids[..., 2]
        ]  # [n_keep, 27]

        neighbor_data = torch.gather(
            self.grid_data, dim=0, index=neighbor_idxs.view(-1, 1).expand(1, self.grid_data.shape[-1]))
        neighbor_data = neighbor_data.view(neighbor_idxs.shape[0], 27, -1)  # [n_keep, 27, n_ch]

        # Sum on the 27 axis. [1, 8, 27, 1] * [n_keep, 1, 27, n_ch] => [n_keep, 8, n_ch]
        new_data = torch.einsum('betc,betc->bec',
                                self.split_weights.view(1, 8, 27, 1), neighbor_data.unsqueeze(1))
        new_data = new_data.reshape(-1, new_data.shape[-1])  # [n_keep * 8, n_ch]
        delattr(self, "grid_data")
        del neighbor_data
        self.grid_data = torch.nn.Parameter(new_data, requires_grad=True)

        # 8 new indices for every index
        new_valid_idx = valid_idx.mul_(2).repeat(1, 1, 8)  # [n_keep, 3, 8]
        new_valid_idx[:, 0, 4:].add_(1)
        new_valid_idx[:, 1, [2, 3, 6, 7]].add_(1)
        new_valid_idx[:, 2, 1::2].add_(1)
        new_valid_idx = new_valid_idx.permute(0, 2, 1).view(-1, 3)  # [n_keep*8, 3]
        new_grid_idx[new_valid_idx[:, 0],
                     new_valid_idx[:, 1],
                     new_valid_idx[:, 2]] = torch.arange(new_data.shape[0])
        del self.grid_idx_, new_valid_idx
        delattr(self, "grid_idx")
        self.grid_idx_ = new_grid_idx
        self.register_buffer("grid_idx", self.grid_idx_)
        self.resolution = new_resolution

    @torch.autograd.no_grad()
    def prune_grid(self):
        density = self.grid_data[..., -1]
        keep_mask = density >= self.prune_threshold    # [n_valid]
        # Currently valid
        valid_idx = torch.nonzero(self.grid_idx >= 0)  # [n_valid, 3]
        # New valid after pruning by density
        valid_idx = valid_idx[keep_mask]               # [n_keep, 3]
        # Also consider as valid all neighbors
        valid_idx = valid_idx.unsqueeze(1) + self.prune_offsets.unsqueeze(0)  # [n_valid, 7, 3]
        valid_idx = valid_idx.view(-1, 3)  # [n_valid*7, 3]
        valid_idx = valid_idx.clamp_(min=0, max=self.resolution - 1)
        # Filter the grid according to valid indices. We set invalid indices to -1.
        # The following are IDs into the data (1D IDs)
        valid_grid_ids = self.grid_idx[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]  # [n_valid*7,]
        valid_grid_ids, uniq_idx = torch.unique(valid_grid_ids, sorted=True, return_inverse=True)
        del self.grid_idx_
        delattr(self, "grid_idx")
        # Prune dataset
        pruned_data = torch.gather(
            self.grid_data, dim=0,
            index=valid_grid_ids.view(-1, 1).expand(-1, self.grid_data.shape[-1]))  # [n_valid*7, n_ch]
        del self.grid_data
        delattr(self, "grid_data")
        # Now create new indices
        new_grid_idx = torch.full(
            self.resolution, -1, dtype=torch.long, device=pruned_data.device)
        new_grid_idx[valid_idx[uniq_idx, 0],
                     valid_idx[uniq_idx, 1],
                     valid_idx[uniq_idx, 2]] = torch.arange(pruned_data.shape[0], device=new_grid_idx.device)
        self.grid_idx_ = new_grid_idx
        self.register_buffer("grid_idx", self.grid_idx_)
        self.grid_data = torch.nn.Parameter(pruned_data, requires_grad=True)
        print(f'After pruning have {pruned_data.shape[0]} non-empty indices.')


class HashGrid(AbstractNerF):
    def __init__(self, resolution: torch.Tensor, aabb: torch.Tensor, deg: int,
                 ini_sigma: float, ini_rgb: float, sh_encoder, hg_encoder,
                 white_bkgd: bool, uniform_rays: float,
                 count_intersections: str, harmonic_degree: int):
        super().__init__(resolution, aabb, count_intersections=count_intersections,
                         uniform_rays=uniform_rays)

        grid = initialize_grid(self.resolution, harmonic_degree=deg, device=None,
                               dtype=torch.float32, init_indices=False,
                               ini_sigma=ini_sigma, ini_rgb=ini_rgb)
        self.grid_data = torch.nn.Parameter(grid.grid, requires_grad=True)

        self.sh_encoder = sh_encoder
        self.hg_encoder = hg_encoder
        self.white_bkgd = white_bkgd
        self.harmonic_degree = harmonic_degree

    def forward(self, rays_d: torch.Tensor, rays_o: torch.Tensor):
        with torch.autograd.no_grad():
            intrs = get_intersections(
                rays_o=rays_o, rays_d=rays_d, step_size=self.voxel_len, aabb=self.aabb,
                uniform=self.uniform_rays, n_samples=self.n_intersections)  # [batch, n_intrs]
            intrs_trunc = intrs[:, :-1]  # [batch, n_intrs - 1]
            # Intersections in the real world
            intrs_pts = (rays_o.unsqueeze(1) +
                         intrs_trunc.unsqueeze(2) * rays_d.unsqueeze(1))  # [batch, n_intrs - 1, 3]
            # Normalize to -1, 1 range
            intrs_pts = self.normalize_coord(intrs_pts)

        batch, nintrs = intrs_pts.shape[:2]
        interp_data = self.hg_encoder(intrs_pts.view(-1, 3))  # [batch * n_intrs - 1, n_ch]
        interp_data = interp_data.view(batch, nintrs, -1)

        # Split the channels in density (sigma) and RGB. Here we ignore any extra channels which
        # may be present in interp_data. First channel are lower resolution than the later ones.
        sh_dim = ((self.harmonic_degree + 1) ** 2) * 3
        interp_rgb_data = interp_data[..., :sh_dim]
        sigma_data = interp_data[..., sh_dim]

        # Deal with RGB data.
        # We flip the SH coefficients so that the lower-order are applied to higher resolution data.
        sh_mult = torch.flip(self.sh_encoder(rays_d), (-1, ))  # [batch, ch/3]
        # sh_mult :=> [batch, 1, ch/3] => [batch, n_intrs, ch/3] => [batch, nintrs, 1, ch/3]
        sh_mult = sh_mult.unsqueeze(1).expand(batch, nintrs, -1).unsqueeze(2)
        # rgb_sh : [batch, nintrs, 3, ch/3]
        rgb_sh = interp_rgb_data.view(batch, nintrs, 3, sh_mult.shape[-1])
        rgb_data = torch.sum(sh_mult * rgb_sh, dim=-1)  # [batch, nintrs, 3]

        rgb_map, alpha, depth = volumetric_rendering(
            rgb_data, sigma_data, intrs, rays_d, self.white_bkgd)
        return rgb_map, alpha, depth


class RegularGrid(AbstractNerF):
    def __init__(self, resolution: torch.Tensor, aabb: torch.Tensor, deg: int,
                 ini_sigma: float, ini_rgb: float, sh_encoder,
                 white_bkgd: bool, uniform_rays: float,
                 count_intersections: str, near_far: Tuple[float],
                 abs_light_thresh: float, occupancy_thresh: float,
                 voxel_mul: float):
        super().__init__(resolution, aabb, count_intersections=count_intersections,
                         uniform_rays=uniform_rays, voxel_mul=voxel_mul)

        grid = initialize_grid(self.resolution, harmonic_degree=deg, device=None,
                               dtype=torch.float32, init_indices=False, separate_grids=True,
                               ini_sigma=ini_sigma, ini_rgb=ini_rgb)
        rgb_data = grid.grid[0].T.view(1, -1, *self.s_res)
        self.rgb_data = torch.nn.Parameter(rgb_data, requires_grad=True)
        sigma_data = grid.grid[1].T.view(1, -1, *self.s_res)
        self.sigma_data = torch.nn.Parameter(sigma_data, requires_grad=True)

        self.register_buffer("occupancy", None)

        self.sh_encoder = sh_encoder
        self.white_bkgd = white_bkgd
        self.abs_light_thresh = abs_light_thresh
        self.occupancy_thresh = occupancy_thresh
        self.near_far = near_far
        self.params_changed = False
        self.softplus_offset_hp = 1e-6

    def update_occupancy(self):
        with torch.autograd.no_grad():
            sigma = self.sigma_data  # 1, 1, res, res, res
            #act_sigma = shifted_softplus(sigma, self.shifted_softplus_offset)
            act_sigma = sigma
            #dists = torch.linalg.norm(self.aabb)
            #alpha: torch.Tensor = 1 - torch.exp(-torch.relu(sigma) * dists)
            #alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)
            self.occupancy = act_sigma.detach().clone()#alpha
            print("Updated occupancy. Have %.2f%% entries full" % ((self.occupancy > self.occupancy_thresh).float().mean() * 100))

    def shrink(self, really_shrink=True):
        if self.occupancy is None:
            return
        with torch.autograd.no_grad():
            ooc_mask = self.occupancy > self.occupancy_thresh
            ooc_mask = ooc_mask.squeeze()
            dev = ooc_mask.device
            xyz_min, xyz_max = torch.empty(3, dtype=torch.long, device=dev), torch.empty(3, dtype=torch.long, device=dev)
            i_coord = ooc_mask.amax(0)
            ij_coord = i_coord.amax(0)
            xyz_min[2] = torch.max(ij_coord, dim=0).indices
            xyz_max[2] = len(ij_coord) - torch.max(ij_coord.flip(0), dim=0).indices
            ik_coord = i_coord.amax(1)
            xyz_min[1] = torch.max(ik_coord, dim=0).indices
            xyz_max[1] = len(ik_coord) - torch.max(ik_coord.flip(0), dim=0).indices
            j_coord = ooc_mask.amax(1)
            ji_coord = j_coord.amax(1)
            xyz_min[0] = torch.max(ji_coord, dim=0).indices
            xyz_max[0] = len(ji_coord) - torch.max(ji_coord.flip(0), dim=0).indices

            units = self.aabb_size / self.resolution
            new_aabb = torch.empty_like(self.aabb)

            new_aabb[0] = self.aabb[0] + xyz_min * units
            new_aabb[1] = self.aabb[1] - (self.resolution - xyz_max) * units

            new_reso = xyz_max - xyz_min
            if torch.all(new_reso == self.resolution):  # noqa
                print("No shrinkage possible.")
                return
            if really_shrink:
                # Now actually shrink the data
                self.sigma_data = torch.nn.Parameter(
                    self.sigma_data[:, :, xyz_min[0]: xyz_max[0], xyz_min[1]: xyz_max[1], xyz_min[2]: xyz_max[2]], requires_grad=True)
                self.rgb_data = torch.nn.Parameter(
                    self.rgb_data[:, :, xyz_min[0]: xyz_max[0], xyz_min[1]: xyz_max[1], xyz_min[2]: xyz_max[2]], requires_grad=True)
                if self.occupancy is not None:
                    self.occupancy = self.occupancy[
                         :, :, xyz_min[0]: xyz_max[0], xyz_min[1]: xyz_max[1], xyz_min[2]: xyz_max[2]]
                self.resolution = new_reso
                self.aabb = new_aabb
                self.params_changed = True

    def upscale(self, new_resolution):
        with torch.autograd.no_grad():
            new_res_s = (new_resolution[0].item(), new_resolution[1].item(), new_resolution[2].item())
            print("Upsampling to new resolution %s" % (new_res_s, ))
            new_sigma = F.interpolate(
                self.sigma_data, size=new_res_s, align_corners=True, mode='trilinear')
            delattr(self, "sigma_data")
            self.sigma_data = torch.nn.Parameter(new_sigma, requires_grad=True)

            new_rgb = F.interpolate(
                self.rgb_data, size=new_res_s, align_corners=True, mode='trilinear')
            delattr(self, "rgb_data")
            self.rgb_data = torch.nn.Parameter(new_rgb, requires_grad=True)

            if self.occupancy is not None:
                self.occupancy = F.interpolate(
                    self.occupancy, size=new_res_s, align_corners=True, mode='trilinear')

            self.resolution = new_resolution
            self.params_changed = True

    # noinspection PyMethodMayBeStatic
    def _interp(self, grid, pts):
        # grid [1, ch, res, res, res]
        # pts  [1, n, 1, 1, 3]
        pts = pts.to(dtype=grid.dtype)
        interp_data = F.grid_sample(
            grid, pts, mode='bilinear', align_corners=True)  # [1, ch, n, 1, 1]
        interp_data = interp_data.squeeze()  # [ch, n] or [n] if ch is 1
        return interp_data

    @property
    def shifted_softplus_offset(self):
        return torch.log((1 - self.softplus_offset_hp)**(-1/self.voxel_len) - 1)

    def forward(self, rays_d: torch.Tensor, rays_o: torch.Tensor, calc_sparsity: bool = False):
        with torch.autograd.no_grad():
            intersections = get_intersections(
                rays_o=rays_o, rays_d=rays_d, aabb=self.aabb, step_size=self.voxel_len,
                n_samples=self.n_intersections, near=self.near_far[0],
                far=self.near_far[1])  # [batch, n_intrs]
            intersections_trunc = intersections[:, :-1]  # [batch, n_intrs - 1]
            batch, nintrs = intersections_trunc.shape
            intrs_pts = rays_o.unsqueeze(1) + intersections_trunc.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch, n_intrs - 1, 3]
            # noinspection PyTypeChecker
            intrs_pts_mask = torch.all((self.aabb[0] < intrs_pts) & (intrs_pts < self.aabb[1]), dim=-1)  # [batch, n_intrs-1]
            # Normalize to -1, 1 range
            intrs_pts = normalize_coord(intrs_pts, self.aabb, self.inv_aabb_size).flip(-1)
            if self.occupancy is not None:
                occ_interp = self._interp(self.occupancy, intrs_pts[intrs_pts_mask].view(1, -1, 1, 1, 3))
                # aabb_diag = torch.linalg.norm(self.aabb_size)
                # Threshold as in instant-ngp (app. C)
                # density_threshold = self.n_intersections / aabb_diag * 1e-2
                occ_interp = occ_interp > self.occupancy_thresh
                intrs_pts_mask.masked_scatter_(intrs_pts_mask, occ_interp)

        # 1. Process density: Un-masked sigma (batch, n_intrs-1), and compute.
        sigma_full = torch.zeros(batch, nintrs, dtype=self.sigma_data.dtype, device=self.sigma_data.device)
        sigma_interp = self._interp(
            self.sigma_data, intrs_pts[intrs_pts_mask].view(1, -1, 1, 1, 3))  # [mask_pts]
        sigma_full.masked_scatter_(intrs_pts_mask, sigma_interp)
        # Post-activation (either shifted softplus or relu)
        #sigma_full = shifted_softplus(sigma_full, self.shifted_softplus_offset)
        sigma_full = F.relu(sigma_full)
        #print("sigma_full", sigma_full[0])
        alpha, abs_light = sigma2alpha(sigma_full, intersections, rays_d)  # both [batch, n_intrs-1]

        # 2. Create mask for rgb computations. This is a subset of the intrs_pts_mask.
        rgb_valid_mask = abs_light > self.abs_light_thresh  # [batch, n_intrs-1]

        # 3. Create SH coefficients and mask them
        sh_mult = self.sh_encoder(rays_d).unsqueeze(1).expand(batch, nintrs, -1)  # [batch, nintrs, ch/3]
        sh_mult = sh_mult[rgb_valid_mask].unsqueeze(1)  # [mask_pts, 1, ch/3]

        # 4. Interpolate rgbdata
        rgb_full = torch.zeros(batch, nintrs, 3, dtype=self.rgb_data.dtype, device=self.rgb_data.device)
        rgb_interp = self._interp(
            self.rgb_data, intrs_pts[rgb_valid_mask].view(1, -1, 1, 1, 3)).T  # [mask_pts, ch]
        rgb_interp = rgb_interp.view(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
        rgb_interp = torch.sum(sh_mult * rgb_interp, dim=-1)  # [mask_pts, 3]
        rgb_full.masked_scatter_(rgb_valid_mask.unsqueeze(-1), rgb_interp)
        rgb_full = shrgb2rgb(rgb_full, abs_light, self.white_bkgd)
        depth = depth_map(abs_light, intersections)

        return rgb_full, alpha, depth

    def _tv_reg(self, data, b_start_x, b_len_x, b_start_y, b_len_y, b_start_z, b_len_z):
        block = data.narrow(2, b_start_x, b_len_x).narrow(3, b_start_y, b_len_y).narrow(4, b_start_z, b_len_z)
        tv = (block.diff(dim=2).div(256/self.s_res[0]).square().sum() +
              block.diff(dim=3).div(256/self.s_res[1]).square().sum() +
              block.diff(dim=4).div(256/self.s_res[2]).square().sum()) / \
            (b_len_x * b_len_y * b_len_z * data.shape[1])
        return tv

    def approx_density_tv_reg(self, subsampling: float, sh_weight: float, sigma_weight: float):
        block_res = (self.resolution / (subsampling**(1/3)) + 1).long()
        b_len_x, b_len_y, b_len_z = block_res[0], block_res[1], block_res[2]
        res = self.rgb_data.shape[2:]
        b_start_x = random.randint(0, res[0] - b_len_x - 2)
        b_start_y = random.randint(0, res[1] - b_len_y - 2)
        b_start_z = random.randint(0, res[2] - b_len_z - 2)
        try:
            tv_rgb = sh_weight * self._tv_reg(
                self.rgb_data, b_start_x, b_len_x, b_start_y, b_len_y, b_start_z, b_len_z)
            tv_sigma = sigma_weight * self._tv_reg(
                self.sigma_data, b_start_x, b_len_x, b_start_y, b_len_y, b_start_z, b_len_z)
        except:
            print(self.rgb_data.shape)
            print(self.resolution, self.s_res)
            raise
        return tv_rgb + tv_sigma

    def density_l1_reg(self):
        return torch.mean(torch.abs(self.sigma_data))

    # noinspection PyMethodMayBeStatic
    def sparsity_reg(self, alpha):
        # See eq. 4 in plenoxel paper. Not sure it's correct to use alpha
        return torch.log(1 + 2 * alpha.square()).sum()
