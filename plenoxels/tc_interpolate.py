import torch


def map_neighbors(offset_x, offset_y, offset_z):
    # offset is a ternary 3-vector; return its index into the offsets array (of length 27, in expand_data)
    return (offset_x + 1) * 9 + (offset_y + 1) * 3 + offset_z + 1


# Split each nonempty voxel in each dimension, using trilinear interpolation to initialize child voxels
def split_weights(childx, childy, childz):
    # childx, childy, childz are each -1 or 1 denoting the position of the child voxel within the parent (-1 instead of 0 for convenience)
    # all 27 neighbors of the parent are considered in the weights, but only 8 of the weights are nonzero for each child
    weights = torch.zeros(8, 27)
    # center of parent voxel is distance 1/4 from center of each child (nearest neighbor in all dimensions).
    weights[:, 13] = 0.75 * 0.75 * 0.75
    # neighbors that are one away have 2 zeros and one nonzero. There should be 3 of these.
    weights[:, map_neighbors(childx, 0, 0)] = 0.75 * 0.75 * 0.25
    weights[:, map_neighbors(0, childy, 0)] = 0.75 * 0.75 * 0.25
    weights[:, map_neighbors(0, 0, childz)] = 0.75 * 0.75 * 0.25
    # neighbors that are 2 away have 1 zero and two nonzeros. There should be 3 of these.
    weights[:, map_neighbors(childx, childy, 0)] = 0.75 * 0.25 * 0.25
    weights[:, map_neighbors(childx, 0, childz)] = 0.75 * 0.25 * 0.25
    weights[:, map_neighbors(0, childy, childz)] = 0.75 * 0.25 * 0.25
    # one neighbor is 3 away and has all 3 nonzeros.
    weights[:, map_neighbors(childx, childy, childz)] = 0.25 * 0.25 * 0.25
    return weights


def trilinear_upsampling_weights():
    child_idx = torch.tensor(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
         [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])  # [8, 3]
    return split_weights(child_idx[:, 0], child_idx[:, 1], child_idx[:, 2])  # [8, 27]


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
        # neighbor_data: [batch, channels, 8] or [batch, 8, channels]
        # offsets:       [batch, 3]
        # out:           [batch, channels]
        offsets = offsets.to(dtype=neighbor_data.dtype)  # [batch, 3]
        nbr_axis = 1 if neighbor_data.shape[1] == 8 else 2  # TODO: This won't work if channels==8
        other_axis = 2 if nbr_axis == 1 else 1

        weights = get_interp_weights(xs=offsets[:, 0], ys=offsets[:, 1], zs=offsets[:, 2])
        weights = weights.unsqueeze(other_axis)  # [batch, 1, 8] or [batch, 8, 1]

        out = neighbor_data.mul_(weights).sum(nbr_axis)  # [batch, ch]

        ctx.weights = weights
        ctx.nbr_axis = nbr_axis
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # weights:      [batch, 1, 8]
        # grad_output:  [batch, n_channels]
        # out:          [batch, n_channels, 8]
        return grad_output.unsqueeze(ctx.nbr_axis) * ctx.weights, None, None

    @staticmethod
    def test_autograd():
        data = torch.randn(5, 6, 8).to(dtype=torch.float64).requires_grad_()
        weights = torch.randn(5, 3).to(dtype=torch.float64)

        torch.autograd.gradcheck(lambda d: TrilinearInterpolate.apply(d, weights),
                                 inputs=data)
