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

