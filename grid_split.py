import jax
import jax.numpy as jnp

from utils import grid_lookup, scalarize, vectorize


def map_neighbors(offset):
    # offset is a ternary 3-vector; return its index into the offsets array (of length 27, in expand_data)
    return (offset[0] + 1) * 9 + (offset[1] + 1) * 3 + offset[2] + 1


# Split each nonempty voxel in each dimension, using trilinear interpolation to initialize child voxels
def split_weights(childx, childy, childz):
    # childx, childy, childz are each -1 or 1 denoting the position of the child voxel within the parent (-1 instead of 0 for convenience)
    # all 27 neighbors of the parent are considered in the weights, but only 8 of the weights are nonzero for each child
    weights = jnp.zeros(27)
    # center of parent voxel is distance 1/4 from center of each child (nearest neighbor in all dimensions).
    weights = weights.at[13].set(0.75 * 0.75 * 0.75)
    # neighbors that are one away have 2 zeros and one nonzero. There should be 3 of these.
    weights = weights.at[map_neighbors([childx, 0, 0])].set(0.75 * 0.75 * 0.25)
    weights = weights.at[map_neighbors([0, childy, 0])].set(0.75 * 0.75 * 0.25)
    weights = weights.at[map_neighbors([0, 0, childz])].set(0.75 * 0.75 * 0.25)
    # neighbors that are 2 away have 1 zero and two nonzeros. There should be 3 of these.
    weights = weights.at[map_neighbors([childx, childy, 0])].set(0.75 * 0.25 * 0.25)
    weights = weights.at[map_neighbors([childx, 0, childz])].set(0.75 * 0.25 * 0.25)
    weights = weights.at[map_neighbors([0, childy, childz])].set(0.75 * 0.25 * 0.25)
    # one neighbor is 3 away and has all 3 nonzeros.
    weights = weights.at[map_neighbors([childx, childy, childz])].set(0.25 * 0.25 * 0.25)
    return weights


def expand_data(idx, grid):
    # idx is a vector index of the voxel to be split
    offsets = jnp.array(
        [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1],
         [-1, 1, 0], [-1, 1, 1],
         [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1],
         [0, 1, 0], [0, 1, 1],
         [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1],
         [1, 1, 0], [1, 1, 1]])  # [27, 3]
    neighbor_idx = idx[jnp.newaxis, :] + offsets  # [27, 3]
    neighbor_data = grid_lookup(neighbor_idx[:, 0], neighbor_idx[:, 1], neighbor_idx[:, 2], grid)
    child_idx = jnp.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1],
         [1, 1, 1]])  # [8, 3]

    # weights: [8, 27] first index is over the 8 child voxels, second index is over the neighbors
    # for the parent, only 8 of which are relevant to each child
    weights = jax.vmap(split_weights)(child_idx[:, 0], child_idx[:, 1], child_idx[:, 2])
    expanded_data = [jnp.sum(weights[..., jnp.newaxis] * d, axis=1) for d in neighbor_data[:-1]]
    expanded_data.append(jnp.sum(weights * neighbor_data[-1], axis=1))
    del weights, offsets, neighbor_idx, neighbor_data, child_idx
    return expanded_data


# Map an index in a grid to a set of 8 child indices in the split grid
def expand_index(idx, new_resolution):
    i000 = scalarize(idx[0] * 2, idx[1] * 2, idx[2] * 2, new_resolution)
    i001 = scalarize(idx[0] * 2, idx[1] * 2, idx[2] * 2 + 1, new_resolution)
    i010 = scalarize(idx[0] * 2, idx[1] * 2 + 1, idx[2] * 2, new_resolution)
    i011 = scalarize(idx[0] * 2, idx[1] * 2 + 1, idx[2] * 2 + 1, new_resolution)
    i100 = scalarize(idx[0] * 2 + 1, idx[1] * 2, idx[2] * 2, new_resolution)
    i101 = scalarize(idx[0] * 2 + 1, idx[1] * 2, idx[2] * 2 + 1, new_resolution)
    i110 = scalarize(idx[0] * 2 + 1, idx[1] * 2 + 1, idx[2] * 2, new_resolution)
    i111 = scalarize(idx[0] * 2 + 1, idx[1] * 2 + 1, idx[2] * 2 + 1, new_resolution)
    return jnp.array([i000, i001, i010, i011, i100, i101, i110, i111])


# Subdivide each voxel into 8 voxels, using trilinear interpolation and respecting sparsity
def split_grid(grid):
    indices, data = grid
    # Expand the indices, respecting sparsity
    new_resolution = len(indices) * 2
    big_indices = jnp.ones((new_resolution, new_resolution, new_resolution), dtype=int) * -1
    keep_idx = jnp.argwhere(indices >= 0)  # [N_keep, 3]
    # Expand the data, with trilinear interpolation
    big_data_partial = jax.vmap(expand_data, in_axes=(0, None))(keep_idx, grid)
    big_data = [d.reshape(len(data[-1]) * 8, 3) for d in big_data_partial[:-1]]
    big_data.append(big_data_partial[-1].reshape(len(data[-1]) * 8))
    del data
    big_keep_idx = jnp.ravel(
        jax.vmap(lambda index: expand_index(index, new_resolution), in_axes=0)(keep_idx))
    idx = vectorize(big_keep_idx, new_resolution)  # [3, N_keep*8]
    big_indices = big_indices.at[idx[0, :], idx[1, :], idx[2, :]].set(
        jnp.arange(len(big_keep_idx), dtype=int))
    del idx, big_keep_idx, keep_idx, indices
    print(
        f'after splitting, the number of nonempty indices is {len(jnp.argwhere(big_indices >= 0))}')
    return (big_indices, big_data)
