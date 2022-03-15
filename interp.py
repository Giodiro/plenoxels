import numpy as np
import jax
import jax.numpy as jnp

from utils import grid_lookup


@jax.jit
def trilinear_interpolation_weight(xyzs):
    # xyzs should have shape [n_pts, 3] and denote the offset (as a fraction of voxel_len) from the 000 interpolation point
    xs = xyzs[:, 0]
    ys = xyzs[:, 1]
    zs = xyzs[:, 2]
    weight000 = (1 - xs) * (1 - ys) * (1 - zs)  # [n_pts]
    weight001 = (1 - xs) * (1 - ys) * zs  # [n_pts]
    weight010 = (1 - xs) * ys * (1 - zs)  # [n_pts]
    weight011 = (1 - xs) * ys * zs  # [n_pts]
    weight100 = xs * (1 - ys) * (1 - zs)  # [n_pts]
    weight101 = xs * (1 - ys) * zs  # [n_pts]
    weight110 = xs * ys * (1 - zs)  # [n_pts]
    weight111 = xs * ys * zs  # [n_pts]
    weights = jnp.stack(
        [weight000, weight001, weight010, weight011, weight100, weight101, weight110, weight111],
        axis=-1)  # [n_pts, 8]
    return weights


def apply_power(power, xyzs):
    return xyzs[:, 0] ** power[0] * xyzs[:, 1] ** power[1] * xyzs[:, 2] ** power[2]


@jax.jit
def tricubic_interpolation(xyzs, corner_pts, grid, matrix, powers):
    # xyzs should have shape [n_pts, 3] and denote the offset (as a fraction of voxel_len) from the 000 interpolation point
    # corner_pts should have shape [n_pts, 3] and denote the grid coordinates of the 000 interpolation point
    # matrix should be [64, 64] output of tricubic_interpolation_matrix
    # powers should be [64, 3] and contain all combinations of the powers 0 through 3 in three dimensions
    neighbor_data = jax.vmap(lambda pts: tricubic_neighbors(pts, grid))(
        corner_pts)  # list where each entry has shape [n_pts, 64, ...]
    coeffs = [jnp.clip(jax.vmap(lambda d: jnp.matmul(matrix, d))(d), a_min=-1e7, a_max=1e7) for d in
              neighbor_data]  # list where each entry has shape [n_pts, 64, ...]
    things_to_multiply_by_coeffs = jnp.clip(
        jax.vmap(lambda power: apply_power(power, xyzs), out_axes=-1)(powers), a_min=-1e7,
        a_max=1e7)  # [n_pts, 64]
    result = [jnp.sum(coeff * things_to_multiply_by_coeffs[..., jnp.newaxis], axis=1) for coeff in
              coeffs[:-1]]  # list where each entry has shape [n_pts, ...]
    result.append(jnp.sum(coeffs[-1] * things_to_multiply_by_coeffs, axis=1))
    return result


@jax.jit
# Get the data at the 64 neighboring voxels needed for tricubic interpolation
def tricubic_neighbors(idx, grid):
    # idx is a vector index of the voxel to be interpolated
    offsets = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                offsets.append([i - 1, j - 1, k - 1])
    offsets = jnp.array(offsets)  # [64, 3]
    neighbor_idx = idx[jnp.newaxis, :] + offsets  # [64, 3]
    resolution = len(grid[0])
    neighbor_idx = jnp.clip(neighbor_idx, a_min=0, a_max=resolution - 1)
    neighbor_data = jax.vmap(
        lambda neighbor: grid_lookup(neighbor[0], neighbor[1], neighbor[2], grid))(neighbor_idx)
    return neighbor_data


# Generate the 64 by 64 weight matrix that maps grid values to polynomial coefficients
@jax.jit
def tricubic_interpolation_matrix():
    # Set up the indices
    powers = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                powers.append([i, j, k])
    powers = np.asarray(
        powers)  # [64, 3]  all combinations of the powers 0 through 3 in three dimensions
    coords = powers - 1  # [64, 3]  relative coordinates of neighboring voxels
    # Set up the weight matrix
    matrix = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            x = coords[i, 0]
            y = coords[i, 1]
            z = coords[i, 2]
            matrix[i, j] = x ** powers[j, 0] * y ** powers[j, 1] * z ** powers[j, 2]
    # Invert the weight matrix
    inverted_matrix = np.linalg.inv(matrix)
    return jnp.array(inverted_matrix), powers
