import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import sh
from interp import (
    tricubic_interpolation, trilinear_interpolation_weight,
    tricubic_interpolation_matrix
)
from utils import grid_lookup, vectorize, scalarize
from fwd_model import volumetric_rendering

eps = 1e-5


def near_zero(vector):
    return jnp.abs(vector) < eps


def safe_floor(vector):
    return jnp.floor(vector + eps)


def safe_ceil(vector):
    return jnp.ceil(vector - eps)




# @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 8))
def intersection_distances(inputs, data_dict, resolution, radius, jitter, uniform, key, sh_dim,
                           interpolation, matrix, powers):
    start, stop = inputs["start"], inputs["stop"]
    # offset, interval: arrays of size [3]
    offset, interval = inputs["offset"], inputs["interval"]
    print("uniform: ", uniform)
    print(f"start: {start.shape} - offset: {offset.shape}")
    # print(start[0], start[1], start[2])
    print(f"grid[0]: {data_dict[0].shape} grid[1][0]: {data_dict[1][0].shape}")
    if uniform == 0:
        # For a single ray, compute all the possible voxel intersections up to the upper bound number, starting when the ray hits the cube
        upper_bound = int(1 + resolution)  # per dimension upper bound on the number of voxel intersections
        intersections0 = jnp.linspace(start=start[0] + offset[0],
                                      stop=start[0] + offset[0] + interval[0] * upper_bound,
                                      num=upper_bound, endpoint=False)
        print(f"intersections0 {intersections0.shape} (ub {upper_bound})")
        intersections1 = jnp.linspace(start=start[1] + offset[1],
                                      stop=start[1] + offset[1] + interval[1] * upper_bound,
                                      num=upper_bound, endpoint=False)
        intersections2 = jnp.linspace(start=start[2] + offset[2],
                                      stop=start[2] + offset[2] + interval[2] * upper_bound,
                                      num=upper_bound, endpoint=False)
        intersections = jnp.concatenate([intersections0, intersections1, intersections2], axis=0)  # 3 * upper_bound? or transpose?
        print(f"concat intersections: {intersections.shape}")
        intersections = jnp.sort(intersections)  # TODO: replace this with just a merge of the three intersection arrays
    else:
        voxel_len = radius * 2.0 / resolution
        realstart = jnp.min(start)
        count = int(resolution * 3 / uniform)
        intersections = jnp.linspace(start=realstart + uniform * voxel_len,
                                     stop=realstart + uniform * voxel_len * (count + 1), num=count,
                                     endpoint=False)
        print(f"intersections: {intersections.shape}")  # [n_intersections]
    intersections = jnp.where(intersections <= stop, intersections, stop)
    # Get the values at these intersection points
    ray_o, ray_d = inputs["ray_o"], inputs["ray_d"]
    voxel_sh, voxel_sigma, intersections = values_oneray(intersections, data_dict, ray_o, ray_d,
                                                         resolution, key, radius, jitter,
                                                         1e-5, interpolation, matrix, powers)
    return voxel_sh, voxel_sigma, intersections


# Vectorization over the 'batch_size' dimension (first dimension of all the stuff inside inputs)
get_intersections_partial = jax.vmap(
    fun=intersection_distances, in_axes=(
        {"start": 0, "stop": 0, "offset": 0, "interval": 0, "ray_o": 0, "ray_d": 0},  # inputs
        None,  # data_dict (this is called 'grid' in the caller)
        None,  # resolution
        None,  # radius
        None,  # jitter
        None,  # uniform
        0,     # key
        None,  # sh_dim
        None,  # interpolation
        None,  # matrix
        None   # powers
    ), out_axes=0)
get_intersections = jax.vmap(fun=get_intersections_partial, in_axes=(
    {"start": 1, "stop": 1, "offset": 1, "interval": 1, "ray_o": 1, "ray_d": 1}, None, None, None, None,
    None, 1, None, None, None, None), out_axes=1)


@functools.partial(jax.jit, static_argnums=(3, 4))
def voxel_ids_oneray(intersections, ray_o, ray_d, voxel_len, resolution, eps=1e-5):
    # For a single ray, compute the ids of all the voxels it passes through
    # Compute the midpoint of the ray segment inside each voxel
    midpoints = (intersections[Ellipsis, 1:] + intersections[Ellipsis, :-1]) / 2.0
    midpoints = ray_o[jnp.newaxis, :] + midpoints[:, jnp.newaxis] * ray_d[jnp.newaxis, :]
    ids = jnp.array(jnp.floor(midpoints / voxel_len + eps) + resolution / 2, dtype=int)
    return ids


voxel_ids_partial = jax.jit(
    jax.vmap(fun=voxel_ids_oneray, in_axes=(0, 0, 0, None, None), out_axes=0))
voxel_ids = jax.jit(jax.vmap(fun=voxel_ids_partial, in_axes=(1, 1, 1, None, None), out_axes=1))


# Remove voxels that are empty, where empty is determined by weight (contribution to training pixels) or sigma (opacity)
def prune_grid(grid, method, threshold, train_c2w, H, W, focal, batch_size, resolution, key, radius,
               harmonic_degree, jitter, uniform, interpolation):
    # method can be 'weight' or 'sigma'
    # sigma: prune by opacity
    # weight: prune by contribution to the training rays
    indices, data = grid
    if method == 'sigma':
        keep_idx = jnp.argwhere(data[-1] >= threshold)  # [N_keep, 1]
    elif method == 'weight':
        print(f'rendering all the training views to accumulate weight')
        max_contribution = np.zeros((resolution, resolution, resolution))
        for c2w in tqdm(train_c2w):
            rays_o, rays_d = get_rays(H, W, focal, c2w)
            rays_o = np.reshape(rays_o, [-1, 3])
            rays_d = np.reshape(rays_d, [-1, 3])
            for i in range(int(np.ceil(H * W / batch_size))):
                start = i * batch_size
                stop = min(H * W, (i + 1) * batch_size)
                if jitter > 0:
                    _, _, _, weightsi, voxel_idsi = jax.lax.stop_gradient(
                        render_rays(grid, (rays_o[start:stop], rays_d[start:stop]), resolution,
                                    key[start:stop], radius, harmonic_degree, jitter, uniform,
                                    interpolation))
                else:
                    _, _, _, weightsi, voxel_idsi = jax.lax.stop_gradient(
                        render_rays(grid, (rays_o[start:stop], rays_d[start:stop]), resolution, key,
                                    radius, harmonic_degree, jitter, uniform, interpolation))
                weightsi = np.asarray(weightsi)
                voxel_idsi = np.asarray(voxel_idsi[:, :-1, :])
                max_contribution[
                    voxel_idsi[..., 0], voxel_idsi[..., 1], voxel_idsi[..., 2]] = np.maximum(
                    max_contribution[voxel_idsi[..., 0], voxel_idsi[..., 1], voxel_idsi[..., 2]],
                    weightsi)
        keep_idx = jnp.argwhere(max_contribution >= threshold)  # [N_keep, 3]
        keep_idx = indices[keep_idx[:, 0], keep_idx[:, 1], keep_idx[:, 2]]  # [N_keep, 1]
        del max_contribution, weightsi, voxel_idsi
    keep_idx = jnp.squeeze(keep_idx)  # Indexes into the data
    # Also keep any neighbors of any voxels that are kept
    keep_idx = jax.vmap(lambda idx: data_index_to_scalar(idx, grid))(
        keep_idx)  # Map index into data to scalar spatial index
    keep_idx = jax.vmap(lambda idx: get_neighbors(idx, resolution))(
        keep_idx).flatten()  # Get neighbors of these spatial indices
    jnpindices = jnp.array(indices)
    keep_idx = jax.vmap(lambda idx: scalar_to_data_index(idx, jnpindices))(
        keep_idx)  # Map scalar spatial index to index into data
    # Filter the data
    keep_idx = jnp.unique(keep_idx)  # dedup and sort
    data = [d[keep_idx] for d in data]
    sort_idx = jnp.argsort(indices[indices >= 0])
    idx = jnp.argwhere(indices >= 0)[sort_idx][keep_idx]  # [N_keep, 3]
    indices = jnp.ones((resolution, resolution, resolution), dtype=int) * -1
    indices = indices.at[idx[:, 0], idx[:, 1], idx[:, 2]].set(jnp.arange(len(keep_idx), dtype=int))
    print(f'after pruning, the number of nonempty indices is {len(jnp.argwhere(indices >= 0))}')
    del idx, keep_idx, jnpindices
    return (indices, data)


# Map a position in the data array to the corresponding scalar spatial index
def data_index_to_scalar(idx, grid):
    indices, data = grid
    active_voxels = jnp.argwhere(indices >= 0)  # [N_active_voxels, 3]
    assert len(data[-1]) == len(active_voxels)
    resolution = len(indices)
    return scalarize(active_voxels[idx, 0], active_voxels[idx, 1], active_voxels[idx, 2],
                     resolution)


# Map a scalar index idx to the corresponding position in the data array, or -1 if pruned
def scalar_to_data_index(idx, indices):
    resolution = len(indices)
    vector_idx = vectorize(idx, resolution)
    print(f'indices has type {type(indices)} and idx has type {type(idx)}')
    return indices[vector_idx[0], vector_idx[1], vector_idx[2]]


# Map an index (scalarized) to itself and its 6 neighbors
def get_neighbors(idx, resolution):
    volid = vectorize(idx, resolution)
    front = scalarize(jnp.minimum(resolution - 1, volid[0] + 1), volid[1], volid[2], resolution)
    back = scalarize(jnp.maximum(0, volid[0] - 1), volid[1], volid[2], resolution)
    top = scalarize(volid[0], jnp.minimum(resolution - 1, volid[1] + 1), volid[2], resolution)
    bottom = scalarize(volid[0], jnp.maximum(0, volid[1] - 1), volid[2], resolution)
    right = scalarize(volid[0], volid[1], jnp.minimum(resolution - 1, volid[2] + 1), resolution)
    left = scalarize(volid[0], volid[1], jnp.maximum(0, volid[2] - 1), resolution)
    return jnp.array([idx, front, back, top, bottom, right, left])


def initialize_grid(resolution, ini_rgb=0.0, ini_sigma=0.1, harmonic_degree=0):
    """
    :param resolution:
    :param ini_rgb: Initial value in the spherical harmonics
    :param ini_sigma: Initial value for density sigma
    :param harmonic_degree:
    :return:
        Tuple containing the indices of each voxel in the grid, and the data contained in each voxel.
        The data contains the RGB values of the spherical harmonics, and the value for density sigma.
    """
    sh_dim = (harmonic_degree + 1) ** 2
    data = []  # data is a list of length sh_dim + 1
    for _ in range(sh_dim):
        data.append(jnp.full((resolution ** 3, 3), ini_rgb, dtype=np.float32))
    data.append(jnp.full((resolution ** 3), ini_sigma, dtype=np.float32))
    indices = jnp.arange(resolution ** 3, dtype=int).reshape((resolution, resolution, resolution))
    return (indices, data)


def save_grid(grid, dirname):
    indices, data = grid
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(os.path.join(dirname, f'sigma_grid.npy'), data[-1])
    for i in range(len(data) - 1):
        np.save(os.path.join(dirname, f'sh_grid{i}.npy'), data[i])
    np.save(os.path.join(dirname, f'indices.npy'), indices)


def load_grid(dirname, sh_dim):
    data = []
    for i in range(sh_dim):
        data.append(np.load(os.path.join(dirname, f'sh_grid{i}.npy')))
    data.append(np.load(os.path.join(dirname, f'sigma_grid.npy')))
    indices = np.load(os.path.join(dirname, f'indices.npy'))
    return (indices, data)


@functools.partial(jax.jit, static_argnames=("resolution", "radius", "jitter", "interpolation", "eps", "jitter"))
def values_oneray(intersections,  # [?]
                  grid,           # [?]
                  ray_o,          # [3]
                  ray_d,          # [3]
                  resolution,
                  key,
                  radius,
                  jitter,
                  eps,
                  interpolation,
                  matrix,
                  powers):
    voxel_len = radius * 2.0 / resolution
    if not jitter:
        pts = ray_o[jnp.newaxis, :] + intersections[:, jnp.newaxis] * ray_d[jnp.newaxis, :]  # [n_intersections, 3]
        pts = pts[:, jnp.newaxis, :]  # [n_intersections, 1, 3]
        offsets = jnp.array(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1],
             [1, 1, -1], [1, 1, 1]]) * voxel_len / 2.0  # [8, 3]
        neighbors = jnp.clip(pts + offsets[jnp.newaxis, :, :], a_min=-radius, a_max=radius)  # [n_intersections, 8, 3]
        neighbor_centers = jnp.clip((jnp.floor(neighbors / voxel_len + eps) + 0.5) * voxel_len,
                                    a_min=-(radius - voxel_len / 2),
                                    a_max=radius - voxel_len / 2)  # [n_intersections, 8, 3]
        neighbor_ids = jnp.array(jnp.floor(neighbor_centers / voxel_len + eps) + resolution / 2,
                                 dtype=int)  # [n_intersections, 8, 3]
        neighbor_ids = jnp.clip(neighbor_ids, a_min=0, a_max=resolution - 1)
        xyzs = (pts[:, 0, :] - neighbor_centers[:, 0, :]) / voxel_len
        if interpolation == 'tricubic':
            pt_data = tricubic_interpolation(xyzs, neighbor_ids[:, 0, :], grid, matrix, powers)
            pt_sigma = pt_data[-1][:-1]
            pt_sh = [d[:-1, :] for d in pt_data[:-1]]
        elif interpolation == 'trilinear':
            weights = trilinear_interpolation_weight(xyzs)  # [n_intersections, 8]
            neighbor_data = grid_lookup(
                neighbor_ids[..., 0], neighbor_ids[..., 1], neighbor_ids[..., 2], grid)
            neighbor_sh = neighbor_data[:-1]  # list [n_intersections, 8, 3]
            neighbor_sigma = neighbor_data[-1]
            print("neighbor_sh[0]: ", neighbor_sh[0].shape)
            pt_sigma = jnp.sum(weights * neighbor_sigma, axis=1)[:-1]
            pt_sh = [jnp.sum(weights[..., jnp.newaxis] * nsh, axis=1)[:-1, :] for nsh in neighbor_sh]
        elif interpolation == 'constant':
            voxel_ids = neighbor_ids[:, 0, :]
            voxel_data = jax.vmap(
                lambda voxel_id: grid_lookup(voxel_id[0], voxel_id[1], voxel_id[2], grid))(
                voxel_ids)
            pt_sigma = voxel_data[-1][:-1]
            pt_sh = [d[:-1, :] for d in voxel_data[:-1]]
        else:
            print(f'Unrecognized interpolation method {interpolation}.')
            assert False
        return pt_sh, pt_sigma, intersections
    else:  # Only does trilinear with jitter
        jitters = jax.random.normal(key=key, shape=(intersections.shape[0],)) * voxel_len * jitter
        jittered_intersections = jnp.clip(intersections + jitters, a_min=intersections[0],
                                          a_max=intersections[-1])
        jittered_pts = ray_o[jnp.newaxis, :] + jittered_intersections[:, jnp.newaxis] * ray_d[
                                                                                        jnp.newaxis,
                                                                                        :]  # [n_intersections, 3]
        jittered_pts = jittered_pts[:, jnp.newaxis, :]  # [n_intersections, 1, 3]
        offsets = jnp.array(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1],
             [1, 1, -1], [1, 1, 1]]) * voxel_len / 2.0  # [8, 3]
        neighbors = jnp.clip(jittered_pts + offsets[jnp.newaxis, :, :], a_min=-radius,
                             a_max=radius)  # [n_intersections, 8, 3]
        neighbor_centers = jnp.clip((jnp.floor(neighbors / voxel_len + eps) + 0.5) * voxel_len,
                                    a_min=-(radius - voxel_len / 2),
                                    a_max=radius - voxel_len / 2)  # [n_intersections, 8, 3]
        neighbor_ids = jnp.array(jnp.floor(neighbor_centers / voxel_len + eps) + resolution / 2,
                                 dtype=int)  # [n_intersections, 8, 3]
        neighbor_ids = jnp.clip(neighbor_ids, a_min=0, a_max=resolution - 1)
        xyzs = (jittered_pts[:, 0, :] - neighbor_centers[:, 0, :]) / voxel_len
        weights = trilinear_interpolation_weight(xyzs)  # [n_intersections, 8]
        neighbor_data = grid_lookup(neighbor_ids[..., 0], neighbor_ids[..., 1],
                                    neighbor_ids[..., 2], grid)
        neighbor_sh = neighbor_data[:-1]
        neighbor_sigma = neighbor_data[-1]
        pt_sigma = jnp.sum(weights * neighbor_sigma, axis=1)[:-1]
        pt_sh = [jnp.sum(weights[..., jnp.newaxis] * nsh, axis=1)[:-1, :] for nsh in neighbor_sh]
        idx = jnp.argsort(jittered_intersections)  # Should be nearly sorted already
        return [sh[idx][:-1] for sh in pt_sh], pt_sigma[idx][:-1], jittered_intersections[idx]


# @functools.partial(jax.jit, static_argnums=(2, 4, 5, 6, 7, 8, 9))
def render_rays(grid, rays, resolution, keys, radius=1.3, harmonic_degree=0, jitter=0, uniform=0,
                interpolation='trilinear'):
    sh_dim = (harmonic_degree + 1) ** 2
    voxel_len = radius * 2.0 / resolution
    assert resolution % 2 == 0  # Renderer assumes resolution is a multiple of 2
    # rays_o, rays_d: batch * 3
    rays_o, rays_d = rays
    # Compute when the rays enter and leave the grid
    offsets_pos = jax.lax.stop_gradient((radius - rays_o) / rays_d)
    offsets_neg = jax.lax.stop_gradient((-radius - rays_o) / rays_d)
    offsets_in = jax.lax.stop_gradient(jnp.minimum(offsets_pos, offsets_neg))
    offsets_out = jax.lax.stop_gradient(jnp.maximum(offsets_pos, offsets_neg))
    start = jax.lax.stop_gradient(jnp.max(offsets_in, axis=-1, keepdims=True))
    stop = jax.lax.stop_gradient(jnp.min(offsets_out, axis=-1, keepdims=True))
    first_intersection = jax.lax.stop_gradient(rays_o + start * rays_d)
    # Compute locations of ray-voxel intersections along each dimension
    interval = jax.lax.stop_gradient(voxel_len / jnp.abs(rays_d))
    # TODO: giacomo - why always divide by rays_d?
    offset_bigger = jax.lax.stop_gradient(
        (safe_ceil(first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d)
    offset_smaller = jax.lax.stop_gradient(
        (safe_floor(first_intersection / voxel_len) * voxel_len - first_intersection) / rays_d)
    offset = jax.lax.stop_gradient(jnp.maximum(offset_bigger, offset_smaller))
    print(f"offset: {offset.shape} - interval: {interval.shape} - start: {start.shape} - stop: {stop.shape}")
    # print(f"start: {start} - stop: {stop}")
    # Compute the samples along each ray
    matrix = None
    powers = None
    if interpolation == 'tricubic':
        matrix, powers = tricubic_interpolation_matrix()
    if len(rays_o.shape) > 2:  # TODO: giacomo - When would this happen? rays_o should be batch_size * 3?
        voxel_sh, voxel_sigma, intersections = get_intersections(
            {"start": start, "stop": stop, "offset": offset, "interval": interval, "ray_o": rays_o,
             "ray_d": rays_d}, grid, resolution, radius, jitter, uniform, keys, sh_dim,
            interpolation, matrix, powers)
    else:
        voxel_sh, voxel_sigma, intersections = get_intersections_partial(
            {"start": start, "stop": stop, "offset": offset, "interval": interval, "ray_o": rays_o,
             "ray_d": rays_d}, grid, resolution, radius, jitter, uniform, keys, sh_dim,
            interpolation, matrix, powers)
    # Apply spherical harmonics
    print("voxel_sh", voxel_sh[0].shape, "rays_d", rays_d.shape)
    voxel_rgb = sh.eval_sh(harmonic_degree, voxel_sh, rays_d)
    # Call volumetric_rendering
    rgb, disp, acc, weights = volumetric_rendering(voxel_rgb, voxel_sigma, intersections, rays_d)
    pts = rays_o[:, jnp.newaxis, :] + intersections[:, :, jnp.newaxis] * rays_d[:, jnp.newaxis, :]  # [n_rays, n_intersections, 3]
    ids = jnp.clip(jnp.array(jnp.floor(pts / voxel_len + eps) + resolution / 2, dtype=int), a_min=0,
                   a_max=resolution - 1)
    return rgb, disp, acc, weights, ids


def get_rays(H, W, focal, c2w):
    i, j = jnp.meshgrid(jnp.linspace(0, W - 1, W) + 0.5, jnp.linspace(0, H - 1, H) + 0.5)
    dirs = jnp.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -jnp.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jnp.sum(dirs[..., jnp.newaxis, :] * c2w[:3, :3],
                     -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = jnp.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d
