#pragma once

#include <torch/extension.h>

constexpr uint32_t n_threads_linear = 128;


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}


__device__ __inline__ float3 diff_prod(const float3 &a, const float3 &b, const float &c) {
    // (a - b) * c
    return make_float3(
        (a.x - b.x) * c,
        (a.y - b.y) * c,
        (a.z - b.z) * c
    );
}


__device__ __inline__ float prod_diff(const float &a, const float &b, const float &c, const float &d) {
    // a * b - c * d (using kahan sum)
    float cd = __fmul_rn(c, d);  // use intrinsic to avoid compiler optimizing this out.
    float err = fmaf(-c, d, cd);
    float dop = fmaf(a, b, -cd);
    return dop + err;
}


__device__ __inline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}


__device__ __inline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}


__host__ __device__ __inline__ void clamp_coord(float3 & __restrict__ q_out, float lower, float upper) {
    q_out.x = q_out.x < lower ? lower : (upper < q_out.x ? upper : q_out.x);
    q_out.y = q_out.y < lower ? lower : (upper < q_out.y ? upper : q_out.y);
    q_out.z = q_out.z < lower ? lower : (upper < q_out.z ? upper : q_out.z);
}


template <int32_t branching>
__device__ __inline__ void traverse_tree_level(
    float3 & __restrict__ coordinate,
    int32_t* __restrict__ u_out,
    int32_t* __restrict__ v_out,
    int32_t* __restrict__ w_out
) {
    coordinate.x *= branching;
    coordinate.y *= branching;
    coordinate.z *= branching;
    *u_out = floorf(coordinate.x);
    *v_out = floorf(coordinate.y);
    *w_out = floorf(coordinate.z);
    coordinate.x -= *u_out;
    coordinate.y -= *v_out;
    coordinate.z -= *w_out;
}


__device__ __inline__ void interp_quad_3d_newt(
    float * __restrict__ weights,
    const float3 * __restrict__ point,
    const float3 * __restrict__ n
)
{
    /*
    https://stackoverflow.com/questions/808441/inverse-bilinear-interpolation
    by points p1,...,p8, where the points are ordered consistent with
    p1~(0,0,0), p2~(0,0,1), p3~(0,1,0), p4~(1,0,0), p5~(0,1,1),
    p6~(1,0,1), p7~(1,1,0), p8~(1,1,1)
    */

    const int32_t num_iter = 3;
    float3 stw = make_float3(0.49, 0.49, 0.49);
    float3 r, js, jt, jw;
    float inv_det_j, det_other;
    for (int32_t i = 0; i < num_iter; i++) {
        weights[0] = (1 - stw.x) * (1 - stw.y) * (1 - stw.z) ;
        weights[1] = (1 - stw.x) * (1 - stw.y) * stw.z       ;
        weights[2] = (1 - stw.x) * stw.y       * (1 - stw.z) ;
        weights[3] = (1 - stw.x) * stw.y       * stw.z       ;
        weights[4] = stw.x       * (1 - stw.y) * (1 - stw.z) ;
        weights[5] = stw.x       * (1 - stw.y) * stw.z       ;
        weights[6] = stw.x       * stw.y       * (1 - stw.z) ;
        weights[7] = stw.x       * stw.y       * stw.z       ;
        for (i = 0; i < 8; i++) {
            r.x += n[i].x * weights[i];
            r.y += n[i].y * weights[i];
            r.z += n[i].z * weights[i];
        }
        js = diff_prod(n[4], n[0], (1 - stw.y) * (1 - stw.z)) +
             diff_prod(n[5], n[1], (1 - stw.y) * stw.z      ) +
             diff_prod(n[6], n[2], stw.y       * (1 - stw.z)) +
             diff_prod(n[7], n[3], stw.y       * stw.z      ) - *point;
        jt = diff_prod(n[2], n[0], (1 - stw.x) * (1 - stw.z)) +
             diff_prod(n[3], n[1], (1 - stw.x) * stw.z      ) +
             diff_prod(n[6], n[4], stw.x       * (1 - stw.z)) +
             diff_prod(n[7], n[5], stw.x       * stw.z      ) - *point;
        jw = diff_prod(n[1], n[0], (1 - stw.x) * (1 - stw.y)) +
             diff_prod(n[3], n[2], (1 - stw.x) * stw.y      ) +
             diff_prod(n[5], n[4], stw.x       * (1 - stw.y)) +
             diff_prod(n[7], n[6], stw.x       * stw.y      ) - *point;
        inv_det_j = 1 / (
             js.x * prod_diff(jt.y, jw.z, jw.y, jt.z) -
             js.y * prod_diff(jt.x, jw.z, jt.z, jw.x) +
             js.z * prod_diff(jt.x, jw.y, jt.y, jw.x));

        // To solve for r we need 3 other determinants
        det_other =
             r.x  * prod_diff(jt.y, jw.z, jw.y, jt.z) -
             js.y * prod_diff(r.y, jw.z, jt.z, r.z) +
             js.z * prod_diff(r.y, jw.y, jt.y, r.z);
        stw.x = fmaf(inv_det_j, -det_other, stw.x);  // stw.x - det_other / det_j
        det_other =
             js.x * prod_diff(r.y, jw.z, r.z, jt.z) -
             r.x  * prod_diff(jt.x, jw.z, jt.z, jw.x) +
             js.z * prod_diff(jt.x, r.z, r.y, jw.x);
        stw.y = fmaf(inv_det_j, -det_other, stw.y);
        det_other =
             js.x * prod_diff(jt.y, r.z, jw.y, r.y) -
             js.y * prod_diff(jt.x, r.z, r.y, jw.x) +
             r.x * prod_diff(jt.x, jw.y, jt.y, jw.x);
        stw.z = fmaf(inv_det_j, -det_other, stw.z);
    }
    weights[0] = (1 - stw.x) * (1 - stw.y) * (1 - stw.z) ;
    weights[1] = (1 - stw.x) * (1 - stw.y) * stw.z       ;
    weights[2] = (1 - stw.x) * stw.y       * (1 - stw.z) ;
    weights[3] = (1 - stw.x) * stw.y       * stw.z       ;
    weights[4] = stw.x       * (1 - stw.y) * (1 - stw.z) ;
    weights[5] = stw.x       * (1 - stw.y) * stw.z       ;
    weights[6] = stw.x       * stw.y       * (1 - stw.z) ;
    weights[7] = stw.x       * stw.y       * stw.z       ;
}



// Device kernels
template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void _dev_query_sum(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    float3& __restrict__ coordinate,
    scalar_t* __restrict__ out_val
)
{
    clamp_coord(coordinate, 0.0, 1.0 - 1e-9);
    int32_t node_id = 0;
    int32_t u, v, w, skip, i;

    for (i = 0; i < data_dim; i++) {
        out_val[i] = data[0][i];
    }
    while (true) {
        traverse_tree_level<branching>(coordinate, &u, &v, &w);
        skip = child[node_id][u][v][w];
        for (i = 0; i < data_dim; i++) {
            out_val[i] += data[node_id + skip][i];
        }
        if (is_child_leaf[node_id][u][v][w]) {
            return;
        }
        node_id += skip;
    }
}


template <typename scalar_t, int32_t branching>
__device__ __inline__ scalar_t* _dev_query_single(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    float3 & __restrict__ coordinate
)
{
    clamp_coord(coordinate, 0.0, 1.0 - 1e-9);

    int32_t node_id = 0;
    int32_t u, v, w, skip;
    while (true) {
        traverse_tree_level<branching>(coordinate, &u, &v, &w);
        skip = child[node_id][u][v][w];
        if (is_child_leaf[node_id][u][v][w]) {
            return &data[node_id + skip][0];
        }
        node_id += skip;
    }
}


__constant__
static const float3 OFFSET[8] = {make_float3(-1, -1, -1), make_float3(-1, -1, 0), make_float3(-1, 0, -1),
                                 make_float3(-1, 0, 0), make_float3(0, -1, -1), make_float3(0, -1, 0),
                                 make_float3(0, 0, -1), make_float3(0, 0, 0)};
__constant__
static const float3 OFFSET2[8] = {make_float3(-0.5, -0.5, -0.5), make_float3(-0.5, -0.5, 0.5), make_float3(-0.5, 0.5, -0.5),
                                  make_float3(-0.5, 0.5, 0.5), make_float3(0.5, -0.5, -0.5), make_float3(0.5, -0.5, 0.5),
                                  make_float3(0.5, 0.5, -0.5), make_float3(0.5, 0.5, 0.5)};


template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void _dev_query_interp(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    float3& __restrict__ coordinate,
    float* __restrict__ weights,
    scalar_t* __restrict__ out_val,
    const bool parent_sum
)
{
//    constexpr float3 offset[8] = {make_float3(-1, -1, -1), make_float3(-1, -1, 0), make_float3(-1, 0, -1),
//                                  make_float3(-1, 0, 0), make_float3(0, -1, -1), make_float3(0, -1, 0),
//                                  make_float3(0, 0, -1), make_float3(0, 0, 0)};
//    constexpr float3 offset2[8] = {make_float3(-0.5, -0.5, -0.5), make_float3(-0.5, -0.5, 0.5), make_float3(-0.5, 0.5, -0.5),
//                                  make_float3(-0.5, 0.5, 0.5), make_float3(0.5, -0.5, -0.5), make_float3(0.5, -0.5, 0.5),
//                                  make_float3(0.5, 0.5, -0.5), make_float3(0.5, 0.5, 0.5)};
    clamp_coord(coordinate, 0.0, 1.0 - 1e-9);
    int32_t node_id = 0;
    int32_t u, v, w, skip, i, j;
    int32_t uc, vc, wc;
    int32_t cube_sz = branching;

    const float3 in_coo = make_float3(coordinate.x, coordinate.y, coordinate.z);
    float3 tmp_coo;
    float3 neigh_coo[8];
    int64_t valid_neighbors[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    if (parent_sum) {
        for (i = 0; i < data_dim; i++) {
            out_val[i] = data[0][i];
        }
    }
    while (true) {
        traverse_tree_level<branching>(&coordinate, &u, &v, &w);
        tmp_coo = make_float3(coordinate.x, coordinate.y, coordinate.z);
        traverse_tree_level<2>(&tmp_coo, &uc, &vc, &wc);

        // Identify valid neighbors
        for(i = 0; i < 8; i++) {
            if (u + uc + OFFSET[i].x >= 0 && u + uc + OFFSET[i].x < branching &&
                v + vc + OFFSET[i].y >= 0 && v + vc + OFFSET[i].y < branching &&
                w + wc + OFFSET[i].z >= 0 && w + wc + OFFSET[i].z < branching)
            {
                skip = child[node_id][u + uc + OFFSET[i].x][v + vc + OFFSET[i].y][w + wc + OFFSET[i].z];
                // Keep track of neighbor coordinates as well as neighbor indices. Coordinates cannot be computed
                // at the end due to dependency on current cube size.
                neigh_coo[i] = make_float3(
                    (floorf(in_coo.x * cube_sz + OFFSET2[i].x + 1e-5) + 0.5) / cube_sz,
                    (floorf(in_coo.y * cube_sz + OFFSET2[i].y + 1e-5) + 0.5) / cube_sz,
                    (floorf(in_coo.z * cube_sz + OFFSET2[i].z + 1e-5) + 0.5) / cube_sz
                );
                clamp_coord(neigh_coo[i], 1 / (cube_sz * branching), 1 - 1 / (cube_sz * branching));
                // Simpler formula (without clamping)
                // (floor((in_coordinate[0] + OFFSET2[i][0] / (cube_sz * 2)) * cube_sz + 1e-5) + 0.5) / cube_sz,
                valid_neighbors[i] = node_id + skip;
            }
        }

        // Determine whether we have finished, and we must interpolate
        if (is_child_leaf[node_id][u][v][w]) {
            interp_quad_3d_newt(weights, &in_coo, neigh_coo);
            for (j = 0; j < 8; j++) {
                if (valid_neighbors[j] < 0) continue;
                for (i = 0; i < data_dim; i++) {
                    if (parent_sum) {
                        out_val[i] += weights[j] * data[valid_neighbors[j]][i];
                    } else {
                        out_val[i] = weights[j] * data[valid_neighbors[j]][i];
                    }
                }
            }
            return;
        }
        // Not finished yet. Add the current node's value to results
        skip = child[node_id][u][v][w];
        if (parent_sum) {
            for (i = 0; i < data_dim; i++) {
                out_val[i] += data[node_id + skip][i];
            }
        }
        node_id += skip;
        cube_sz *= branching;
    }
}


// Global kernels
template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void octree_query_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> tree_data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out_values,
    const size_t n_elements,
    const bool parent_sum
)
{
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 coord = make_float3(indices[i][0], indices[i][1], indices[i][2]);
    if (parent_sum) {
        _dev_query_sum<scalar_t, branching, data_dim>(
            tree_data, child, is_child_leaf, coord, &out_values[i][0]
        );
    } else {
        scalar_t* data_at_coo = _dev_query_single<scalar_t, branching>(
            tree_data, child, is_child_leaf, coord
        );
        for (int32_t j = 0; j < data_dim; j++) {
            out_values[i][j] = data_at_coo[j];
        }
    }
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void octree_query_interp_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> tree_data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out_values,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> weights,
    const size_t n_elements,
    const bool parent_sum
)
{
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 coord = make_float3(indices[i][0], indices[i][1], indices[i][2]);
    _dev_query_interp<scalar_t, branching, data_dim>(
        tree_data, child, is_child_leaf,
        coord,
        &weights[i][0],
        &out_values[i][0],
        parent_sum
    );
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void octree_set_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> tree_data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> indices,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
    const size_t n_elements
)
{
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 coord = make_float3(indices[i][0], indices[i][1], indices[i][2]);
    scalar_t* data_at_coo = _dev_query_single<scalar_t, branching>(
        tree_data, child, is_child_leaf, coord
    );
    for (int32_t j = 0; j < data_dim; j++) {
        data_at_coo[j] = values[i][j];
    }
}
