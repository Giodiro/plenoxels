#pragma once

#include <tuple>
#include <torch/extension.h>
#include "cuda_common.h"
#include "octree.h"


constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
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
    const torch::TensorAccessor<float, 2, torch::RestrictPtrTraits, int32_t> n
)
{
    /*
    https://stackoverflow.com/questions/808441/inverse-bilinear-interpolation
    by points p1,...,p8, where the points are ordered consistent with
    p1~(0,0,0), p2~(0,0,1), p3~(0,1,0), p4~(1,0,0), p5~(0,1,1),
    p6~(1,0,1), p7~(1,1,0), p8~(1,1,1)
    */
    int32_t num_iter = 4;
    float3 stw = make_float3(0.49, 0.49, 0.49);
    float3 js = make_float3(0, 0, 0), jt = make_float3(0, 0, 0), jw = make_float3(0, 0, 0), r = make_float3(0, 0, 0);
    float inv_det_j, det_other;
    for (;num_iter > 0; num_iter--) {
        weights[0] = (1 - stw.x) * (1 - stw.y) * (1 - stw.z) ;
        weights[1] = (1 - stw.x) * (1 - stw.y) * stw.z       ;
        weights[2] = (1 - stw.x) * stw.y       * (1 - stw.z) ;
        weights[3] = (1 - stw.x) * stw.y       * stw.z       ;
        weights[4] = stw.x       * (1 - stw.y) * (1 - stw.z) ;
        weights[5] = stw.x       * (1 - stw.y) * stw.z       ;
        weights[6] = stw.x       * stw.y       * (1 - stw.z) ;
        weights[7] = stw.x       * stw.y       * stw.z       ;
        r.x = 0; r.y = 0; r.z = 0;
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            r.x += n[i][0] * weights[i];
            r.y += n[i][1] * weights[i];
            r.z += n[i][2] * weights[i];
        }
        r -= *point;

        js.x = 0; js.y = 0; js.z = 0;
        diff_prod(&n[4][0], &n[0][0], (1 - stw.y) * (1 - stw.z), js);
        diff_prod(&n[5][0], &n[1][0], (1 - stw.y) * stw.z,       js);
        diff_prod(&n[6][0], &n[2][0], stw.y       * (1 - stw.z), js);
        diff_prod(&n[7][0], &n[3][0], stw.y       * stw.z      , js);
        jt.x = 0; jt.y = 0; jt.z = 0;
        diff_prod(&n[2][0], &n[0][0], (1 - stw.x) * (1 - stw.z), jt);
        diff_prod(&n[3][0], &n[1][0], (1 - stw.x) * stw.z,       jt);
        diff_prod(&n[6][0], &n[4][0], stw.x       * (1 - stw.z), jt);
        diff_prod(&n[7][0], &n[5][0], stw.x       * stw.z,       jt);
        jw.x = 0; jw.y = 0; jw.z = 0;
        diff_prod(&n[1][0], &n[0][0], (1 - stw.x) * (1 - stw.y), jw);
        diff_prod(&n[3][0], &n[2][0], (1 - stw.x) * stw.y,       jw);
        diff_prod(&n[5][0], &n[4][0], stw.x       * (1 - stw.y), jw);
        diff_prod(&n[7][0], &n[6][0], stw.x       * stw.y,       jw);
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
        if (num_iter > 1)
            clamp_coord(stw, 1e-4, 1-1e-4);
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
template <typename scalar_t, int32_t branching>
__device__ __inline__ void _dev_query_ninfo(
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
    float3& __restrict__ coordinate,
    float* __restrict__ cube_sz_out,
    int64_t* __restrict__ node_id_out)
{
    clamp_coord(coordinate, 0.0, 1.0 - 1e-9);

    int32_t u, v, w;
    *cube_sz_out = branching;
    *node_id_out = 0;
    while (true) {
        traverse_tree_level<branching>(coordinate, &u, &v, &w);
        bool is_child = is_child_leaf[*node_id_out][u][v][w];
        *node_id_out += child[*node_id_out][u][v][w];
        if (is_child) { return; }
        *cube_sz_out *= branching; // TODO: Check if multiplication to be performed before or after returning
    }
}
template <typename scalar_t, int32_t branching>
__device__ __inline__ void _dev_query_ninfo(
    const Acc32<int32_t, 4> child,
    float3& __restrict__ coordinate,
    float* __restrict__ cube_sz_out,
    int64_t* __restrict__ node_id_out)
{
    clamp_coord(coordinate, 0.0, 1.0 - 1e-9);

    int32_t u, v, w, skip;
    *cube_sz_out = branching;
    *node_id_out = 0;
    while (true) {
        traverse_tree_level<branching>(coordinate, &u, &v, &w);
        skip = child[*node_id_out][u][v][w];
        if (skip < 0) { return; }
        *node_id_out += skip;
        *cube_sz_out *= branching; // TODO: Check if multiplication to be performed before or after returning
    }
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void _dev_query_sum(
    Acc64<scalar_t, 2> data,
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
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


template<typename scalar_t, int branching>
__device__ __inline__ void _dev_query_corners(
    const Acc32<int, 4>         child,
    const Acc32<int, 5>         nids,
          float3 & __restrict__ coordinate,
          float  * const __restrict__ weights,
          int    * const __restrict__ nid_ptr
)
{
    int u, v, w;
    int node_id = 0, skip;
    const int * const nid_start_ptr = &nids[0][0][0][0][0];
    clamp_coord(coordinate, 0.0, 1.0 - 1e-6);
    while (true) {
        traverse_tree_level<branching>(coordinate, &u, &v, &w);
        skip = child[node_id][u][v][w];
        if (skip < 0) {
            weights[0] = (1 - coordinate.x) * (1 - coordinate.y) * (1 - coordinate.z);
            weights[1] = (1 - coordinate.x) * (1 - coordinate.y) * coordinate.z;
            weights[2] = (1 - coordinate.x) * coordinate.y       * (1 - coordinate.z);
            weights[3] = (1 - coordinate.x) * coordinate.y       * coordinate.z;
            weights[4] = coordinate.x       * (1 - coordinate.y) * (1 - coordinate.z);
            weights[5] = coordinate.x       * (1 - coordinate.y) * coordinate.z;
            weights[6] = coordinate.x       * coordinate.y       * (1 - coordinate.z);
            weights[7] = coordinate.x       * coordinate.y       * coordinate.z;
            *nid_ptr = (int)(&nids[node_id][u][v][w][0] - nid_start_ptr);
            return;
        }
        node_id += skip;
    }
}

template<typename scalar_t, int data_dim>
__device__ __inline__ void _dev_query_corners_bwd(
    const int      * __restrict__ nid_ptr,
    const float    * __restrict__ neighbor_w,
    const scalar_t * __restrict__ grad_val,
          Acc32<scalar_t, 2>      grad_output
)
{
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < data_dim; j++) {
            atomicAdd(&grad_output[nid_ptr[i]][j], grad_val[j] * neighbor_w[i]);
        }
    }
}



template <typename scalar_t, int32_t branching>
__device__ __inline__ scalar_t* _dev_query_single(
    Acc64<scalar_t, 2> data,
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
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

template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void _dev_query_single_outv(
    Acc64<scalar_t, 2> data,
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
    float3 & __restrict__ in_coo,
    const float * __restrict__ weights,  // unused (for interface)
    const torch::TensorAccessor<float, 2, torch::RestrictPtrTraits, int32_t> neighbor_coo,    // unused (for interface)
    int64_t     * __restrict__ neighbor_ids,  // only first element gets set
    scalar_t    * __restrict__ out_val,
    const bool parent_sum
)
{
    clamp_coord(in_coo, 0.0, 1.0 - 1e-9);

    int32_t node_id = 0;
    int32_t u, v, w, skip;
    int32_t i;

    if (parent_sum) {
        for (i = 0; i < data_dim; i++) { out_val[i] = data[0][i]; }
    } else {
        for (i = 0; i < data_dim; i++) { out_val[i] = 0; }
    }
    while (true) {
        traverse_tree_level<branching>(in_coo, &u, &v, &w);
        skip = child[node_id][u][v][w];
        if (parent_sum || is_child_leaf[node_id][u][v][w]) {
            for (i = 0; i < data_dim; i++) {
                out_val[i] += data[node_id + skip][i];
            }
        }
        if (is_child_leaf[node_id][u][v][w]) {
            neighbor_ids[0] = node_id + skip;
            return;
        }
        node_id += skip;
    }
} // _dev_query_single_outv

template <typename scalar_t, int32_t data_dim>
__device__ __inline__ void _dev_query_single_outv_bwd(
    const Acc32<int32_t, 1> parent,       // tree description
    Acc64<scalar_t, 2> grad,
    const float* __restrict__ weights,       // unused
    const int64_t* __restrict__ neighbor_ids,   // Only first element used.
    const scalar_t* __restrict__ grad_output)   // [data_dim].
{
    int32_t j;
    int64_t node_id = neighbor_ids[0];
    for (j = 0; j < data_dim; ++j) {
        atomicAdd(&grad[node_id][j], grad_output[j]);
    }
//    while (node_id >= 0) {
//        node_id = parent[node_id];
//    }
}


__constant__
static const float OFFSET[8][3] = {{-1, -1, -1}, {-1, -1, 0}, {-1, 0, -1}, {-1, 0, 0},
                                   {0, -1, -1}, {0, -1, 0}, {0, 0, -1}, {0, 0, 0}};
__constant__
static const float OFFSET2[8][3] = {{-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5},
                                    {0.5, -0.5, -0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, -0.5}, {0.5, 0.5, 0.5}};


template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void _dev_query_interp(
    Acc64<scalar_t, 2> data,               // tree description
    const Acc32<int32_t, 4> child,         // tree description
    const Acc32<bool, 4> is_child_leaf,    // tree description
    const float3& __restrict__ in_coo,     // query coordinate
    float    * __restrict__ weights,       // [8]. output parameter. Interpolation weights
    torch::TensorAccessor<float, 2, torch::RestrictPtrTraits, int32_t> neighbor_coo,    // [8,3]. output parameter (temp buffer). Coordinates of the neighbors
    int64_t  * __restrict__ neighbor_ids,  // [8]. output parameter. Node IDs of the interpolation neighbors. Assumed to all be -1 on input.
    scalar_t * __restrict__ out_val,       // [data_dim]. output parameter. Interpolated value
    const bool parent_sum                                                                    // tree description
)
{
    float3 coo = make_float3(in_coo.x, in_coo.y, in_coo.z);
    clamp_coord(coo, 0.0, 1.0 - 1e-9);
    int32_t node_id = 0;
    int32_t u, v, w, i, j;
    int32_t uc, vc, wc;
    int32_t cube_sz = branching;

    if (parent_sum) {
        for (i = 0; i < data_dim; i++) {
            out_val[i] = data[0][i];
        }
    }
    neighbor_coo[0][0] =  0;
    neighbor_coo[0][1] =  0;
    neighbor_coo[0][2] =  0;
    neighbor_coo[1][0] =  0;
    neighbor_coo[1][1] =  0;
    neighbor_coo[1][2] = +1;
    neighbor_coo[2][0] =  0;
    neighbor_coo[2][1] = +1;
    neighbor_coo[2][2] =  0;
    neighbor_coo[3][0] =  0;
    neighbor_coo[3][1] = +1;
    neighbor_coo[3][2] = +1;
    neighbor_coo[4][0] = +1;
    neighbor_coo[4][1] =  0;
    neighbor_coo[4][2] =  0;
    neighbor_coo[5][0] = +1;
    neighbor_coo[5][1] =  0;
    neighbor_coo[5][2] = +1;
    neighbor_coo[6][0] = +1;
    neighbor_coo[6][1] = +1;
    neighbor_coo[6][2] =  0;
    neighbor_coo[7][0] = +1;
    neighbor_coo[7][1] = +1;
    neighbor_coo[7][2] = +1;
    while (true) {
        traverse_tree_level<branching>(coo, &u, &v, &w);
        uc = floorf(coo.x * 2);
        vc = floorf(coo.y * 2);
        wc = floorf(coo.z * 2);

        // Identify valid neighbors
        for(i = 0; i < 8; i++) {
            if (u + uc + OFFSET[i][0] >= 0 && u + uc + OFFSET[i][0] < branching &&
                v + vc + OFFSET[i][1] >= 0 && v + vc + OFFSET[i][1] < branching &&
                w + wc + OFFSET[i][2] >= 0 && w + wc + OFFSET[i][2] < branching)
            {
                neighbor_ids[i] = node_id + child[node_id][u + uc + OFFSET[i][0]][v + vc + OFFSET[i][1]][w + wc + OFFSET[i][2]];
                // Keep track of neighbor coordinates as well as neighbor indices. Coordinates cannot be computed
                // at the end due to dependency on current cube size.
                // Simpler formula (without clamping)
                // (floor((in_coordinate[0] + OFFSET2[i][0] / (cube_sz * 2)) * cube_sz + 1e-5) + 0.5) / cube_sz,
                neighbor_coo[i][0] = (floorf(in_coo.x * cube_sz + OFFSET2[i][0] + 1e-5) + 0.5) / cube_sz;
                neighbor_coo[i][1] = (floorf(in_coo.y * cube_sz + OFFSET2[i][1] + 1e-5) + 0.5) / cube_sz;
                neighbor_coo[i][2] = (floorf(in_coo.z * cube_sz + OFFSET2[i][2] + 1e-5) + 0.5) / cube_sz;
                clamp_coord(&neighbor_coo[i][0], 1 / (cube_sz * branching), 1 - 1 / (cube_sz * branching));

                #ifdef DEBUG
                    printf("Set valid neighbor %d: %ld, coordinate %f %f %f \n",
                        i, neighbor_ids[i], neighbor_coo[i][0], neighbor_coo[i][1], neighbor_coo[i][2]);
                #endif
            }
        }

        // Determine whether we have finished, and we must interpolate
        if (is_child_leaf[node_id][u][v][w]) {
            interp_quad_3d_newt(weights, &in_coo, neighbor_coo);
            #ifdef DEBUG
                printf("Weights: %f %f %f %f %f %f %f %f\n", weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], weights[7]);
            #endif
            for (j = 0; j < 8; j++) {
                if (neighbor_ids[j] < 0) continue;
                for (i = 0; i < data_dim; i++) {
                    if (parent_sum) {
                        out_val[i] += weights[j] * data[neighbor_ids[j]][i];
                    } else {
                        out_val[i] = weights[j] * data[neighbor_ids[j]][i];
                    }
                }
            }
            return;
        }
        // Not finished yet. Add the current node's value to results
        node_id += child[node_id][u][v][w];
        if (parent_sum) {
            for (i = 0; i < data_dim; i++) {
                out_val[i] += data[node_id][i];
            }
        }
        cube_sz *= branching;
    }
}


__device__ __inline__ int32_t argmax_8arr(const int64_t *__restrict__ arr) {
    int32_t o = 0;
    if (arr[1] > arr[o]) o = 1;
    if (arr[2] > arr[o]) o = 2;
    if (arr[3] > arr[o]) o = 3;
    if (arr[4] > arr[o]) o = 4;
    if (arr[5] > arr[o]) o = 5;
    if (arr[6] > arr[o]) o = 6;
    if (arr[7] > arr[o]) o = 7;
    return o;
}

/*
 *  Backward pass for Query Interpolate.
 */
template <typename scalar_t, int32_t data_dim>
__device__ __inline__ void _dev_query_interp_bwd(
    const Acc32<int32_t, 1> parent,       // tree description
    Acc64<scalar_t, 2> grad,
    const float* __restrict__ weights,       // [8]. Interpolation weights
    const int64_t* __restrict__ neighbor_ids,   // [8]. Neighbor IDs
    const scalar_t* __restrict__ grad_output)   // [data_dim].
{
    int32_t i, j;
    int64_t node_id;
    int32_t max_nid = argmax_8arr(neighbor_ids);

    for (i = 0; i < 8; ++i) {
        node_id = neighbor_ids[i];
        if (i == max_nid) {
            bool is_parent = false;
            while (node_id >= 0) {
                for (j = 0; j < data_dim; ++j) {
                    (is_parent) ?
                        atomicAdd(&grad[node_id][j], grad_output[j]) :
                        atomicAdd(&grad[node_id][j], grad_output[j] * weights[i]);
                }
                node_id = parent[node_id];
                is_parent = true;
            }
        } else if (node_id >= 0) {
            for (j = 0; j < data_dim; ++j) {
                atomicAdd(&grad[node_id][j], grad_output[j] * weights[i]);
            }
        }
    }
}



// Global kernels
template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void octree_query_kernel(
    Acc64<scalar_t, 2> tree_data,
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
    const Acc32<float, 2> indices,
    Acc32<scalar_t, 2> out_values,
    const int64_t n_elements,
    const bool parent_sum
)
{
	const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
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
torch::Tensor octree_query (
    OctreeCppSpec<scalar_t> & tree,
    const torch::Tensor & indices)
{
    int64_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        torch::Tensor undefined;
        return undefined;
    }
    // Create output tensor
    torch::Tensor values_out = torch::empty({n_elements, data_dim}, tree.data.options().requires_grad(false));
    octree_query_kernel<scalar_t, branching, data_dim><<<n_blocks_linear<uint32_t>(n_elements, 128), 128>>>(
        tree.data_acc(), tree.child_acc(), tree.is_child_leaf_acc(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements,
        tree.parent_sum
    );
    return values_out;
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void octree_query_interp_kernel(
    Acc64<scalar_t, 2> tree_data,
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
    const Acc32<float, 2> indices,             // n_elements, 3
    Acc32<scalar_t, 2> out_values,       // n_elements, data_dim
    Acc32<float, 2> weights,             // n_elements, 8
    Acc32<float, 3> neighbor_coo,        // n_elements, 8, 3
    Acc32<int64_t, 2> neighbor_ids,      // n_elements, 8
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
        neighbor_coo[i],
        &neighbor_ids[i][0],
        &out_values[i][0],
        parent_sum
    );
}

template <typename scalar_t, int32_t branching, int32_t data_dim>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> octree_query_interp(
    OctreeCppSpec<scalar_t> & tree,
    const torch::Tensor     & indices)
{
    int64_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        torch::Tensor undefined;
        return std::make_tuple(undefined, undefined, undefined);
    }
    // Temporary tensors
    torch::Tensor neighbor_coo = torch::empty({n_elements, 8, 3}, torch::dtype(torch::kFloat32).device(tree.data.device()));
    torch::Tensor neighbor_ids = torch::full({n_elements, 8}, -1, torch::dtype(torch::kInt64).device(tree.data.device()));
    // Create output tensors
    torch::Tensor values_out = torch::empty({n_elements, data_dim}, tree.data.options().requires_grad(false));
    torch::Tensor weights_out = torch::empty({n_elements, 8}, indices.options());
    // Call CUDA kernel
    octree_query_interp_kernel<scalar_t, branching, data_dim><<<n_blocks_linear<uint32_t>(n_elements, 128), 128>>>(
        tree.data_acc(), tree.child_acc(), tree.is_child_leaf_acc(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        weights_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        neighbor_coo.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        neighbor_ids.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        n_elements,
        tree.parent_sum
    );
    return std::make_tuple(values_out, weights_out, neighbor_ids);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void octree_set_kernel(
    Acc64<scalar_t, 2> tree_data,
    const Acc32<int32_t, 4> child,
    const Acc32<bool, 4> is_child_leaf,
    const Acc32<float, 2> indices,
    const Acc32<scalar_t, 2> values,
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


template <class scalar_t, int32_t branching, int32_t data_dim>
void octree_set(
    OctreeCppSpec<scalar_t> & tree,
    const torch::Tensor &indices,
    const torch::Tensor &vals,
    const bool update_avg)
{
    int64_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        return;
    }
    octree_set_kernel<scalar_t, branching, data_dim><<<n_blocks_linear<uint32_t>(n_elements, 128), 128>>>(
        tree.data_acc(), tree.child_acc(), tree.is_child_leaf_acc(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        vals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements
    );

    // Set all parents to be their child's average (bottom to top)
    // Remove the average from the children
    if (update_avg) {
        torch::Tensor max_depth_t = tree.depth.max();
        int32_t max_depth = max_depth_t.item<int32_t>();
        for (int i = max_depth; i > 0; i--) {
            auto child_ids = (tree.depth == torch::tensor({i}, tree.depth.options())).nonzero().squeeze();
            auto parent_ids = tree.parent.index({child_ids}).to(torch::kInt64);
            tree.data.index_put_({parent_ids}, torch::tensor({0}, tree.data.options().requires_grad(false)));
            tree.data.scatter_add_(
                0, parent_ids.unsqueeze(-1).expand({parent_ids.size(0), data_dim}), tree.data.index({child_ids}));
            tree.data.index({parent_ids}).div_(branching * branching * branching);
            if (tree.parent_sum) {
                tree.data.scatter_add_(
                    0, child_ids.unsqueeze(-1).expand({child_ids.size(0), data_dim}), -tree.data.index({parent_ids}));
            }
        }
    }
}
