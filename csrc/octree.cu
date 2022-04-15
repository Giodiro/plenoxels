#include <torch/extension.h>

#include "octree.hpp"
#include "octree_common.cuh"


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
    constexpr float3 offset[8] = {make_float3(-1, -1, -1), make_float3(-1, -1, 0), make_float3(-1, 0, -1),
                                  make_float3(-1, 0, 0), make_float3(0, -1, -1), make_float3(0, -1, 0),
                                  make_float3(0, 0, -1), make_float3(0, 0, 0)};
    constexpr float3 offset2[8] = {make_float3(-0.5, -0.5, -0.5), make_float3(-0.5, -0.5, 0.5), make_float3(-0.5, 0.5, -0.5),
                                  make_float3(-0.5, 0.5, 0.5), make_float3(0.5, -0.5, -0.5), make_float3(0.5, -0.5, 0.5),
                                  make_float3(0.5, 0.5, -0.5), make_float3(0.5, 0.5, 0.5)};
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
        traverse_tree_level<2>(tmp_coo, &uc, &vc, &wc);

        // Identify valid neighbors
        for(i = 0; i < 8; i++) {
            if (u + uc + offset[i].x >= 0 && u + uc + offset[i].x < branching &&
                v + vc + offset[i].y >= 0 && v + vc + offset[i].y < branching &&
                w + wc + offset[i].z >= 0 && w + wc + offset[i].z < branching)
            {
                skip = child[node_id][u + uc + offset[i].x][v + vc + offset[i].y][w + wc + offset[i].z];
                // Keep track of neighbor coordinates as well as neighbor indices. Coordinates cannot be computed
                // at the end due to dependency on current cube size.
                neigh_coo[i] = make_float3(
                    (floorf(in_coo.x * cube_sz + offset2[i].x + 1e-5) + 0.5) / cube_sz,
                    (floorf(in_coo.y * cube_sz + offset2[i].y + 1e-5) + 0.5) / cube_sz,
                    (floorf(in_coo.z * cube_sz + offset2[i].z + 1e-5) + 0.5) / cube_sz
                );
                clamp_coord(neigh_coo[i], 1 / (cube_sz * branching), 1 - 1 / (cube_sz * branching));
                // Simpler formula (without clamping)
                // (floor((in_coordinate[0] + offset2[i][0] / (cube_sz * 2)) * cube_sz + 1e-5) + 0.5) / cube_sz,
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


template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void _dev_query_sum(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf,
    float3 & __restrict__ coordinate,
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

    scalar_t* data_at_coo = _dev_query_single<scalar_t, branching>(
        tree_data, child, is_child_leaf, make_float3(indices[i][0], indices[i][1], indices[i][2])
    );
    for (int32_t j = 0; j < data_dim; j++) {
        data_at_coo[j] = values[i][j];
    }
}


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

    if (parent_sum) {
        _dev_query_sum<scalar_t, branching, data_dim>(
            tree_data, child, is_child_leaf,
            make_float3(indices[i][0], indices[i][1], indices[i][2]),
            &out_values[i][0]
        );
    } else {
        scalar_t* data_at_coo = _dev_query_single<scalar_t, branching>(
            tree_data, child, is_child_leaf, make_float3(indices[i][0], indices[i][1], indices[i][2])
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
) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    _dev_query_interp<scalar_t, branching, data_dim>(
        tree_data, child, is_child_leaf,
        make_float3(indices[i][0], indices[i][1], indices[i][2]),
        &weights[i][0],
        &out_values[i][0],
        parent_sum
    );
}



template <typename scalar_t, int32_t branching, int32_t data_dim>
void set_octree(at::Tensor &indices,
                const at::Tensor &vals,
                at::Tensor &data,
                at::Tensor &child,
                at::Tensor &is_child_leaf,
                at::Tensor &parent,
                at::Tensor &depth,
                const bool update_avg,
                const bool parent_sum)
{
    size_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        return;
    }

    octree_set_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        vals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements
    );

    // Set all parents to be their child's average (bottom to top)
    // Remove the average from the children
    int32_t node_size = branching * branching * branching;
    if (update_avg) {
        auto max_depth = depth.max();
        for (int i = max_depth; i > 0; i--) {
            auto child_ids = (depth == torch::tensor({i})).nonzero().squeeze();
            auto parent_ids = parent.index({child_ids}).to(torch::kInt64);
            data.index_put_({parent_ids}, torch::tensor({0}));
            data.scatter_add_(
                0, parent_ids.unsqueeze(-1).expand(parent_ids.size(0), data_dim), data.index(child_ids));
            data.index({parent_ids}).div_(node_size);
            if (parent_sum) {
                data.scatter_add_(
                    0, child_ids.unsqueeze(-1).expand(child_ids.size(0), data_dim), -data.index(parent_ids));
            }
        }
    }
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
at::Tensor query_octree(at::Tensor &indices,
                        torch::Tensor &data,
                        torch::Tensor &child,
                        torch::Tensor &is_child_leaf,
                        const bool parent_sum)
{
    size_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        torch::Tensor undefined;
        return undefined;
    }

    // Create output tensor
    at::Tensor values_out = torch::empty({n_elements, data_dim}, data.options());
    octree_query_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements,
        parent_sum
    );
    return values_out;
}

template <typename scalar_t, int32_t branching, int32_t data_dim>
std::tuple<at::Tensor, at::Tensor> query_interp_octree(at::Tensor &indices,
                                                       at::Tensor &data,
                                                       at::Tensor &child,
                                                       at::Tensor &is_child_leaf,
                                                       const bool parent_sum)
{
    size_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        torch::Tensor undefined;
        return undefined;
    }

    // Create output tensors
    at::Tensor values_out = torch::empty({n_elements, data_dim}, data.options());
    at::Tensor weights_out = torch::empty({n_elements, 8}, indices.options());
    octree_query_interp_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        weights_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        n_elements,
        parent_sum
    );
    return std::make_tuple(values_out, weights_out);
}

