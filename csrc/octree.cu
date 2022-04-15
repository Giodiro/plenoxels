#pragma once

#include <tuple>
#include <c10/util/typeid.h>
#include <torch/extension.h>

#include "octree_common.h"


template <typename scalar_t, int32_t branching, int32_t data_dim>
class Octree {
    private:
        size_t _n_internal;
        bool _parent_sum;
        int32_t _node_size;
        void _resize_add_cap(const size_t num_new_internal);

    public:
        at::Tensor data;
        at::Tensor child;
        at::Tensor is_child_leaf;
        at::Tensor parent;
        at::Tensor depth;

        Octree(int32_t levels, bool parent_sum, torch::Device device) {
            _parent_sum = parent_sum;
            _node_size = branching * branching * branching;

            const auto data_dt = caffe2::TypeMeta::Make<scalar_t>().toScalarType();

            data = torch::zeros({1, data_dim},
                torch::dtype(data_dt).layout(torch::kStrided).device(device));
            child = torch::zeros({0, branching, branching, branching},
                torch::dtype(torch::kInt32).layout(torch::kStrided).device(device));
            is_child_leaf = torch::ones({0, branching, branching, branching},
                torch::dtype(torch::kBool).layout(torch::kStrided).device(device));
            parent = torch::zeros({1},
                torch::dtype(torch::kInt32).layout(torch::kStrided).device(device));
            depth = torch::zeros({1},
                torch::dtype(torch::kInt32).layout(torch::kStrided).device(device));
        }
        ~Octree() { }

        void refine(const at::optional<at::Tensor> & opt_leaves);
        void set(at::Tensor indices, const at::Tensor vals, const bool update_avg);
        at::Tensor query(at::Tensor indices);
        std::tuple<at::Tensor, at::Tensor> query_interp(at::Tensor indices);
};


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
    clamp_coord(coordinate, 0.0, 1.0 - 1e-9);
    int32_t node_id = 0;
    int32_t u, v, w, skip, i, j;
    int32_t uc, vc, wc;

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
        traverse_tree_level<branching>(coordinate, &u, &v, &w);
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
                    (floorf(in_coo.x * cube_sz + offset2[i].x / 2 + 1e-5) + 0.5) / cube_sz,
                    (floorf(in_coo.y * cube_sz + offset2[i].y / 2 + 1e-5) + 0.5) / cube_sz,
                    (floorf(in_coo.z * cube_sz + offset2[i].z / 2 + 1e-5) + 0.5) / cube_sz
                );
                neigh_coo[i] = clamp_coord(neigh_coo, 1 / (cube_sz * branching), 1 - 1 / (cube_sz * branching));
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
                        out_val[i] += weights[j] * tree_data[valid_neighbors[j]][i];
                    } else {
                        out_val[i] = weights[j] * tree_data[valid_neighbors[j]][i];
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
    scalar_t* __restrict__ out_val,
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
    float3 & __restrict__ coordinate,
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
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits weights,
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


void refine(const at::optional<at::Tensor> & opt_leaves) {
    const auto leaves = opt_leaves.has_value() ? opt_leaves.value() : is_child_leaf.nonzero();
    const size_t total_nodes = data.size(0);
    const size_t new_internal = leaves.size(0);
    const size_t new_total_nodes = total_nodes + new_internal * _node_size;

    _resize_add_cap(new_internal);

    if (total_nodes == 1) // root node is an exception
    {
        child.index_put_({Ellipsis}, torch::arange(total_nodes, new_total_nodes, child.options()));
        is_child_leaf.index_put_({Ellipsis}, true);
        parent.index_put_({Ellipsis}, 0);
        depth.index_put_({Ellipsis}, 1);
    }
    else
    {
        // child
        torch::Tensor new_child_ids =
            torch::arange(total_nodes, new_total_nodes, child.options()).view({-1, branching, branching, branching})
            - torch::arange(_n_internal, _n_internal + new_internal, 1).view({-1, 1, 1, 1});
        child.index_put_({Slice(_n_internal, _n_internal + new_internal, 1), Ellipsis},
                         new_child_ids);
        // is_child_leaf
        is_child_leaf.index_put_(leaves.unbind(1), true);
        // parent_depth
        auto packed_leaves = pack_index_3d<branching>(leaves).repeat_interleave(_node_size) + 1;  // +1 for the invisible root node.
        parent.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            packed_leaves
        );
        depth.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            depth.index({packed_leaves}) + 1
        );
    }
    _n_internal += new_internal;
}

void _resize_add_cap(const size_t num_new_internal) {
    child = torch::cat({
        child,
        torch::zeros({num_new_internal, branching, branching, branching}, child.options())
    });
    data = torch::cat({
        data,
        torch::zeros({num_new_internal * branching * branching * branching, data_dim}, data.options())
    });
    is_child_leaf = torch::cat({
        is_child_leaf,
        torch::ones({num_new_internal, branching, branching, branching}, is_child_leaf.options())
    });
    parent = torch::cat({
        parent,
        torch::zeros({num_new_internal * branching * branching * branching}, parent.options())
    });
    depth = torch::cat({
        depth,
        torch::zeros({num_new_internal * branching * branching * branching}, depth.options())
    });
}

void set(at::Tensor indices, const at::Tensor vals, const bool update_avg) {
    size_t n_elements = indices.shape(0);
    if (n_elements <= 0) {
        return;
    }

    octree_set_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements
    );

    // Set all parents to be their child's average (bottom to top)
    // Remove the average from the children
    if (update_avg) {
        max_depth = depth.max();
        for (int i = max_depth; i > 0; i--) {
            auto child_ids = (depth == torch::tensor({i})).nonzero().squeeze();
            auto parent_ids = parent.index({child_ids}).to(torch::kInt64);
            data.index_put_(parent_ids, torch::tensor({0}));
            data.scatter_add_(
                0, parent_ids.unsqueeze(-1).expand(parent_ids.size(0), data_dim), data.index(child_ids));
            data.index(parent_ids).div_(_node_size);
            if (_parent_sum) {
                data.scatter_add_(
                    0, child_ids.unsqueeze(-1).expand(child_ids.size(0), data_dim), -data.index(parent_ids));
            }
        }
    }
}

at::Tensor query(at::Tensor indices) {
    size_t n_elements = indices.shape(0);
    if (n_elements <= 0) {
        return torch::tensor();  // undefined tensor
    }

    // Create output tensor
    at::Tensor values_out = torch::empty({n_elements, data_dim}, data.options());
    octree_query_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements,
        _parent_sum
    );
    return values_out;
}

std::tuple<at::Tensor, at::Tensor> query_interp(at::Tensor indices) {
    size_t n_elements = indices.shape(0);
    if (n_elements <= 0) {
        return torch::tensor();  // undefined tensor
    }

    // Create output tensors
    at::Tensor values_out = torch::empty({n_elements, data_dim}, data.options());
    at::Tensor weights_out = torch::empty({n_elements, 8}, indices.options());
    octree_query_interp_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        weights_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        n_elements,
        _parent_sum
    );
    return make_tuple(values_out, weights_out);
}
