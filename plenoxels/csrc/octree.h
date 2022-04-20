#pragma once

#include <tuple>
#include <iostream>
#include <torch/extension.h>
#include <math.h>
#include <c10/util/typeid.h>

#include "octree_common.h"

using namespace torch::indexing;

template <typename scalar_t, int32_t branching, int32_t data_dim>
struct Octree {
    Octree(int32_t levels,
           bool parent_sum,
           torch::Device device,
           torch::optional<torch::Tensor> radius,
           torch::optional<torch::Tensor> center) : parent_sum(parent_sum) {
        max_depth = 0;
        n_internal = 0;

        const auto data_dt = caffe2::TypeMeta::Make<scalar_t>().toScalarType();

        // 1 / (2 * radius); default: 1 (equiv. diameter = 1).
        this->scaling = radius.has_value() ?
            (0.5 / radius.value()).to(device) :
            torch::tensor({1, 1, 1}, torch::dtype(torch::kFloat32).layout(torch::kStrided).device(device));
        // ??
        this->offset = center.has_value() ?
            (0.5 - center.value().to(device) * this->scaling) :
            (0.5 - 0.5 * this->scaling);
        std::cout << "scaling " << this->scaling << std::endl;
        std::cout << "offset " << this->offset << std::endl;

        data = torch::zeros({1, data_dim},
            torch::dtype(data_dt).layout(torch::kStrided).device(device).requires_grad(true));
        child = torch::zeros({0, branching, branching, branching},
            torch::dtype(torch::kInt32).layout(torch::kStrided).device(device));
        is_child_leaf = torch::ones({0, branching, branching, branching},
            torch::dtype(torch::kBool).layout(torch::kStrided).device(device));
        parent = torch::tensor({-1},
            torch::dtype(torch::kInt32).layout(torch::kStrided).device(device));
        depth = torch::zeros({1},
            torch::dtype(torch::kInt32).layout(torch::kStrided).device(device));

        int64_t n_internal_prealloc = 0;
        for (int i = 0; i < levels; i++) {
            n_internal_prealloc += pow(node_size, i);
        }
        _resize_add_cap(n_internal_prealloc);
    }

    int64_t n_internal;
    int32_t max_depth;
    bool parent_sum;
    const int32_t node_size = branching * branching * branching;

    torch::Tensor data;
    torch::Tensor child;
    torch::Tensor is_child_leaf;
    torch::Tensor parent;
    torch::Tensor depth;
    torch::Tensor offset;
    torch::Tensor scaling;

    void _resize_add_cap(const int64_t num_new_internal);
    void refine_octree(const torch::optional<torch::Tensor> &opt_leaves);

    torch::Tensor query_octree(torch::Tensor &indices);
    std::tuple<torch::Tensor, torch::Tensor> query_interp_octree(torch::Tensor &indices);
    void set_octree(torch::Tensor &indices, const torch::Tensor &vals, const bool update_avg);

    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data_acc() {
        return data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
    }
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child_acc() {
        return child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
    }
    torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf_acc() {
        return is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>();
    }
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> parent_acc() {
        return parent.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();
    }
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> depth_acc() {
        return depth.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>();
    }
    float * offset_ptr() {
        return offset.data_ptr<float>();
    }
    float * scaling_ptr() {
        return scaling.data_ptr<float>();
    }
    Octree<scalar_t, branching, data_dim>& eval() {
        return *this;
    }
    Octree<scalar_t, branching, data_dim>& train() {
        return *this;
    }
};


template <int32_t branching>
torch::Tensor pack_index_3d(const torch::Tensor &leaves) {
    auto multiplier = torch::tensor({branching * branching * branching, branching * branching, branching, 1}, leaves.options());
    return leaves.mul(multiplier.unsqueeze(0)).sum(-1);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void Octree<scalar_t, branching, data_dim>::_resize_add_cap(const int64_t num_new_internal)
{
    torch::NoGradGuard no_grad;
    printf("[Octree] Adding %ld nodes to tree\n", num_new_internal);
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
        torch::zeros({(int64_t)num_new_internal * branching * branching * branching}, parent.options())
    });
    depth = torch::cat({
        depth,
        torch::zeros({(int64_t)num_new_internal * branching * branching * branching}, depth.options())
    });
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void Octree<scalar_t, branching, data_dim>::refine_octree(const torch::optional<torch::Tensor> &opt_leaves)
{
    torch::NoGradGuard no_grad;
    torch::Tensor leaves;
    if (opt_leaves.has_value()) {  // Check validity of user-supplied leaves
        auto sel = opt_leaves.value().unbind(1);
        auto valid_leaves = is_child_leaf.index({sel[0], sel[1], sel[2], sel[3]}).logical_and(sel[0] < n_internal);
        leaves = opt_leaves.value().index({valid_leaves, Ellipsis});
    } else {
        leaves = is_child_leaf.index({Slice(0, n_internal), Ellipsis}).nonzero();
    }
    const bool is_root_node = n_internal == 0;
    const int64_t new_internal = is_root_node ? 1 : leaves.size(0);
    const int64_t alloc_internal = child.size(0);
    printf("Current internal %ld - New internal %ld\n", n_internal, new_internal);

    if (new_internal <= 0) {
        return;
    }
    if (new_internal + n_internal > alloc_internal) {
        _resize_add_cap(new_internal);
    }

    const int64_t cur_tot_nodes = n_internal * node_size + 1;  // +1 for root node
    const int64_t new_tot_nodes = cur_tot_nodes + new_internal * node_size;
    printf("Current tot nodes %ld - New tot nodes %ld\n", cur_tot_nodes, new_tot_nodes);

    // Child tensor
    torch::Tensor new_child_ids =
        torch::arange(cur_tot_nodes, new_tot_nodes, child.options()).view({-1, branching, branching, branching})
        - torch::arange(n_internal, n_internal + new_internal, child.options()).view({-1, 1, 1, 1});
    child.index_put_({Slice(n_internal, n_internal + new_internal), Ellipsis}, new_child_ids);

    // Is-child-leaf tensor (fill new internal nodes with False)
    auto sel = leaves.unbind(1);
    is_child_leaf.index_put_({sel[0], sel[1], sel[2], sel[3]}, torch::tensor({false}, is_child_leaf.options()));

    // Parent + Depth
    if (is_root_node) {
        parent.index_put_({Slice(cur_tot_nodes, new_tot_nodes)}, 0);
        depth.index_put_({Slice(cur_tot_nodes, new_tot_nodes)}, 1);
        max_depth = 1;
    } else {
        auto packed_leaves = pack_index_3d<branching>(leaves).repeat_interleave(node_size) + 1;  // +1 for the invisible root node.
        parent.index_put_(
            {Slice(cur_tot_nodes, new_tot_nodes)},
            packed_leaves
        );
        auto old_depth = depth.index({packed_leaves});
        torch::Tensor new_max_depth = torch::max(old_depth);
        int new_max_depth_i = new_max_depth.item<int>() + 1;
        max_depth = max_depth > new_max_depth_i ? max_depth : new_max_depth_i;
        depth.index_put_(
            {Slice(cur_tot_nodes, new_tot_nodes)},
            old_depth + 1
        );
    }
    n_internal += new_internal;
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
torch::Tensor Octree<scalar_t, branching, data_dim>::query_octree (torch::Tensor &indices)
{
    int64_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        torch::Tensor undefined;
        return undefined;
    }
    // Create output tensor
    torch::Tensor values_out = torch::empty({n_elements, data_dim}, data.options().requires_grad(false));
    octree_query_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements,
        parent_sum
    );
    return values_out;
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
std::tuple<torch::Tensor, torch::Tensor> Octree<scalar_t, branching, data_dim>::query_interp_octree(torch::Tensor &indices)
{
    int64_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        torch::Tensor undefined;
        return std::make_tuple(undefined, undefined);
    }

    // Temporary tensors
    torch::Tensor neighbor_coo = torch::empty({n_elements, 8, 3}, torch::dtype(torch::kFloat32).device(data.device()).layout(data.layout()));
    torch::Tensor neighbor_ids = torch::full({n_elements, 8}, -1, torch::dtype(torch::kInt64).device(data.device()).layout(data.layout()));

    // Create output tensors
    torch::Tensor values_out = torch::empty({n_elements, data_dim}, data.options().requires_grad(false));
    torch::Tensor weights_out = torch::empty({n_elements, 8}, indices.options());
    octree_query_interp_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        weights_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        neighbor_coo.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        neighbor_ids.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        n_elements,
        parent_sum
    );
    return std::make_tuple(values_out, weights_out);
}


template <class scalar_t, int32_t branching, int32_t data_dim>
void Octree<scalar_t, branching, data_dim>::set_octree(torch::Tensor &indices, const torch::Tensor &vals, const bool update_avg)
{
    int64_t n_elements = indices.size(0);
    if (n_elements <= 0) {
        return;
    }
    octree_set_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        vals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        n_elements
    );

    // Set all parents to be their child's average (bottom to top)
    // Remove the average from the children
    int32_t node_size = branching * branching * branching;
    if (update_avg) {
        for (int i = max_depth; i > 0; i--) {
            auto child_ids = (depth == torch::tensor({i}, depth.options())).nonzero().squeeze();
            auto parent_ids = parent.index({child_ids}).to(torch::kInt64);
            data.index_put_({parent_ids}, torch::tensor({0}, data.options().requires_grad(false)));
            data.scatter_add_(
                0, parent_ids.unsqueeze(-1).expand(parent_ids.size(0), data_dim), data.index({child_ids}));
            data.index({parent_ids}).div_(node_size);
            if (parent_sum) {
                data.scatter_add_(
                    0, child_ids.unsqueeze(-1).expand(child_ids.size(0), data_dim), -data.index({parent_ids}));
            }
        }
    }
}
