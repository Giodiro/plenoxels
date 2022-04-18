#pragma once

#include <tuple>
#include <torch/extension.h>
#include <c10/util/typeid.h>

#include "octree_common.h"

using namespace torch::indexing;

template <typename scalar_t, int32_t branching, int32_t data_dim>
struct Octree {
    Octree(int32_t levels, bool parent_sum, torch::Device device) : parent_sum(parent_sum) {
        max_depth = 0;
        n_internal = 0;

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

    int64_t n_internal;
    int32_t max_depth;
    bool parent_sum;
    const int32_t node_size = branching * branching * branching;

    torch::Tensor data;
    torch::Tensor child;
    torch::Tensor is_child_leaf;
    torch::Tensor parent;
    torch::Tensor depth;

    void _resize_add_cap(const int64_t num_new_internal);
    void refine_octree(const torch::optional<torch::Tensor> &opt_leaves);

    torch::Tensor query_octree(torch::Tensor &indices);
    std::tuple<torch::Tensor, torch::Tensor> query_interp_octree(torch::Tensor &indices);
    void set_octree(torch::Tensor &indices, const torch::Tensor &vals, const bool update_avg);
};


template <int32_t branching>
torch::Tensor pack_index_3d(const torch::Tensor &leaves) {
    auto multiplier = torch::tensor({branching * branching * branching, branching * branching, branching, 1}, leaves.options());
    return leaves.mul(multiplier.unsqueeze(0)).sum(-1);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void Octree<scalar_t, branching, data_dim>::_resize_add_cap(const int64_t num_new_internal)
{
    printf("Resizing with %ld new internal\n", num_new_internal);
    child = torch::cat({
        child,
        torch::zeros({num_new_internal, branching, branching, branching}, child.options())
    });
    printf("child size(0): %ld\n", this->child.size(0));
    data = torch::cat({
        data,
        torch::zeros({num_new_internal * branching * branching * branching, data_dim}, data.options())
    });
    printf("data size(0): %ld\n", data.size(0));
    is_child_leaf = torch::cat({
        is_child_leaf,
        torch::ones({num_new_internal, branching, branching, branching}, is_child_leaf.options())
    });
    printf("child leaf size(0): %ld\n", is_child_leaf.size(0));
    parent = torch::cat({
        parent,
        torch::zeros({(int64_t)num_new_internal * branching * branching * branching}, parent.options())
    });
    printf("parent size(0): %ld\n", parent.size(0));
    depth = torch::cat({
        depth,
        torch::zeros({(int64_t)num_new_internal * branching * branching * branching}, depth.options())
    });
    printf("depth size(0): %ld\n", depth.size(0));
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void Octree<scalar_t, branching, data_dim>::refine_octree(const torch::optional<torch::Tensor> &opt_leaves)
{
    const auto leaves = opt_leaves.has_value() ? opt_leaves.value() : is_child_leaf.nonzero();
    const int64_t total_nodes = data.size(0);
    int64_t new_internal = leaves.size(0);

    if (total_nodes == 1) // root node is an exception
    {
        new_internal = 1;
        _resize_add_cap(new_internal);
        const int64_t new_total_nodes = total_nodes + new_internal * node_size;
        child.index_put_({Ellipsis}, torch::arange(total_nodes, new_total_nodes, child.options()).view({-1, branching, branching, branching}));
        is_child_leaf.index_put_({Ellipsis}, true);
        parent.index_put_({Ellipsis}, 0);
        depth.index_put_({Ellipsis}, 1);
        max_depth += 1;
    }
    else
    {
        const int64_t new_total_nodes = total_nodes + new_internal * node_size;
        if (new_internal == 0) {
            return;
        }
        _resize_add_cap(new_internal);
        printf("resize complete\n");
        // child
        torch::Tensor new_child_ids =
            torch::arange(total_nodes, new_total_nodes, child.options()).view({-1, branching, branching, branching})
            - torch::arange(n_internal, n_internal + new_internal, child.options()).view({-1, 1, 1, 1});
        printf("n_internal %d, new internal %d - new ids shape(0) %ld\n", n_internal, new_internal, new_child_ids.size(0));
        child.index_put_({Slice(n_internal, n_internal + new_internal), Ellipsis}, new_child_ids);
        printf("child complete\n");
        // is_child_leaf
        auto sel = leaves.unbind(1);
        is_child_leaf.index_put_({sel[0], sel[1], sel[2], sel[3]}, torch::tensor({true}));
        printf("child leaf complete\n");
        // parent_depth
        auto packed_leaves = pack_index_3d<branching>(leaves).repeat_interleave(node_size) + 1;  // +1 for the invisible root node.
        parent.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            packed_leaves
        );
        printf("parent complete\n");
        auto old_depth = depth.index({packed_leaves});
        torch::Tensor new_max_depth = torch::max(old_depth);
        int new_max_depth_i = new_max_depth.item<int>() + 1;
        max_depth = max_depth > new_max_depth_i ? max_depth : new_max_depth_i;
        depth.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            old_depth + 1
        );
        printf("depth complete\n");
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
    torch::Tensor values_out = torch::empty({n_elements, data_dim}, data.options());
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

    // Create output tensors
    torch::Tensor values_out = torch::empty({n_elements, data_dim}, data.options());
    torch::Tensor weights_out = torch::empty({n_elements, 8}, indices.options());
    octree_query_interp_kernel<scalar_t, branching, data_dim><<<n_blocks_linear(n_elements), n_threads_linear>>>(
        data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
        child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
        is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
        indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        values_out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        weights_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
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
            data.index_put_({parent_ids}, torch::tensor({0}, data.options()));
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
