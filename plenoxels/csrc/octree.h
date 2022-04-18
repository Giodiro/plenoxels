#pragma once

#include <tuple>
#include <torch/extension.h>
#include <c10/util/typeid.h>

#include "octree_common.h"

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
    void refine_octree(const torch::optional<torch::Tensor> opt_leaves);

    torch::Tensor query_octree(torch::Tensor indices) {
        return query_octree_impl<scalar_t, branching, data_dim>(indices, this->data, this->child, this->is_child_leaf, this->parent_sum);
    }
};


using namespace torch::indexing;

template <int32_t branching>
torch::Tensor pack_index_3d(const torch::Tensor leaves) {
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
void Octree<scalar_t, branching, data_dim>::refine_octree(const torch::optional<torch::Tensor> opt_leaves)
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
