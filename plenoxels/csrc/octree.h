#pragma once

#include <tuple>
#include <torch/extension.h>
#include <c10/util/typeid.h>


template <typename scalar_t, int32_t branching, int32_t data_dim>
struct Octree {
//    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data_acc;
//    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child_acc;
//    torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf_acc;
//    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> parent_acc;
//    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> depth_acc;

    Octree(int32_t levels, bool parent_sum, torch::Device device) : parent_sum(parent_sum) {
        max_depth = 0;

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

//        data_acc = data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
//        child_acc = child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
//        is_child_leaf_acc = is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>();
//        parent_acc = parent.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
//        depth_acc = depth.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
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

};


template <typename scalar_t, int32_t branching, int32_t data_dim>
void set_octree(Octree<scalar_t, branching, data_dim> tree, torch::Tensor indices, const torch::Tensor vals, const bool update_avg);

//template <typename scalar_t, int32_t branching, int32_t data_dim>
//torch::Tensor query_octree(Octree<scalar_t, branching, data_dim> tree, torch::Tensor indices);

//template <typename scalar_t, int32_t branching, int32_t data_dim>
//std::tuple<torch::Tensor, torch::Tensor> query_interp_octree(Octree<scalar_t, branching, data_dim> tree, torch::Tensor indices);
//
//template <typename scalar_t, int32_t branching, int32_t data_dim>
//void refine_octree(Octree<scalar_t, branching, data_dim> tree, const torch::optional<torch::Tensor> opt_leaves);


template <int32_t branching>
torch::Tensor pack_index_3d(const torch::Tensor leaves) {
    auto multiplier = torch::tensor({branching * branching * branching, branching * branching, branching, 1}, leaves.options());
    return leaves.mul(multiplier.unsqueeze(0)).sum(-1);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void _resize_add_cap(Octree<scalar_t, branching, data_dim> tree, const int64_t num_new_internal)
{
    tree.child = torch::cat({
        tree.child,
        torch::zeros({num_new_internal, branching, branching, branching}, tree.child.options())
    });
    tree.data = torch::cat({
        tree.data,
        torch::zeros({num_new_internal * branching * branching * branching, data_dim}, tree.data.options())
    });
    tree.is_child_leaf = torch::cat({
        tree.is_child_leaf,
        torch::ones({num_new_internal, branching, branching, branching}, tree.is_child_leaf.options())
    });
    tree.parent = torch::cat({
        tree.parent,
        torch::zeros({(int64_t)num_new_internal * branching * branching * branching}, tree.parent.options())
    });
    tree.depth = torch::cat({
        tree.depth,
        torch::zeros({(int64_t)num_new_internal * branching * branching * branching}, tree.depth.options())
    });

//    tree.data_acc = tree.data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
//    tree.child_acc = tree.child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
//    tree.is_child_leaf_acc = tree.is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>();
//    tree.parent_acc = tree.parent.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
//    tree.depth_acc = tree.depth.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void refine_octree(Octree<scalar_t, branching, data_dim> tree, const torch::optional<torch::Tensor> opt_leaves)
{
    const auto leaves = opt_leaves.has_value() ? opt_leaves.value() : tree.is_child_leaf.nonzero();
    const int64_t total_nodes = tree.data.size(0);
    const int64_t new_internal = leaves.size(0);
    const int64_t new_total_nodes = total_nodes + new_internal * tree.node_size;
    if (new_internal == 0) {
        return;
    }
    _resize_add_cap(tree, new_internal);

    if (total_nodes == 1) // root node is an exception
    {
        tree.child.index_put_({Ellipsis}, torch::arange(total_nodes, new_total_nodes, tree.child.options()));
        tree.is_child_leaf.index_put_({Ellipsis}, true);
        tree.parent.index_put_({Ellipsis}, 0);
        tree.depth.index_put_({Ellipsis}, 1);
        tree.max_depth += 1;
    }
    else
    {
        // child
        torch::Tensor new_child_ids =
            torch::arange(total_nodes, new_total_nodes, tree.child.options()).view({-1, branching, branching, branching})
            - torch::arange(tree.n_internal, tree.n_internal + new_internal, tree.child.options()).view({-1, 1, 1, 1});
        tree.child.index_put_({Slice(tree.n_internal, tree.n_internal + new_internal, 1), Ellipsis},
                         new_child_ids);
        // is_child_leaf
        auto sel = leaves.unbind(1);
        tree.is_child_leaf.index_put_({sel[0], sel[1], sel[2], sel[3]}, torch::tensor({true}));
        // parent_depth
        auto packed_leaves = pack_index_3d<branching>(leaves).repeat_interleave(tree.node_size) + 1;  // +1 for the invisible root node.
        tree.parent.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            packed_leaves
        );
        auto old_depth = tree.depth.index({packed_leaves});
        torch::Tensor new_max_depth = torch::max(old_depth);
        int new_max_depth_i = new_max_depth.item<int>() + 1;
        tree.max_depth = tree.max_depth > new_max_depth_i ? tree.max_depth : new_max_depth_i;
        tree.depth.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            old_depth + 1
        );
    }
    tree.n_internal += new_internal;
}
