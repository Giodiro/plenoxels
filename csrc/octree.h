#pragma once

#include <tuple>
#include <c10/util/typeid.h>
#include <torch/extension.h>


template <typename scalar_t, int32_t branching, int32_t data_dim>
void set_octree(Octree<scalar_t, branching, data_dim> &tree, at::Tensor &indices, const at::Tensor &vals, const bool update_avg);

template <typename scalar_t, int32_t branching, int32_t data_dim>
at::Tensor query_octree(Octree<scalar_t, branching, data_dim> &tree, at::Tensor &indices);

template <typename scalar_t, int32_t branching, int32_t data_dim>
std::tuple<at::Tensor, at::Tensor> query_interp_octree(Octree<scalar_t, branching, data_dim> &tree, at::Tensor &indices);

template <typename scalar_t, int32_t branching, int32_t data_dim>
void refine_octree(Octree<scalar_t, branching, data_dim> &tree, const at::optional<at::Tensor> &opt_leaves);


template <typename scalar_t, int32_t branching, int32_t data_dim>
struct Octree {
    int64_t n_internal;
    int32_t max_depth;
    const bool parent_sum;
    const int32_t node_size;

    at::Tensor data;
    at::Tensor child;
    at::Tensor is_child_leaf;
    at::Tensor parent;
    at::Tensor depth;

    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> data_acc;
    torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child_acc;
    torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> is_child_leaf_acc;
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> parent_acc;
    torch::PackedTensorAccessor64<int32_t, 2, torch::RestrictPtrTraits> depth_acc;

    Octree(int32_t levels, bool parent_sum, torch::Device device) : parent_sum(parent_sum) {
        node_size = branching * branching * branching;
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

        data_acc = data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>();
        child_acc = child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>();
        is_child_leaf_acc = is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>();
        parent_acc = parent.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
        depth_acc = depth.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>();
    }
    ~Octree() { }
};
