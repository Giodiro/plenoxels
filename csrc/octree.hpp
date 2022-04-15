#pragma once

#include <tuple>
#include <c10/util/typeid.h>
#include <torch/extension.h>

using namespace torch::indexing;


at::Tensor query_octree(at::Tensor &indices,
                        torch::Tensor &data,
                        torch::Tensor &child,
                        torch::Tensor &is_child_leaf,
                        const bool parent_sum);


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

//        void refine(const at::optional<at::Tensor> & opt_leaves);
//        void set(at::Tensor indices, const at::Tensor vals, const bool update_avg);
        at::Tensor query(at::Tensor indices) {
            return query_octree(indices, data, child, is_child_leaf, _parent_sum);
        }
//        std::tuple<at::Tensor, at::Tensor> query_interp(at::Tensor indices);
};



