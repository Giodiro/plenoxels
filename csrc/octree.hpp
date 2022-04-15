#pragma once

#include <tuple>
#include <c10/util/typeid.h>
#include <torch/extension.h>

using namespace torch::indexing;


template <typename scalar_t, int32_t branching, int32_t data_dim>
at::Tensor query_octree(at::Tensor &indices,
                        torch::Tensor &data,
                        torch::Tensor &child,
                        torch::Tensor &is_child_leaf,
                        const bool parent_sum);

template <typename scalar_t, int32_t branching, int32_t data_dim>
std::tuple<at::Tensor, at::Tensor> query_interp_octree(at::Tensor &indices,
                                                       at::Tensor &data,
                                                       at::Tensor &child,
                                                       at::Tensor &is_child_leaf,
                                                       const bool parent_sum);

template <typename scalar_t, int32_t branching, int32_t data_dim>
void set_octree(at::Tensor &indices,
                const at::Tensor &vals,
                at::Tensor &data,
                at::Tensor &child,
                at::Tensor &is_child_leaf,
                at::Tensor &parent,
                at::Tensor &depth,
                const bool update_avg,
                const bool parent_sum,
                const int32_t max_depth);

template <typename scalar_t, int32_t branching, int32_t data_dim>
class Octree {
    private:
        int64_t _n_internal;
        bool _parent_sum;
        int32_t _node_size;
        int32_t _max_depth;
        void _resize_add_cap(const int64_t num_new_internal)
        {
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
    public:
        at::Tensor data;
        at::Tensor child;
        at::Tensor is_child_leaf;
        at::Tensor parent;
        at::Tensor depth;

        Octree(int32_t levels, bool parent_sum, torch::Device device) {
            _parent_sum = parent_sum;
            _node_size = branching * branching * branching;
            _max_depth = 0;

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
        void set(at::Tensor indices, const at::Tensor vals, const bool update_avg) {
            set_octree<scalar_t, branching, data_dim>(indices, vals, data, child, is_child_leaf, parent, depth, update_avg, _parent_sum);
        }
        at::Tensor query(at::Tensor indices) {
            return query_octree<scalar_t, branching, data_dim>(indices, data, child, is_child_leaf, _parent_sum);
        }
        std::tuple<at::Tensor, at::Tensor> query_interp(at::Tensor indices) {
            return query_interp_octree<scalar_t, branching, data_dim>(indices, data, child, is_child_leaf, _parent_sum);
        }
};


template <int32_t branching>
at::Tensor pack_index_3d(const at::Tensor & leaves) {
    auto multiplier = torch::tensor({branching * branching * branching, branching * branching, branching, 1}, leaves.options());
    return leaves.mul(multiplier.unsqueeze(0)).sum(-1);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
void Octree<scalar_t, branching, data_dim>::refine(const at::optional<at::Tensor> & opt_leaves)
{
    int32_t node_size = branching * branching * branching;
    const auto leaves = opt_leaves.has_value() ? opt_leaves.value() : is_child_leaf.nonzero();
    const int64_t total_nodes = data.size(0);
    const int64_t new_internal = leaves.size(0);
    const int64_t new_total_nodes = total_nodes + new_internal * node_size;

    if (new_internal == 0) {
        return;
    }
    _resize_add_cap(new_internal);

    if (total_nodes == 1) // root node is an exception
    {
        child.index_put_({Ellipsis}, torch::arange(total_nodes, new_total_nodes, child.options()));
        is_child_leaf.index_put_({Ellipsis}, true);
        parent.index_put_({Ellipsis}, 0);
        depth.index_put_({Ellipsis}, 1);
        _max_depth += 1;
    }
    else
    {
        // child
        torch::Tensor new_child_ids =
            torch::arange(total_nodes, new_total_nodes, child.options()).view({-1, branching, branching, branching})
            - torch::arange(_n_internal, _n_internal + new_internal, child.options()).view({-1, 1, 1, 1});
        child.index_put_({Slice(_n_internal, _n_internal + new_internal, 1), Ellipsis},
                         new_child_ids);
        // is_child_leaf
        auto sel = leaves.unbind(1);
        is_child_leaf.index_put_({sel[0], sel[1], sel[2], sel[3]}, torch::tensor({true}));
        // parent_depth
        auto packed_leaves = pack_index_3d<branching>(leaves).repeat_interleave(_node_size) + 1;  // +1 for the invisible root node.
        parent.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            packed_leaves
        );
        auto old_depth = depth.index({packed_leaves});
        _max_depth = max(_max_depth, torch::max(old_depth).item<int32_t>() + 1);
        depth.index_put_(
            {Slice(total_nodes, new_total_nodes)},
            old_depth + torch::tensor({1}) // TODO: Think wrapping is unnecessary
        );
    }
    _n_internal += new_internal;
}
