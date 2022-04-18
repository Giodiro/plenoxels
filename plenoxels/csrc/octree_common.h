#pragma once

#include <torch/extension.h>

constexpr uint32_t n_threads_linear = 128;

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

template <typename T, int32_t b, int32_t d>
torch::Tensor query_octree_impl (torch::Tensor &indices,
                                 torch::Tensor &data,
                                 torch::Tensor &child,
                                 torch::Tensor &is_child_leaf,
                                 const bool parent_sum);
