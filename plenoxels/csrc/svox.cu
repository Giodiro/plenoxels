/*
 * Copyright 2021 PlenOctree Authors
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// This file contains only forward declarations and Python bindings

#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <string>

//#define DEBUG
#include "include/data_spec.hpp"
#include "octree.h"
#include "rt_kernel.h"

namespace py = pybind11;
using torch::Tensor;

template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
void declare_octree(py::module &m, const std::string &typestr) {
//    using TOctree = Octree<scalar_t, branching, data_dim>;
//    std::string pyclass_name = std::string("Octree") + typestr;
//    py::class_<TOctree>(m, pyclass_name.c_str())
//    torch::python::bind_module<TOctree>(m, pyclass_name.c_str())
//        .def(py::init<int32_t, bool, torch::Device, torch::optional<torch::Tensor>, torch::optional<torch::Tensor>, int32_t>())
//        .def_readonly("n_internal", &TOctree::n_internal)
//        .def_readonly("max_depth", &TOctree::max_depth)
//        .def_readonly("parent_sum", &TOctree::parent_sum)
//        .def_readonly("node_size", &TOctree::node_size)
//        .def_readwrite("data", &TOctree::data)
//        .def_readwrite("child", &TOctree::child)
//        .def_readwrite("is_child_leaf", &TOctree::is_child_leaf)
//        .def_readwrite("parent", &TOctree::parent)
//        .def_readwrite("depth", &TOctree::depth)
//        .def("refine", &TOctree::refine_octree)
//        .def("query", &TOctree::query_octree)
//        .def("set", &TOctree::set_octree)
//        .def("query_interp", &TOctree::query_interp_octree);
//        .def("train", &TOctree::train)
//        .def("eval", &TOctree::eval);

    std::string fn_name = std::string("volume_render") + typestr;
    m.def(fn_name.c_str(), &volume_render<scalar_t, branching, data_dim, out_data_dim>);

    fn_name = std::string("volume_render_bwd") + typestr;
    m.def(fn_name.c_str(), &volume_render_bwd<scalar_t, branching, data_dim, out_data_dim>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    declare_octree<double, 2, 4, 3>(m, "d2d0");
    declare_octree<double, 2, 13, 3>(m, "d2d1");
    declare_octree<float, 2, 13, 3>(m, "f2d1");
    declare_octree<float, 4, 13, 3>(m, "f4d1");
    declare_octree<float, 2, 28, 3>(m, "f2d2");
    declare_octree<float, 4, 28, 3>(m, "f4d2");

    py::class_<RenderingOutput>(m, "RenderOutput")
        .def(py::init<>())
        .def_readonly("output_rgb", &RenderingOutput::output_rgb)
        .def_readonly("interpolated_vals", &RenderingOutput::interpolated_vals)
        .def_readonly("interpolated_n_ids", &RenderingOutput::interpolated_n_ids)
        .def_readonly("interpolation_weights", &RenderingOutput::interpolation_weights)
        .def_readonly("ray_offsets", &RenderingOutput::ray_offsets)
        .def_readonly("ray_steps", &RenderingOutput::ray_steps);

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<>())
        .def_readwrite("step_size", &RenderOptions::step_size)
        .def_readwrite("background_brightness", &RenderOptions::background_brightness)
        .def_readwrite("max_samples_per_node", &RenderOptions::max_samples_per_node)
        .def_readwrite("ndc_width", &RenderOptions::ndc_width)
        .def_readwrite("ndc_height", &RenderOptions::ndc_height)
        .def_readwrite("ndc_focal", &RenderOptions::ndc_focal)
        .def_readwrite("format", &RenderOptions::format)
        .def_readwrite("basis_dim", &RenderOptions::basis_dim)
        .def_readwrite("min_comp", &RenderOptions::min_comp)
        .def_readwrite("max_comp", &RenderOptions::max_comp)
        .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
        .def_readwrite("stop_thresh", &RenderOptions::stop_thresh)
        .def_readwrite("density_softplus", &RenderOptions::density_softplus)
        .def_readwrite("rgb_padding", &RenderOptions::rgb_padding);

    py::class_<OctreeCppSpec<float>>(m, "OctreeCppSpecf")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool>())
        .def_readwrite("data", &OctreeCppSpec<float>::data)
        .def_readwrite("child", &OctreeCppSpec<float>::child)
        .def_readwrite("is_child_leaf", &OctreeCppSpec<float>::is_child_leaf)
        .def_readwrite("parent", &OctreeCppSpec<float>::parent)
        .def_readwrite("depth", &OctreeCppSpec<float>::depth)
        .def_readwrite("scaling", &OctreeCppSpec<float>::scaling)
        .def_readwrite("offset", &OctreeCppSpec<float>::offset)
        .def_readonly("parent_sum", &OctreeCppSpec<float>::parent_sum);
    py::class_<OctreeCppSpec<double>>(m, "OctreeCppSpecd")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool>())
        .def_readwrite("data", &OctreeCppSpec<double>::data)
        .def_readwrite("child", &OctreeCppSpec<double>::child)
        .def_readwrite("is_child_leaf", &OctreeCppSpec<double>::is_child_leaf)
        .def_readwrite("parent", &OctreeCppSpec<double>::parent)
        .def_readwrite("depth", &OctreeCppSpec<double>::depth)
        .def_readwrite("scaling", &OctreeCppSpec<double>::scaling)
        .def_readwrite("offset", &OctreeCppSpec<double>::offset)
        .def_readonly("parent_sum", &OctreeCppSpec<double>::parent_sum);
}
