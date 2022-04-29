#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>

#include "ray_sampling.h"
#include "include/data_spec.hpp"
#include "cuda_common.h"
#include "octree_common.h"
#include "sh.h"


template <typename T, size_t N>
using Acc32 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int>;
template <typename T, size_t N>
using Acc64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;


template <typename scalar_t>
__device__ __inline__ scalar_t density_fwd(scalar_t sigma, const bool density_softplus) {
    return density_softplus ? _SOFTPLUS_M1(sigma) : sigma;
}

template <typename scalar_t>
__device__ __inline__ scalar_t density_bwd(scalar_t sigma, const bool density_softplus) {
    return density_softplus ? _SIGMOID(sigma - 1) : 1;
}

template <typename scalar_t>
__device__ __inline__ scalar_t rgb_fwd(scalar_t rgb_component, const float rgb_padding) {
    return _SIGMOID(rgb_component) * (scalar_t)(1 + 2 * rgb_padding) - (scalar_t)rgb_padding;
}

template <typename scalar_t>
__device__ __inline__ scalar_t rgb_bwd(scalar_t rgb_component, const float rgb_padding) {
    const scalar_t sigmoid = _SIGMOID(rgb_component);
    return sigmoid * (1.0 - sigmoid) * (1 + 2 * rgb_padding);
}

constexpr int basis_dim(int data_dim, int out_data_dim) {
    return (data_dim - 1) / out_data_dim;
}

template <typename scalar_t, int data_dim, int out_data_dim>
__device__ __inline__ void fwd_loop(const scalar_t * __restrict__ interp,
                                    const scalar_t * __restrict__ basis_fn,
                                    const bool density_softplus,
                                    const scalar_t sigma_thresh,
                                    const float dt,
                                    const float rgb_padding,
                                    scalar_t & __restrict__ light_intensity,
                                    scalar_t * __restrict__ out)
{
    const int bd = basis_dim(data_dim, out_data_dim);
    const scalar_t sigma = density_fwd<scalar_t>(interp[data_dim - 1], density_softplus);

    if (sigma > sigma_thresh) {
        const scalar_t att = expf(-dt * sigma);  // (1 - alpha)
        const scalar_t weight = light_intensity * (1.f - att);

        for (int j = 0, off = 0; j < out_data_dim; ++j, off += bd) {
            scalar_t tmp = 0.0;
            for (int k = 0; k < bd; ++k) {
                tmp += basis_fn[k] * interp[off + k];
            }
            out[j] += weight * rgb_fwd<scalar_t>(tmp, rgb_padding);
        }
        light_intensity *= att;
    }
}

template <typename scalar_t, int data_dim, int out_data_dim>
__device__ __inline__ void bwd_loop_p1(const scalar_t * __restrict__ interp,
                                       const scalar_t * __restrict__ basis_fn,
                                       const scalar_t * __restrict__ grad_output,
                                       const bool density_softplus,
                                       const scalar_t sigma_thresh,
                                       const float dt,
                                       const float rgb_padding,
                                       scalar_t & __restrict__ light_intensity,
                                       scalar_t & __restrict__ accum)
{
    const int bd = basis_dim(data_dim, out_data_dim);
    const scalar_t sigma = density_fwd<scalar_t>(interp[data_dim - 1], density_softplus);
    if (sigma > sigma_thresh) {
        const scalar_t att = expf(-dt * sigma);
        const scalar_t weight = light_intensity * (1.f - att);
        scalar_t total_color = 0.f;
        for (int j = 0, off = 0; j < out_data_dim; ++j, off += bd) {
            scalar_t tmp = 0.0;
            for (int k = 0; k < bd; ++k) {
                tmp += basis_fn[k] * interp[off + k];
            }
            total_color += rgb_fwd<scalar_t>(tmp, rgb_padding) * grad_output[j];
        }
        light_intensity *= att;
        accum += weight * total_color;
    }
}


template <typename scalar_t, int branching, int data_dim>
__global__ void fetch_interpolate(
    const Acc32<scalar_t, 2> t_data,
    const Acc32<int, 4> t_child,
    const Acc32<int, 5> t_nids,
    const Acc32<float, 3> ray_pos,
          Acc32<float, 3> interp_weights,
          Acc32<scalar_t, 3> interp_vals,
          Acc32<int, 2> nid_ptrs,
    const Acc32<int, 1> n_steps,
    const int n_batches)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_batches) return;

    const int n_intrs = n_steps[b];
    const int * const nid_start_ptr = &t_nids[0][0][0][0][0];
    float3 pos;
    scalar_t interp_val[data_dim];

    for (int i = 0; i < n_intrs; i++) {
        pos = make_float3(ray_pos[b][i][0], ray_pos[b][i][1], ray_pos[b][i][2]);
        float * const c_interp_weights = &interp_weights[b][i][0];
        _dev_query_corners<scalar_t, branching>(
                t_child, t_nids, pos,
                /*weights=*/c_interp_weights, /*nid_ptr=*/&nid_ptrs[b][i]);
        const int * n_ptr = nid_start_ptr + nid_ptrs[b][i];
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = c_interp_weights[0], t_data[*n_ptr][k];
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[1], t_data[*(n_ptr + 1)][k], interp_val[k]);
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[2], t_data[*(n_ptr + 2)][k], interp_val[k]);
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[3], t_data[*(n_ptr + 3)][k], interp_val[k]);
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[4], t_data[*(n_ptr + 4)][k], interp_val[k]);
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[5], t_data[*(n_ptr + 5)][k], interp_val[k]);
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[6], t_data[*(n_ptr + 6)][k], interp_val[k]);
        }
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[7], t_data[*(n_ptr + 7)][k], interp_val[k]);
            interp_vals[b][i][k] = interp_val[k];
        }
    }
}


template <typename scalar_t, int branching, int data_dim, int out_data_dim>
__global__ void trace_ray(
    const Acc32<float, 2> ray_steps,
    const Acc32<int, 1> n_steps,
    const Acc32<float, 2> rays_d_norm,
    const Acc32<scalar_t, 3> interp_vals,
    const int n_elements,
    const float rgb_padding,
    const scalar_t background_brightness,
    const bool density_softplus,
    const scalar_t sigma_thresh,
    const scalar_t stop_thresh,
          Acc32<scalar_t, 2> out)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_elements) return;
    const int bd = basis_dim(data_dim, out_data_dim);
    const int n_intrs = n_steps[b];
    const float3 ray_d = make_float3(rays_d_norm[b][0], rays_d_norm[b][1], rays_d_norm[b][2]);
    scalar_t basis_fn[bd];
    calc_sh_basis<scalar_t, bd>(ray_d, basis_fn);

    scalar_t light_intensity = 1.f;
    for (int i = 0; i < n_intrs; i++) {
        const float delta_t = ray_steps[b][i];
        if (delta_t <= 0) break;  // TODO: Unnecessary
        fwd_loop<scalar_t, data_dim, out_data_dim>(&interp_vals[b][i][0], basis_fn, density_softplus, sigma_thresh,
                                                   delta_t, rgb_padding, light_intensity, &out[b][0]);
        if (light_intensity <= stop_thresh) { return; }  // Full opacity, stop
    }
    for (int j = 0; j < out_data_dim; ++j) { out[b][j] += light_intensity * background_brightness; }
}  // trace ray



template <typename scalar_t, int branching, int data_dim, int out_data_dim>
__global__ void trace_ray_backward(
    const Acc32<int, 5> t_nids,
    const Acc32<float, 2> ray_steps,
    const Acc32<float, 3> interp_weights,
    const Acc32<scalar_t, 3> interp_vals,
    const Acc32<int, 2> nid_ptrs,
    const Acc32<int, 1> n_steps,
    const Acc32<scalar_t, 2> grad_output,
          Acc32<scalar_t, 2> grad_data_out,
    const Acc32<float, 2> rays_d_norm,
    const int n_elements,
    const float rgb_padding,
    const scalar_t background_brightness,
    const bool density_softplus,
    const scalar_t sigma_thresh,
    const scalar_t stop_thresh)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_elements) return;
    const int bd = basis_dim(data_dim, out_data_dim);
    const int n_intrs = n_steps[b];
    const int * const t_nids_start = &t_nids[0][0][0][0][0];
    const float3 ray_d = make_float3(rays_d_norm[b][0], rays_d_norm[b][1], rays_d_norm[b][2]);
    scalar_t grad_tree_val[data_dim];
    scalar_t basis_fn[bd];
    calc_sh_basis<scalar_t, bd>(ray_d, basis_fn);


    scalar_t accum = 0.0;
    // PASS 1: Just to compute the accum variable. This could be merged with the fwd pass (if we knew grad_output)
    scalar_t light_intensity = 1.f;
    for (int i = 0; i < n_intrs; i++) {
        float delta_t = ray_steps[b][i];
        if (delta_t <= 0) break;

        bwd_loop_p1<scalar_t, data_dim, out_data_dim>(
            &interp_vals[b][i][0], basis_fn, &grad_output[b][0], density_softplus,
            sigma_thresh, delta_t, rgb_padding, light_intensity, accum);
        if (light_intensity <= stop_thresh) {
            light_intensity = 0;
            break;
        }
    }
    scalar_t total_grad = 0.f;
    for (int j = 0; j < out_data_dim; ++j) { total_grad += grad_output[b][j]; }
    accum += light_intensity * background_brightness * total_grad;
    // PASS 2: Actually compute the gradient
    light_intensity = 1.f;
    for (int i = 0; i < n_intrs; i++) {
        float delta_t = ray_steps[b][i];
        if (delta_t <= 0) break;
        // Zero-out gradient
        for (int j = 0; j < data_dim; ++j) { grad_tree_val[j] = 0; }

        const scalar_t raw_sigma = interp_vals[b][i][data_dim - 1];
        const scalar_t sigma = density_fwd<scalar_t>(raw_sigma, density_softplus);
        if (sigma > sigma_thresh) {
            const scalar_t att = expf(-delta_t * sigma);
            const scalar_t weight = light_intensity * (1.f - att);

            scalar_t total_color = 0.f;
            for (int j = 0, off = 0; j < out_data_dim; ++j, off += bd) {
                scalar_t tmp = 0.0;
                for (int k = 0; k < bd; ++k) {
                    tmp += basis_fn[k] * interp_vals[b][i][off + k];
                }
                total_color += rgb_fwd<scalar_t>(tmp, rgb_padding) * grad_output[b][j];

                const scalar_t tmp2 = rgb_bwd<scalar_t>(tmp, rgb_padding) * weight * grad_output[b][j];
                for (int k = 0; k < bd; ++k) {
                    grad_tree_val[off + k] += basis_fn[k] * tmp2;
                }
            }
            light_intensity *= att;
            accum -= weight * total_color;
            grad_tree_val[data_dim - 1] = delta_t * (total_color * light_intensity - accum) * density_bwd<scalar_t>(raw_sigma, density_softplus);

            const int * n_ptr = t_nids_start + nid_ptrs[b][i];
            _dev_query_corners_bwd<scalar_t, data_dim>(
                n_ptr, &interp_weights[b][i][0], grad_tree_val, grad_data_out);

            if (light_intensity <= stop_thresh) {
                break;
            }
        }
    }
}

#define block_size_2d 32


template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
RenderingOutput corner_tree_render(
    const torch::Tensor & data,
    const torch::Tensor & t_child,
    const torch::Tensor & t_nids,
    const torch::Tensor & t_offset,
    const torch::Tensor & t_scaling,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const RenderOptions & opt)
{
    DEVICE_GUARD(data);
    const uint32_t batch_size = rays_o.size(0);

    const uint32_t gen_samples_n_threads = 128;
    const uint32_t render_ray_n_threads = 128;

    auto gs_start = at::cuda::CUDAEvent(cudaEventDefault);
    auto gs_end = at::cuda::CUDAEvent(cudaEventDefault);
    auto alloc_start = at::cuda::CUDAEvent(cudaEventDefault);
    auto alloc_end = at::cuda::CUDAEvent(cudaEventDefault);
    auto interpolate_start = at::cuda::CUDAEvent(cudaEventDefault);
    auto interpolate_end = at::cuda::CUDAEvent(cudaEventDefault);
    auto fwd_start = at::cuda::CUDAEvent(cudaEventDefault);
    auto fwd_end = at::cuda::CUDAEvent(cudaEventDefault);

    // 1. Generate samples
    gs_start.record();
    auto ray_steps = torch::full({batch_size, opt.max_intersections}, -1.0,
            torch::dtype(torch::kFloat32).device(data.device()));
    auto num_intersections = torch::zeros({batch_size}, torch::dtype(torch::kInt32).device(data.device()));
    auto positions = torch::empty({batch_size, opt.max_intersections, 3},
            torch::dtype(torch::kFloat32).device(data.device()));
    auto rays_d_norm = torch::empty_like(rays_d);

    gen_samples_kernel<scalar_t, branching>
        <<<n_blocks_linear<uint32_t>(batch_size, gen_samples_n_threads), gen_samples_n_threads>>>(
            t_child.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
            t_offset.data_ptr<float>(),
            t_scaling.data_ptr<float>(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            positions.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rays_d_norm.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            opt.max_intersections,
            opt.max_samples_per_node,
            (int)batch_size  // TODO: It would be nice without this cast.
    );
    #ifdef DEBUG
        printf("Gen-samples kernel complete. First few intersections follow\n");
        for (int i = 0; i < 5; i++) {
            printf("t=%f, dt=%f\n", ray_offsets[0][i].item<float>(), ray_steps[0][i].item<float>());
        }
        printf("\n");
    #endif
    gs_end.record();

    // 2. Forward pass (allocate tensors)
    alloc_start.record();
    torch::Tensor output = torch::zeros({batch_size, out_data_dim}, data.options());
    torch::Tensor interp_vals = torch::zeros({batch_size, opt.max_intersections, data_dim}, data.options());
    torch::Tensor interp_nid_ptrs = torch::zeros({batch_size, opt.max_intersections}, torch::dtype(torch::kInt32).device(data.device()));
    torch::Tensor interp_weights = torch::empty({batch_size, opt.max_intersections, 8},
        torch::dtype(torch::kFloat32).device(data.device()).layout(data.layout()));
    alloc_end.record();

    // 3. Interpolate at each valid intersction
    interpolate_start.record();
    fetch_interpolate<scalar_t, branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, 128), 128>>>(
            data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            t_child.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
            t_nids.packed_accessor32<int, 5, torch::RestrictPtrTraits>(),
            positions.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            interp_nid_ptrs.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            (int)batch_size);
    interpolate_end.record();

    // 3. Forward pass (compute)
    fwd_start.record();
    trace_ray<scalar_t, branching, data_dim, out_data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            rays_d_norm.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            (int)batch_size,
            opt.rgb_padding,
            (scalar_t)opt.background_brightness,
            opt.density_softplus,
            (scalar_t)opt.sigma_thresh,
            (scalar_t)opt.stop_thresh,
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    fwd_end.record();

    gs_start.synchronize();
    gs_end.synchronize();
    float gs_ela = gs_start.elapsed_time(gs_end);
    alloc_start.synchronize();
    alloc_end.synchronize();
    float alloc_ela = alloc_start.elapsed_time(alloc_end);
    interpolate_start.synchronize();
    interpolate_end.synchronize();
    float interpolate_ela = interpolate_start.elapsed_time(interpolate_end);
    fwd_start.synchronize();
    fwd_end.synchronize();
    float fwd_ela = fwd_start.elapsed_time(fwd_end);

    printf("Forward timings(ms): Sampling=%f  Alloc=%f  Interpolate=%f  Forward=%f\n", gs_ela, alloc_ela, interpolate_ela, fwd_ela);

    return {
        /*output_rgb=*/output,
        /*ray_steps=*/ray_steps,
        /*rays_d_norm=*/rays_d_norm,
        /*intersection_pos=*/positions,
        /*intersection_num=*/num_intersections,
        /*interpolated_vals=*/interp_vals,
        /*interpolated_n_ids=*/interp_nid_ptrs,
        /*interpolation_weights=*/interp_weights,
    };
}


template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
torch::Tensor corner_tree_render_bwd(
    const torch::Tensor & t_data,
    const torch::Tensor & t_nids,
    const torch::Tensor & rays_d_norm,
    const torch::Tensor & num_intersections,
    const torch::Tensor & grad_output,
    const torch::Tensor & interp_vals,
    const torch::Tensor & interp_nid_ptrs,
    const torch::Tensor & interp_weights,
    const torch::Tensor & ray_steps,
    const RenderOptions & opt)
{
    DEVICE_GUARD(t_data);
    const uint32_t batch_size = rays_d_norm.size(0);
    const uint32_t render_ray_n_threads = 128;

    torch::Tensor output = torch::zeros_like(t_data);

    trace_ray_backward<scalar_t, branching, data_dim, out_data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
        t_nids.packed_accessor32<int, 5, torch::RestrictPtrTraits>(),
        ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        interp_nid_ptrs.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        rays_d_norm.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        (int)batch_size,
        opt.rgb_padding,
        (scalar_t)opt.background_brightness,
        opt.density_softplus,
        (scalar_t)opt.sigma_thresh,
        (scalar_t)opt.stop_thresh
    );
    return output;
}
