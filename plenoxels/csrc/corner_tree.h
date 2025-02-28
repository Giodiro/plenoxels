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


__device__ __inline__ float density_fwd(float sigma, const bool density_softplus) {
    return density_softplus ? _SOFTPLUS_M1(sigma) : sigma;
}

__device__ __inline__ float density_bwd(float sigma, const bool density_softplus) {
    return density_softplus ? _SIGMOID(sigma - 1) : 1;
}


template <int bd>
__device__ __inline__ float3 apply_sh(const float * const __restrict__ basis_fn,
                                      const float * const __restrict__ values)
{
    float3 out = make_float3(0.0, 0.0, 0.0);
    int k = 0, bk;
    for (bk = 0; bk < bd; k++, bk++) {
        out.x = fmaf(basis_fn[bk], values[k], out.x);
    }
    for (bk = 0; bk < bd; k++, bk++) {
        out.y = fmaf(basis_fn[bk], values[k], out.y);
    }
    for (bk = 0; bk < bd; k++, bk++) {
        out.z = fmaf(basis_fn[bk], values[k], out.z);
    }
    return out;
}

template <int bd>
__device__ __inline__ void apply_sh_bwd(const float  * const __restrict__ basis_fn,
                                        const float3 &       __restrict__ grad_output,
                                              float  * const __restrict__ out)
{
    int k = 0, bk;
    for (bk = 0; bk < bd; k++, bk++) {
        out[k] = basis_fn[bk] * grad_output.x;
    }
    for (bk = 0; bk < bd; k++, bk++) {
        out[k] = basis_fn[bk] * grad_output.y;
    }
    for (bk = 0; bk < bd; k++, bk++) {
        out[k] = basis_fn[bk] * grad_output.z;
    }
}


__device__ __inline__ float3 sh_to_rgb(const float3 & __restrict__ sh_output,
                                       const float rgb_padding)
{
    return make_float3(
        _SIGMOID(sh_output.x) * (1 + 2 * rgb_padding) - rgb_padding,
        _SIGMOID(sh_output.y) * (1 + 2 * rgb_padding) - rgb_padding,
        _SIGMOID(sh_output.z) * (1 + 2 * rgb_padding) - rgb_padding
    );
}

__device__ __inline__ float3 sh_to_rgb_backward(const float3 & __restrict__ sh_output,
                                                const float rgb_padding)
{
    float3 out = make_float3(0.0, 0.0, 0.0);
    float sigmoid = _SIGMOID(sh_output.x);
    out.x = sigmoid * (1.0 - sigmoid) * (1 + 2 * rgb_padding);
    sigmoid = _SIGMOID(sh_output.y);
    out.y = sigmoid * (1.0 - sigmoid) * (1 + 2 * rgb_padding);
    sigmoid = _SIGMOID(sh_output.z);
    out.z = sigmoid * (1.0 - sigmoid) * (1 + 2 * rgb_padding);
    return out;
}


constexpr int basis_dim(int data_dim, int out_data_dim) {
    return (data_dim - 1) / out_data_dim;
}


template <int data_dim>
__device__ __inline__ void fwd_loop(const float * __restrict__ interp,
                                    const float * __restrict__ basis_fn,
                                    const bool density_softplus,
                                    const float sigma_thresh,
                                    const float dt,
                                    const float rgb_padding,
                                    float & __restrict__ light_intensity,
                                    float * __restrict__ out)
{
    const int bd = basis_dim(data_dim, 3);
    const float sigma = density_fwd(interp[data_dim - 1], density_softplus);
    if (sigma > sigma_thresh) {
        const float att = __expf(-dt * sigma);  // (1 - alpha)
        const float weight = light_intensity * (1.f - att);
        const float3 sh_out = apply_sh<bd>(basis_fn, interp);
        const float3 rgb = sh_to_rgb(sh_out, rgb_padding);
        out[0] += weight * rgb.x;
        out[1] += weight * rgb.y;
        out[2] += weight * rgb.z;
        light_intensity *= att;
    }
}

template <int data_dim>
__device__ __inline__ void bwd_loop_p1(const float * __restrict__ interp,
                                       const float * __restrict__ basis_fn,
                                       const float * __restrict__ grad_output,
                                       const bool density_softplus,
                                       const float sigma_thresh,
                                       const float dt,
                                       const float rgb_padding,
                                       float & __restrict__ light_intensity,
                                       float & __restrict__ accum)
{
    const int bd = basis_dim(data_dim, 3);
    const float sigma = density_fwd(interp[data_dim - 1], density_softplus);
    if (sigma > sigma_thresh) {
        const float att = __expf(-dt * sigma);
        const float weight = light_intensity * (1.f - att);
        const float3 sh_out = apply_sh<bd>(basis_fn, interp);
        const float3 rgb = sh_to_rgb(sh_out, rgb_padding);
        const float total_color = rgb.x * grad_output[0] + rgb.y * grad_output[1] + rgb.z * grad_output[2];
        light_intensity *= att;
        accum = fmaf(weight, total_color, accum);
    }
}


template <int branching, int data_dim>
__global__ void fetch_interpolate(
    const Acc32<float, 2> t_data,
    const Acc32<int, 4> t_child,
    const Acc32<int, 5> t_nids,
    const Acc32<float, 3> ray_pos,
          Acc32<float, 3> interp_weights,
          Acc32<float, 3> interp_vals,
          Acc32<int, 2> nid_ptrs,
    const Acc32<int, 1> n_steps,
    const int n_batches)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_batches) return;

    const int n_intrs = n_steps[b];
    const int * const nid_start_ptr = &t_nids[0][0][0][0][0];
    int nid_ptr_offset;

    float3 pos;
    float interp_val[data_dim];

    for (int i = 0; i < n_intrs; i++) {
        pos = make_float3(ray_pos[b][i][0], ray_pos[b][i][1], ray_pos[b][i][2]);
        float * const c_interp_weights = &interp_weights[b][i][0];
        _dev_query_corners<branching>(
                t_child, t_nids, pos,
                /*weights=*/c_interp_weights, /*nid_ptr=*/&nid_ptr_offset);//&nid_ptrs[b][i]);
        const int * n_ptr = nid_start_ptr + nid_ptr_offset;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = c_interp_weights[0] * t_data[*n_ptr][k];
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[1], t_data[*n_ptr][k], interp_val[k]);
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[2], t_data[*n_ptr][k], interp_val[k]);
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[3], t_data[*n_ptr][k], interp_val[k]);
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[4], t_data[*n_ptr][k], interp_val[k]);
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[5], t_data[*n_ptr][k], interp_val[k]);
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[6], t_data[*n_ptr][k], interp_val[k]);
        }
        n_ptr += 1;
        #pragma unroll data_dim
        for (int k = 0; k < data_dim; k++) {
            interp_val[k] = fmaf(c_interp_weights[7], t_data[*n_ptr][k], interp_val[k]);
            // Write final result to global memory
            interp_vals[b][i][k] = interp_val[k];
        }
        // Write neighbor-ID pointer offset to global memory
        nid_ptrs[b][i] = nid_ptr_offset;
    }
}


template <int branching, int data_dim>
__global__ void ray_loss_kernel(
    const Acc32<int, 5> t_nids,
    const Acc32<float, 2> ray_steps,
    const Acc32<float, 3> interp_weights,
    const Acc32<float, 3> interp_vals,
    const Acc32<int, 2> nid_ptrs,
    const Acc32<int, 1> n_steps,
    const Acc32<float, 2> rays_d_norm,
    const Acc32<float, 2> targets,
    float * __restrict__ loss_output,
    const int n_elements,
    const float rgb_padding,
    const float background_brightness,
    const bool density_softplus,
    const float sigma_thresh,
    const float stop_thresh,
          Acc32<float, 2> out,
          Acc32<float, 2> grad_data_out)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_elements) return;
    const int bd = basis_dim(data_dim, 3);
    const int * const t_nids_start = &t_nids[0][0][0][0][0];
    const int n_intrs = n_steps[b];
    const float3 ray_d = make_float3(rays_d_norm[b][0], rays_d_norm[b][1], rays_d_norm[b][2]);
    float basis_fn[bd];
    calc_sh_basis<bd>(ray_d, basis_fn);

    float * rgb_ray = &out[b][0];

    // Forward
    float light_intensity = 1.f;
    int i = 0;
    for (; i < n_intrs; i++) {
        const float delta_t = ray_steps[b][i];
        const float sigma = density_fwd(interp_vals[b][i][data_dim - 1], density_softplus);
        const float alpha = 1.f - __expf(-sigma * delta_t);
        const float weight = alpha * light_intensity;
        const float3 rgb = sh_to_rgb(apply_sh<bd>(basis_fn, &interp_vals[b][i][0]), rgb_padding);
        rgb_ray[0] += weight * rgb.x;
        rgb_ray[1] += weight * rgb.y;
        rgb_ray[2] += weight * rgb.z;

        light_intensity *= 1.f - alpha;
        if (light_intensity <= stop_thresh) { break; }
    }
    if (i == n_intrs) {
        for (int j = 0; j < 3; ++j) { rgb_ray[j] += light_intensity * background_brightness; }
    }

    // Loss & Gradient of the loss
    const float loss_scale = 1 / (float)n_elements;
    const float3 diff = make_float3(rgb_ray[0] - targets[b][0], rgb_ray[1] - targets[b][1], rgb_ray[2] - targets[b][2]);
    const float3 loss = make_float3(diff.x * diff.x, diff.y * diff.y, diff.z * diff.z);
    const float3 grad = make_float3(2.0f * diff.x, 2.0f * diff.y, 2.0f * diff.z);
    if (loss_output) {
        loss_output[b] = (loss.x + loss.y + loss.z) / (3 * (float)n_elements);
    }

    // Backward
    float3 rgb_ray2 = make_float3(0., 0., 0.);
    light_intensity = 1.f;
    for (int j = 0; j < i; j++) {
        const float delta_t = ray_steps[b][j];
        const float raw_sigma = interp_vals[b][j][data_dim - 1];
        const float sigma = density_fwd(raw_sigma, density_softplus);
        const float alpha = 1.f - __expf(-sigma * delta_t);
        const float weight = alpha * light_intensity;
        const float3 sh_out = apply_sh<bd>(basis_fn, &interp_vals[b][j][0]);
        const float3 rgb = sh_to_rgb(sh_out, rgb_padding);
		rgb_ray2 += weight * rgb;
		light_intensity *= 1.f - alpha;

		float grad_tree_val[data_dim];

        // Gradient wrt RGB inputs
        const float3 dl_drgb = loss_scale * weight * grad * sh_to_rgb_backward(sh_out, rgb_padding);
		apply_sh_bwd<bd>(basis_fn, dl_drgb, grad_tree_val);
        // Gradient wrt Sigma inputs
		const float3 suffix = make_float3(rgb_ray[0] - rgb_ray2.x, rgb_ray[1] - rgb_ray2.y, rgb_ray[2] - rgb_ray2.z);

        const float dl_dsigma = loss_scale * (
            density_bwd(raw_sigma, density_softplus) * delta_t * dot(grad, light_intensity * rgb - suffix)
        );
        grad_tree_val[data_dim - 1] = dl_dsigma;

        const int * n_ptr = t_nids_start + nid_ptrs[b][j];
        _dev_query_corners_bwd<data_dim>(
            n_ptr, &interp_weights[b][j][0], grad_tree_val, grad_data_out);
    }
}



template <int branching, int data_dim>
__global__ void trace_ray(
    const Acc32<float, 2> ray_steps,
    const Acc32<int, 1> n_steps,
    const Acc32<float, 2> rays_d_norm,
    const Acc32<float, 3> interp_vals,
    const int n_elements,
    const float rgb_padding,
    const float background_brightness,
    const bool density_softplus,
    const float sigma_thresh,
    const float stop_thresh,
          Acc32<float, 2> out)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_elements) return;
    const int bd = basis_dim(data_dim, 3);
    const int n_intrs = n_steps[b];
    const float3 ray_d = make_float3(rays_d_norm[b][0], rays_d_norm[b][1], rays_d_norm[b][2]);
    float basis_fn[bd];
    calc_sh_basis<bd>(ray_d, basis_fn);

    float light_intensity = 1.f;
    for (int i = 0; i < n_intrs; i++) {
        const float delta_t = ray_steps[b][i];
        if (delta_t <= 0) break;
        fwd_loop<data_dim>(&interp_vals[b][i][0], basis_fn, density_softplus, sigma_thresh,
                           delta_t, rgb_padding, light_intensity, &out[b][0]);
        if (light_intensity <= stop_thresh) { return; }  // Full opacity, stop
    }
    for (int j = 0; j < 3; ++j) { out[b][j] += light_intensity * background_brightness; }
}  // trace ray



template <int branching, int data_dim>
__global__ void trace_ray_backward(
    const Acc32<int, 5> t_nids,
    const Acc32<float, 2> ray_steps,
    const Acc32<float, 3> interp_weights,
    const Acc32<float, 3> interp_vals,
    const Acc32<int, 2> nid_ptrs,
    const Acc32<int, 1> n_steps,
    const Acc32<float, 2> grad_output,
          Acc32<float, 2> grad_data_out,
    const Acc32<float, 2> rays_d_norm,
    const int n_elements,
    const float rgb_padding,
    const float background_brightness,
    const bool density_softplus,
    const float sigma_thresh,
    const float stop_thresh)
{
    const int b = threadIdx.x + blockIdx.x * blockDim.x;  // element in batch
    if (b >= n_elements) return;
    const int bd = basis_dim(data_dim, 3);
    const int n_intrs = n_steps[b];
    const int * const t_nids_start = &t_nids[0][0][0][0][0];
    const float3 ray_d = make_float3(rays_d_norm[b][0], rays_d_norm[b][1], rays_d_norm[b][2]);
    float grad_tree_val[data_dim];
    float basis_fn[bd];
    calc_sh_basis<bd>(ray_d, basis_fn);

    float accum = 0.0;
    // PASS 1: Just to compute the accum variable. This could be merged with the fwd pass (if we knew grad_output)
    float light_intensity = 1.f;
    for (int i = 0; i < n_intrs; i++) {
        const float delta_t = ray_steps[b][i];
        // Essentially updates the accum variable
        bwd_loop_p1<data_dim>(
            &interp_vals[b][i][0], basis_fn, &grad_output[b][0], density_softplus,
            sigma_thresh, delta_t, rgb_padding, light_intensity, accum);
        if (light_intensity <= stop_thresh) {
            light_intensity = 0;
            break;
        }
    }
    float total_grad = 0.f;
    for (int j = 0; j < 3; ++j) { total_grad += grad_output[b][j]; }
    accum += light_intensity * background_brightness * total_grad;
    // PASS 2: Actually compute the gradient
    light_intensity = 1.f;
    for (int i = 0; i < n_intrs; i++) {
        float delta_t = ray_steps[b][i];
        // Zero-out gradient
        for (int j = 0; j < data_dim; ++j) { grad_tree_val[j] = 0; }

        const float raw_sigma = interp_vals[b][i][data_dim - 1];
        const float sigma = density_fwd(raw_sigma, density_softplus);
        if (sigma > sigma_thresh) {
            const float att = __expf(-delta_t * sigma);
            const float weight = light_intensity * (1.f - att);
            const float3 sh_out = apply_sh<bd>(basis_fn, &interp_vals[b][i][0]);
            const float3 rgb = sh_to_rgb(sh_out, rgb_padding);
            const float total_color = rgb.x * grad_output[b][0] + rgb.y * grad_output[b][1] + rgb.z * grad_output[b][2];
            light_intensity *= att;
            accum = fmaf(-weight, total_color, accum);

            // Gradient wrt RGB inputs
		    float3 rgb_bwd = sh_to_rgb_backward(sh_out, rgb_padding);
		    rgb_bwd.x = rgb_bwd.x * weight * grad_output[b][0];
		    rgb_bwd.y = rgb_bwd.y * weight * grad_output[b][1];
		    rgb_bwd.z = rgb_bwd.z * weight * grad_output[b][2];
		    apply_sh_bwd<bd>(basis_fn, rgb_bwd, grad_tree_val);
            // Gradient wrt Sigma inputs
            const float sigma_derivative = density_bwd(raw_sigma, density_softplus);
            grad_tree_val[data_dim - 1] = sigma_derivative * delta_t * (total_color * light_intensity - accum);

            const int * n_ptr = t_nids_start + nid_ptrs[b][i];
            _dev_query_corners_bwd<data_dim>(
                n_ptr, &interp_weights[b][i][0], grad_tree_val, grad_data_out);

            if (light_intensity <= stop_thresh) { break; }
        }
    }
}


template <int branching, int data_dim>
RenderingOutput corner_tree_loss_grad(
    torch::Tensor & data,
    const torch::Tensor & t_child,
    const torch::Tensor & t_nids,
    const torch::Tensor & t_offset,
    const torch::Tensor & t_scaling,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const torch::Tensor & targets,
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

    gen_samples_kernel<branching>
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
    gs_end.record();

    // 2. Forward pass (allocate tensors)
    alloc_start.record();
    torch::Tensor output = torch::zeros({batch_size, 3}, data.options());
    torch::Tensor interp_vals = torch::zeros({batch_size, opt.max_intersections, data_dim}, data.options());
    torch::Tensor interp_nid_ptrs = torch::zeros({batch_size, opt.max_intersections}, torch::dtype(torch::kInt32).device(data.device()));
    torch::Tensor interp_weights = torch::empty({batch_size, opt.max_intersections, 8},
        torch::dtype(torch::kFloat32).device(data.device()).layout(data.layout()));
    torch::Tensor data_grad = torch::zeros_like(data);
    torch::Tensor loss_output = torch::empty({batch_size}, data.options());
    alloc_end.record();

    // 3. Interpolate at each valid intersction
    interpolate_start.record();
    fetch_interpolate<branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, 128), 128>>>(
            data.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            t_child.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
            t_nids.packed_accessor32<int, 5, torch::RestrictPtrTraits>(),
            positions.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_nid_ptrs.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            (int)batch_size);
    interpolate_end.record();

    fwd_start.record();
    ray_loss_kernel<branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, 128), 128>>>(
            t_nids.packed_accessor32<int, 5, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_nid_ptrs.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            rays_d_norm.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            targets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            loss_output.data_ptr<float>(),
            (int)batch_size,
            opt.rgb_padding,
            opt.background_brightness,
            opt.density_softplus,
            opt.sigma_thresh,
            opt.stop_thresh,
            output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            data_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );
    fwd_end.record();

    data.mutable_grad() = data_grad;

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

    torch::Tensor loss = loss_output.sum();
    printf("Forward timings(ms): Sampling=%f  Alloc=%f  Interpolate=%f  Forward=%f -- Loss=%f\n", gs_ela, alloc_ela, interpolate_ela, fwd_ela, loss.item<float>());

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


template <int branching, int data_dim>
RenderingOutput gen_samples(
    const torch::Tensor & t_child,
    const torch::Tensor & t_offset,
    const torch::Tensor & t_scaling,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const RenderOptions & opt)
{
    DEVICE_GUARD(t_child);
    const uint32_t batch_size = rays_o.size(0);

    auto ray_steps = torch::full({batch_size, opt.max_intersections}, -1.0,
            torch::dtype(torch::kFloat32).device(t_child.device()));
    auto num_intersections = torch::zeros({batch_size}, torch::dtype(torch::kInt32).device(t_child.device()));
    auto positions = torch::empty({batch_size, opt.max_intersections, 3},
            torch::dtype(torch::kFloat32).device(t_child.device()));
    auto rays_d_norm = torch::empty_like(rays_d);

    gen_samples_kernel<branching>
        <<<n_blocks_linear<uint32_t>(batch_size, 128), 128>>>(
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
            (int)batch_size
    );
    torch::Tensor undefined;
    return {
        /*output_rgb=*/undefined,
        /*ray_steps=*/ray_steps,
        /*rays_d_norm=*/rays_d_norm,
        /*intersection_pos=*/positions,
        /*intersection_num=*/num_intersections,
        /*interpolated_vals=*/undefined,
        /*interpolated_n_ids=*/undefined,
        /*interpolation_weights=*/undefined
    };
}


template <int32_t branching, int32_t data_dim>
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

    gen_samples_kernel<branching>
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
    torch::Tensor output = torch::zeros({batch_size, 3}, data.options());
    torch::Tensor interp_vals = torch::zeros({batch_size, opt.max_intersections, data_dim}, data.options());
    torch::Tensor interp_nid_ptrs = torch::zeros({batch_size, opt.max_intersections}, torch::dtype(torch::kInt32).device(data.device()));
    torch::Tensor interp_weights = torch::empty({batch_size, opt.max_intersections, 8},
        torch::dtype(torch::kFloat32).device(data.device()).layout(data.layout()));
    alloc_end.record();

    // 3. Interpolate at each valid intersction
    interpolate_start.record();
    fetch_interpolate<branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, 128), 128>>>(
            data.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            t_child.packed_accessor32<int, 4, torch::RestrictPtrTraits>(),
            t_nids.packed_accessor32<int, 5, torch::RestrictPtrTraits>(),
            positions.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            interp_nid_ptrs.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            (int)batch_size);
    interpolate_end.record();

    // 3. Forward pass (compute)
    fwd_start.record();
    trace_ray<branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            rays_d_norm.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            (int)batch_size,
            opt.rgb_padding,
            opt.background_brightness,
            opt.density_softplus,
            opt.sigma_thresh,
            opt.stop_thresh,
            output.packed_accessor32<float, 2, torch::RestrictPtrTraits>());
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


template <int32_t branching, int32_t data_dim>
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

    trace_ray_backward<branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
        t_nids.packed_accessor32<int, 5, torch::RestrictPtrTraits>(),
        ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        interp_vals.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        interp_nid_ptrs.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        num_intersections.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        rays_d_norm.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        (int)batch_size,
        opt.rgb_padding,
        opt.background_brightness,
        opt.density_softplus,
        opt.sigma_thresh,
        opt.stop_thresh
    );
    return output;
}
