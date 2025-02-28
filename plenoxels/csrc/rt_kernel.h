#include <stdexcept>
#include <cstdint>
#include <vector>

#include "cuda_common.h"
#include "octree_common.h"
#include "sh.h"
#include "include/data_spec.hpp"
#include "ray_sampling.h"


using namespace torch::indexing;



template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
__device__ __inline__ void trace_ray_backward(
        const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> t_parent,
        const float* __restrict__ t_scaling,
        const torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> interp_vals,
        const torch::TensorAccessor<int64_t, 2, torch::RestrictPtrTraits, int32_t> interp_nids,
        const torch::TensorAccessor<float, 2, torch::RestrictPtrTraits, int32_t> interp_weights,

        const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> ray_offsets,
        const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> ray_steps,

        const torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> grad_output,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_data_out,
        const float3 & __restrict__ ray_o,
        float3 & __restrict__ ray_d,
        const float rgb_padding,
        const float background_brightness,
        const bool density_softplus,
        const float sigma_thresh,
        const float stop_thresh)
{
    constexpr int32_t basis_dim = (data_dim - 1) / out_data_dim;
    const float delta_scale = _get_delta_scale(t_scaling, ray_d);
    float tmin, tmax;
    scalar_t grad_tree_val[data_dim];
    scalar_t light_intensity;
    const scalar_t d_rgb_pad = 1 + 2 * rgb_padding;
    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    }

    scalar_t basis_fn[basis_dim];
    calc_sh_basis<scalar_t, basis_dim>(ray_d, basis_fn);

    scalar_t accum = 0.0;
    // PASS 1: Just to compute the accum variable. This could be merged with the fwd pass (if we knew grad_output)
    light_intensity = 1.f;
    for (int i = 0; i < ray_offsets.size(0); i++) {
        float t = ray_offsets[i];
        if (t < tmin) continue;
        if (t >= tmax) break;
        float delta_t = ray_steps[i] * delta_scale;

        scalar_t sigma = interp_vals[i][data_dim - 1];
        if (density_softplus) { sigma = _SOFTPLUS_M1(sigma); }
        if (sigma > sigma_thresh) {
            const scalar_t att = expf(-delta_t * sigma);
            const scalar_t weight = light_intensity * (1.f - att);

            scalar_t total_color = 0.f;
            for (int j = 0, off = 0; j < out_data_dim; ++j, off += basis_dim) {
                scalar_t tmp = 0.0;
                for (int k = 0; k < basis_dim; ++k) {
                    tmp += basis_fn[k] * interp_vals[i][off + k];
                }
                total_color += (_SIGMOID(tmp) * d_rgb_pad - rgb_padding) * grad_output[j];
            }
            light_intensity *= att;
            accum += weight * total_color;
            if (light_intensity <= stop_thresh) {
                light_intensity = 0;
                break;
            }
        }
    }
    scalar_t total_grad = 0.f;
    for (int j = 0; j < out_data_dim; ++j) { total_grad += grad_output[j]; }
    accum += light_intensity * background_brightness * total_grad;
    // PASS 2: Actually compute the gradient
    light_intensity = 1.f;
    for (int i = 0; i < ray_offsets.size(0); i++) {
        float t = ray_offsets[i];
        if (t < tmin) continue;
        if (t >= tmax) break;
        float delta_t = ray_steps[i] * delta_scale;
        // Zero-out gradient
        for (int j = 0; j < data_dim; ++j) { grad_tree_val[j] = 0; }

        scalar_t sigma = interp_vals[i][data_dim - 1];
        const scalar_t raw_sigma = sigma;  // needed for softplus-bwd
        if (density_softplus) { sigma = _SOFTPLUS_M1(sigma); }
        if (sigma > sigma_thresh) {
            const scalar_t att = expf(-delta_t * sigma);
            const scalar_t weight = light_intensity * (1.f - att);

            scalar_t total_color = 0.f;
            for (int j = 0, off = 0; j < out_data_dim; ++j, off += basis_dim) {
                scalar_t tmp = 0.0;
                for (int k = 0; k < basis_dim; ++k) {
                    tmp += basis_fn[k] * interp_vals[i][off + k];
                }
                const scalar_t sigmoid = _SIGMOID(tmp);
                const scalar_t tmp2 = weight * sigmoid * (1.0 - sigmoid) * grad_output[j] * d_rgb_pad;
                for (int k = 0; k < basis_dim; ++k) {
                    grad_tree_val[off + k] += basis_fn[k] * tmp2;
                }
                total_color += (sigmoid * d_rgb_pad - rgb_padding) * grad_output[j];
            }
            light_intensity *= att;
            accum -= weight * total_color;
            grad_tree_val[data_dim - 1] = delta_t * (total_color * light_intensity - accum)
                *  (density_softplus ? _SIGMOID(raw_sigma - 1) : 1);
            #ifdef DEBUG
                printf("t=%f - setting sigma gradient to %f\n", t, grad_tree_val[data_dim - 1]);
            #endif
            _dev_query_interp_bwd<scalar_t, data_dim>(
            //_dev_query_single_outv_bwd<scalar_t, data_dim>(
                /*parent=*/t_parent, /*grad=*/grad_data_out, /*weights=*/&interp_weights[i][0],
                /*neighbor_ids=*/&interp_nids[i][0], /*grad_output=*/grad_tree_val);

            if (light_intensity <= stop_thresh) {
                break;
            }
        }
    }
}  // trace_ray_backward


template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
__device__ __inline__ void trace_ray(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> t_data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> t_icf,  // is_child_leaf
    const bool t_parent_sum,
    const float* __restrict__ t_scaling,
    const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> ray_offsets,
    const torch::TensorAccessor<float, 1, torch::RestrictPtrTraits, int32_t> ray_steps,
    torch::TensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, int32_t> interp_vals,
    torch::TensorAccessor<int64_t, 2, torch::RestrictPtrTraits, int32_t> interp_nids,
    torch::TensorAccessor<float, 2, torch::RestrictPtrTraits, int32_t> interp_weights,
    torch::TensorAccessor<float, 2, torch::RestrictPtrTraits, int32_t> neighbor_coo,
    const float3 & __restrict__ ray_o,
    float3 & __restrict__ ray_d,
    const float rgb_padding,
    const float background_brightness,
    const bool density_softplus,
    const float sigma_thresh,
    const float stop_thresh,
    torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out)
{
    constexpr int32_t basis_dim = (data_dim - 1) / out_data_dim;
    const float delta_scale = _get_delta_scale(t_scaling, ray_d);
    float tmin, tmax;
    float3 pos;
    scalar_t basis_fn[basis_dim];

    const scalar_t d_rgb_pad = 1 + 2 * rgb_padding;

    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) { out[j] = background_brightness; }
        return;
    }

    for (int j = 0; j < out_data_dim; ++j) { out[j] = 0.0f; }
    calc_sh_basis<scalar_t, basis_dim>(ray_d, basis_fn);
    scalar_t light_intensity = 1.f;

    for (int i = 0; i < ray_offsets.size(0); i++) {
        float t = ray_offsets[i];
        if (t < tmin) continue;
        if (t >= tmax) break;
        float delta_t = ray_steps[i];
        pos = ray_o + t * ray_d;
        scalar_t *tree_val = &interp_vals[i][0];
        //_dev_query_single_outv<scalar_t, branching, data_dim>(
        _dev_query_interp<scalar_t, branching, data_dim>(
            t_data, t_child, t_icf, /*in_coo=*/pos, /*weights=*/&interp_weights[i][0],
            /*neighbor_coo=*/neighbor_coo, /*neighbor_ids=*/&interp_nids[i][0], /*out_val=*/&interp_vals[i][0],
            /*parent_sum=*/t_parent_sum);

        scalar_t sigma = interp_vals[i][data_dim - 1];
        if (density_softplus) { sigma = _SOFTPLUS_M1(sigma); }
        if (sigma > sigma_thresh) {
            const scalar_t att = expf(-delta_t * delta_scale * sigma);  // (1 - alpha)
            const scalar_t weight = light_intensity * (1.f - att);

            for (int j = 0, off = 0; j < out_data_dim; ++j, off += basis_dim) {
                scalar_t tmp = 0.0;
                for (int k = 0; k < basis_dim; ++k) {
                    tmp += basis_fn[k] * interp_vals[i][off + k];
                }
                out[j] += weight * (_SIGMOID(tmp) * d_rgb_pad - rgb_padding);
            }
            light_intensity *= att;
            if (light_intensity <= stop_thresh) {  // Full opacity, stop
                return;
            }
        }
    }
    for (int j = 0; j < out_data_dim; ++j) {
        out[j] += light_intensity * background_brightness;
    }
}  // trace ray


template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
__global__ void render_ray_bwd_kernel(
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> t_parent,
    const float* __restrict__ t_offset,
    const float* __restrict__ t_scaling,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_offsets,    // batch_size, n_intersections
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_steps,      // batch_size, n_intersections
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interp_vals,       // batch_size, n_intersections, data_dim
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> interp_nids,        // batch_size, n_intersections, 8
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> interp_weights,       // batch_size, n_intersections, 8
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,         // batch_size, 3
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,         // batch_size, 3
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output, // batch_size, data_dim
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_data_out,     // num_points, data_dim
    const float rgb_padding,
    const float background_brightness,
    const bool density_softplus,
    const float sigma_thresh,
    const float stop_thresh,
    const int32_t n_elements)
{
	const int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 ray_o = make_float3(rays_o[i][0], rays_o[i][1], rays_o[i][2]);
    float3 ray_d = make_float3(rays_d[i][0], rays_d[i][1], rays_d[i][2]);
    transform_coord(ray_o, t_offset, t_scaling);

	trace_ray_backward<scalar_t, branching, data_dim, out_data_dim>(
	    t_parent, t_scaling, interp_vals[i], interp_nids[i], interp_weights[i], ray_offsets[i], ray_steps[i],
        grad_output[i], grad_data_out, ray_o, ray_d, rgb_padding, background_brightness, density_softplus, sigma_thresh, stop_thresh);
}


template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
__global__ void render_ray_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> t_data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> t_icf,  // is_child_leaf
    const bool t_parent_sum,
    const float* __restrict__ t_offset,
    const float* __restrict__ t_scaling,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_offsets,  // batch_size, n_intersections
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_steps,    // batch_size, n_intersections
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interp_vals,     // batch_size, n_intersections, data_dim
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> interp_nids,      // batch_size, n_intersections, 8
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> interp_weights,     // batch_size, n_intersections, 8
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> neighbor_coo,       // batch_size, 8, 3
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,       // batch_size, 3
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,       // batch_size, 3
    const float rgb_padding,
    const float background_brightness,
    const bool density_softplus,
    const float sigma_thresh,
    const float stop_thresh,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,            // batch_size, data_dim
    const int32_t n_elements)
{
	const int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 ray_o = make_float3(rays_o[i][0], rays_o[i][1], rays_o[i][2]);
    float3 ray_d = make_float3(rays_d[i][0], rays_d[i][1], rays_d[i][2]);
    transform_coord(ray_o, t_offset, t_scaling);

	trace_ray<scalar_t, branching, data_dim, out_data_dim>(
	    t_data, t_child, t_icf, t_parent_sum, t_scaling, ray_offsets[i], ray_steps[i], interp_vals[i],
	    interp_nids[i], interp_weights[i], neighbor_coo[i], ray_o, ray_d, rgb_padding, background_brightness, density_softplus, sigma_thresh, stop_thresh, out[i]);
}



template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
RenderingOutput volume_render(
    torch::Tensor & data,
    OctreeCppSpec<scalar_t> & tree,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const RenderOptions & opt)
{
    DEVICE_GUARD(data);
    const uint32_t batch_size = rays_o.size(0);

    const uint32_t gen_samples_n_threads = 128;
    const uint32_t render_ray_n_threads = 128;

    // 1. Generate samples
    const int32_t n_intersections = 1024;
    torch::Tensor ray_offsets = torch::full({batch_size, n_intersections}, -1.0,
        torch::dtype(torch::kFloat32).device(data.device()).layout(data.layout()));
    torch::Tensor ray_steps = torch::full_like(ray_offsets, -1.0);

    gen_samples_kernel<scalar_t, branching>
        <<<n_blocks_linear<uint32_t>(batch_size, gen_samples_n_threads), gen_samples_n_threads>>>(
            tree.child_acc(),
            tree.is_child_leaf_acc(),
            tree.offset_ptr(),
            tree.scaling_ptr(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_offsets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            n_intersections,
            opt.max_samples_per_node,
            (int32_t)batch_size  // TODO: It would be nice without this cast.
    );
    #ifdef DEBUG
        printf("Gen-samples kernel complete. First few intersections follow\n");
        for (int i = 0; i < 5; i++) {
            printf("t=%f, dt=%f\n", ray_offsets[0][i].item<float>(), ray_steps[0][i].item<float>());
        }
        printf("\n");
    #endif

    // 2. Forward pass (allocate tensors)
    torch::Tensor output = torch::empty({batch_size, out_data_dim}, data.options());
    torch::Tensor interp_vals = torch::empty({batch_size, n_intersections, data_dim}, data.options());
    torch::Tensor interp_nids = torch::full({batch_size, n_intersections, 8}, -1,
        torch::dtype(torch::kInt64).device(data.device()));
    torch::Tensor interp_weights = torch::empty({batch_size, n_intersections, 8},
        torch::dtype(torch::kFloat32).device(data.device()).layout(data.layout()));
    torch::Tensor neighbor_coo = torch::empty({batch_size, 8, 3}, interp_weights.options());

    // 3. Forward pass (compute)
    render_ray_kernel<scalar_t, branching, data_dim, out_data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
            data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            tree.child_acc(),
            tree.is_child_leaf_acc(),
            tree.parent_sum,
            tree.offset_ptr(),
            tree.scaling_ptr(),
            ray_offsets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            interp_nids.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            neighbor_coo.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            opt.rgb_padding,
            opt.background_brightness,
            opt.density_softplus,
            opt.sigma_thresh,
            opt.stop_thresh,
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            (int32_t)batch_size
    );
    return {
        /*output_rgb=*/output,
        /*interpolated_vals=*/interp_vals,
        /*interpolated_n_ids=*/interp_nids,
        /*interpolation_weights=*/interp_weights,
        /*ray_offsets=*/ray_offsets,
        /*ray_steps=*/ray_steps
    };
}


template <typename scalar_t, int32_t branching, int32_t data_dim, int32_t out_data_dim>
torch::Tensor volume_render_bwd(
    OctreeCppSpec<scalar_t> & tree,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const torch::Tensor & grad_output,
    const torch::Tensor & interp_vals,
    const torch::Tensor & interp_nids,
    const torch::Tensor & interp_weights,
    const torch::Tensor & ray_offsets,
    const torch::Tensor & ray_steps,
    const RenderOptions & opt)
{
    DEVICE_GUARD(tree.data);
    const uint32_t batch_size = rays_o.size(0);
    const uint32_t render_ray_n_threads = 128;

    torch::Tensor output = torch::zeros_like(tree.data);
    render_ray_bwd_kernel<scalar_t, branching, data_dim, out_data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
            tree.parent_acc(),
            tree.offset_ptr(),
            tree.scaling_ptr(),
            ray_offsets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            interp_nids.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            opt.rgb_padding,
            opt.background_brightness,
            opt.density_softplus,
            opt.sigma_thresh,
            opt.stop_thresh,
            (int32_t)batch_size
    );
    return output;
}
