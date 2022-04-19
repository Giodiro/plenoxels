#include <stdexcept>
#include <cstdint>
#include <vector>

#include "cuda_common.h"
#include "octree_common.h"
#include "data_spec.hpp"
#include "sh.h"

#define _SOFTPLUS_M1(x) (logf(1 + expf((x) - 1)))
#define _SIGMOID(x) (1 / (1 + expf(-(x))))



__device__ __inline__ float _get_delta_scale(
    const float* __restrict__ scaling,
    float3 & __restrict__ dir)
{
    dir.x *= scaling[0];
    dir.y *= scaling[1];
    dir.z *= scaling[2];
    float delta_scale = 1.f / _norm<float>(dir.x, dir.y, dir.z);
    dir.x *= delta_scale;
    dir.y *= delta_scale;
    dir.z *= delta_scale;
    return delta_scale;
}

template <typename scalar_t>
__device__ __inline__ void _dda_unit(
        const float3& __restrict__ cen,
        const float3& __restrict__ invdir,
        float* __restrict__ tmin,
        float* __restrict__ tmax)
{
    // Intersect unit AABB
    float t1, t2;

    t1 = -cen.x * invdir.x;
    t2 = t1 + invdir.x;
    *tmin = min(t1, t2);
    *tmax = max(t1, t2);

    t1 = -cen.y * invdir.y;
    t2 = t1 + invdir.y;
    *tmin = max(*tmin, min(t1, t2));
    *tmax = min(*tmax, max(t1, t2));

    t1 = -cen.z * invdir.z;
    t2 = t1 + invdir.z;
    *tmin = max(*tmin, min(t1, t2));
    *tmax = min(*tmax, max(t1, t2));
}



template <typename scalar_t, int32_t branching>
__device__ __inline__ void stratified_sample_proposal(
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> t_icf,  // is_child_leaf
    const float3 & __restrict__ ray_o,
    const float3 & __restrict__ ray_d,
    const float3 & __restrict__ invdir,
    const int32_t max_samples,
    int32_t*  __restrict__ n_samples_inout,
    float* __restrict__ dt_inout,
    float* __restrict__ t_inout
)
{
    /*
        1. Split the current subcube into appropriate number of samples
        2. Go through the samples one by one.
        3. Once there are no more samples go to the next subcube (adding some small amount to tmax) and repeat
    */
    float3 relpos;
    scalar_t t1, t2, subcube_tmin, subcube_tmax;
    if (*n_samples_inout == 0) {
        // advance to new sub-cube
        *t_inout += *dt_inout / 2 + 1e-4;
        // new sub-cube position
        relpos = ray_o + *t_inout * ray_d;
        // New subcube info pos will hold the current offset in the new subcube
        scalar_t cube_sz;
        int64_t node_id;
        _dev_query_ninfo<scalar_t, branching>(t_child, t_icf, relpos, &cube_sz, &node_id);
        //printf("New subcube offset: %f %f %f - node id %ld - size %f\n", relpos[0], relpos[1], relpos[2], node_id, cube_sz);

        t1 = (-relpos.x + 1.0) / cube_sz * invdir.x;
        t2 = (-relpos.x - 1.0) / cube_sz * invdir.x;
        subcube_tmin = min(t1, t2);
        subcube_tmax = max(t1, t2);
        t1 = (-relpos.y + 1.0) / cube_sz * invdir.y;
        t2 = (-relpos.y - 1.0) / cube_sz * invdir.y;
        subcube_tmin = max(subcube_tmin, min(t1, t2));
        subcube_tmax = min(subcube_tmax, max(t1, t2));
        t1 = (-relpos.z + 1.0) / cube_sz * invdir.z;
        t2 = (-relpos.z - 1.0) / cube_sz * invdir.z;
        subcube_tmin = max(subcube_tmin, min(t1, t2));
        subcube_tmax = min(subcube_tmax, max(t1, t2));

        // Old code for more clarity.
        /*for (int j = 0; j < 3; ++j) {
            // first part gets the center of the cube, then go to its edges.
            // invariant l <= pos[j] <= r
            l = (pos[j] - relpos[j] / cube_sz) - (1.0 / cube_sz);
            r = (pos[j] - relpos[j] / cube_sz) + (1.0 / cube_sz);
            t1 = (r - pos[j]) * invdir[j];
            t2 = (l - pos[j]) * invdir[j];
            subcube_tmin = max(subcube_tmin, min(t1, t2));
            subcube_tmax = min(subcube_tmax, max(t1, t2));
        }*/

        // Calculate the number of samples needed in the new sub-cube
        *n_samples_inout = ceilf(max_samples * (subcube_tmax - subcube_tmin) * cube_sz / 1.7320508075688772);
        // Compute step-size for the new sub-cube
        *dt_inout = (subcube_tmax - subcube_tmin) / *n_samples_inout;
        // Correct sub-cube start position to be in middle of first delta_t-long segment
        *t_inout += subcube_tmin + *dt_inout / 2;
    } else {
        *t_inout += *dt_inout;
    }
    (*n_samples_inout)--;
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void trace_ray_backward(
        const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> t_parent,
        const float* __restrict__ t_scaling,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> interp_vals,
        const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> interp_nids,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> interp_weights,

        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ray_offsets,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ray_steps,

        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_output,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_data_out,
        const float3 & __restrict__ ray_o,
        const float3 & __restrict__ ray_d,
        RenderOptions& __restrict__ opt)
{
    const float delta_scale = _get_delta_scale(t_scaling, ray_d);
    float tmin, tmax;
    float3 pos;
    scalar_t grad_tree_val[data_dim];
    scalar_t light_intensity;
    const int out_data_dim = grad_output.size(0);
    const scalar_t d_rgb_pad = 1 + 2 * opt.rgb_padding;
    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) { out[j] = opt.background_brightness; }
        return;
    }

    scalar_t basis_fn[25];
    maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim, ray_d, basis_fn);

    scalar_t accum = 0.0;
    const scalar_t d_rgb_pad = 1 + 2 * opt.rgb_padding;
    // PASS 1: Just to compute the accum variable. This could be merged with the fwd pass (if we knew grad_output)
    light_intensity = 1.f;
    for (int i = 0; i < ray_offsets.size(0); i++) {
        float t = ray_offsets[i];
        if (t < tmin) continue;
        if (t >= tmax) break;
        float delta_t = ray_steps[i] * delta_scale;
        pos = ray.origin + t * ray.dir;

        scalar_t sigma = interp_vals[i][data_dim - 1];
        if (opt.density_softplus) { sigma = _SOFTPLUS_M1(sigma); }
        if (sigma > opt.sigma_thresh) {
            const scalar_t att = expf(-delta_t * sigma);
            const scalar_t weight = light_intensity * (1.f - att);

            scalar_t total_color = 0.f;
            for (int j = 0; j < out_data_dim; ++j) {
                int off = j * opt.basis_dim;
                scalar_t tmp = 0.0;
                for (int k = opt.min_comp; k <= opt.max_comp; ++k) {
                    tmp += basis_fn[k] * interp_vals[i][off + k];
                }
                total_color += (_SIGMOID(tmp) * d_rgb_pad - opt.rgb_padding) * grad_output[j];
            }
            light_intensity *= att;
            accum += weight * total_color;
        }
    }
    scalar_t total_grad = 0.f;
    for (int j = 0; j < out_data_dim; ++j) { total_grad += grad_output[j]; }
    accum += light_intensity * opt.background_brightness * total_grad;
    // PASS 2: Actually compute the gradient
    light_intensity = 1.f;
    for (int i = 0; i < ray_offsets.size(0); i++) {
        float t = ray_offsets[i];
        if (t < tmin) continue;
        if (t >= tmax) break;
        float delta_t = ray_steps[i] * delta_scale;
        pos = ray.origin + t * ray.dir;
        // Zero-out gradient
        for (int j = 0; j < data_dim; ++j) { grad_tree_val[j] = 0; }

        scalar_t sigma = interp_vals[i][data_dim - 1];
        const scalar_t raw_sigma = sigma;  // needed for softplus-bwd
        if (opt.density_softplus) { sigma = _SOFTPLUS_M1(sigma); }
        if (sigma > opt.sigma_thresh) {
            const scalar_t att = expf(-delta_t * sigma);
            const scalar_t weight = light_intensity * (1.f - att);

            scalar_t total_color = 0.f;
            for (int j = 0, off = 0; j < out_data_dim; ++j, off += opt.basis_dim) {
                scalar_t tmp = 0.0;
                for (int k = opt.min_comp; k <= opt.max_comp; ++k) {
                    tmp += basis_fn[k] * interp_vals[i][off + k];
                }
                const scalar_t sigmoid = _SIGMOID(tmp);
                const scalar_t tmp2 = weight * sigmoid * (1.0 - sigmoid) * grad_output[j] * d_rgb_pad;
                for (int k = opt.min_comp; k <= opt.max_comp; ++k) {
                    grad_tree_val[off + k] += basis_fn[k] * tmp2;
                }
                total_color += (sigmoid * d_rgb_pad - opt.rgb_padding) * grad_output[j];
            }
            light_intensity *= att;
            accum -= weight * total_color;
            grad_tree_val[data_dim - 1] = delta_t * (total_color * light_intensity - accum)
                *  (opt.density_softplus ? _SIGMOID(raw_sigma - 1) : 1);
            #ifdef DEBUG
                printf("t=%f - setting sigma gradient to %f\n", t, grad_tree_val[data_dim - 1]);
            #endif
            _dev_query_interp_bwd<scalar_t, data_dim>(
                /*parent=*/t_parent, /*grad=*/grad_data_out, /*weights=*/interp_weights,
                /*neighbor_ids=*/neighbor_ids, /*grad_output=*/grad_tree_val);
        }
    }
}  // trace_ray_backward


template <typename scalar_t, int32_t branching, int32_t data_dim>
__device__ __inline__ void trace_ray(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> t_data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> t_icf,  // is_child_leaf
    const bool t_parent_sum,
    const float* __restrict__ t_scaling,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ray_offsets,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> ray_steps,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> interp_vals,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> interp_nids,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> interp_weights,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> neighbor_coo,
    const float3 & __restrict__ ray_o,
    const float3 & __restrict__ ray_d,
    RenderOptions& __restrict__ opt,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> out)
{
    const float delta_scale = _get_delta_scale(t_scaling, ray_d);
    float tmin, tmax;
    float3 pos;
    scalar_t basis_fn[25];
    const int out_data_dim = out.size(0);

    const scalar_t d_rgb_pad = 1 + 2 * opt.rgb_padding;

    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) { out[j] = opt.background_brightness; }
        return;
    }

    for (int j = 0; j < out_data_dim; ++j) { out[j] = 0.0f; }
    maybe_precalc_basis<scalar_t>(opt.format, opt.basis_dim, ray_d, basis_fn);
    scalar_t light_intensity = 1.f;

    for (int i = 0; i < ray_offsets.size(0); i++) {
        float t = ray_offsets[i];
        if (t < tmin) continue;
        if (t >= tmax) break;
        float delta_t = ray_steps[i];
        pos = ray.origin + t * ray.dir;
        scalar_t *tree_val = &interp_vals[i][0];
        _dev_query_interp<scalar_t, branching, data_dim>(
            t_data, t_child, t_icf, /*in_coo=*/pos, /*weights=*/&interp_weights[i][0],
            /*neighbor_coo=*/neighbor_coo, /*neighbor_ids=*/&interp_nids[i][0], /*out_val=*/&interp_vals[i][0],
            /*parent_sum=*/t_parent_sum);

        scalar_t sigma = interp_vals[i][data_dim - 1];
        if (opt.density_softplus) { sigma = _SOFTPLUS_M1(sigma); }
        if (sigma > opt.sigma_thresh) {
            const scalar_t att = expf(-delta_t * delta_scale * sigma);  // (1 - alpha)
            const scalar_t weight = light_intensity * (1.f - att);
            light_intensity *= att;

            for (int j = 0, off = 0; j < out_data_dim; ++j, off += opt.basis_dim) {
                scalar_t tmp = 0.0;
                for (int k = opt.min_comp; k <= opt.max_comp; ++i) {
                    tmp += basis_fn[k] * interp_vals[i][off + k];
                }
                out[j] += weight * (_SIGMOID(tmp) * d_rgb_pad - opt.rgb_padding);
            }
            if (light_intensity <= opt.stop_thresh) {  // Full opacity, stop
                scalar_t scale = 1.0 / (1.0 - light_intensity);
                for (int j = 0; j < out_data_dim; ++j) { out[j] *= scale; }
                return;
            }
        }
    }
    for (int j = 0; j < out_data_dim; ++j) {
        out[j] += light_intensity * opt.background_brightness;
    }
}  // trace ray


template <typename scalar_t, int32_t branching, int32_t data_dim>
__global__ void render_ray_bwd_kernel(
    const torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> t_parent,
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
    RenderOptions& __restrict__ opt,
    const int32_t n_elements)
{
	const int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 ray_o = make_float3(rays_o[i][0], rays_o[i][1], rays_o[i][2]);
    const float3 ray_d = make_float3(rays_d[i][0], rays_d[i][1], rays_d[i][2]);
    transform_coord(ray_o, t_offset, t_scaling);

	trace_ray_backward<scalar_t, branching, data_dim>(
	    t_parent, t_scaling, interp_vals[i], interp_nids[i], interp_weights[i], ray_offsets[i], ray_steps[i],
        grad_output[i], grad_data_out, ray_o, ray_d, opt);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
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
    RenderOptions& __restrict__ opt,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,            // batch_size, data_dim
    const int32_t n_elements)
{
	const int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 ray_o = make_float3(rays_o[i][0], rays_o[i][1], rays_o[i][2]);
    const float3 ray_d = make_float3(rays_d[i][0], rays_d[i][1], rays_d[i][2]);
    transform_coord(ray_o, t_offset, t_scaling);

	trace_ray<scalar_t, branching, data_dim>(
	    t_data, t_child, t_icf, t_parent_sum, t_scaling, ray_offsets[i], ray_steps[i], interp_vals[i],
	    interp_nids[i], interp_weights[i], neighbor_coo[i], ray_o, ray_d, opt, out[i]);
}


template <typename scalar_t, int32_t branching>
__global__ void gen_samples_kernel(
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const torch::PackedTensorAccessor32<bool, 4, torch::RestrictPtrTraits> t_icf,   // is_child_leaf
    const float* __restrict__ t_offset,
    const float* __restrict__ t_scaling,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o, // batch_size, 3
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d, // batch_size, 3
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_offsets,  // batch_size, n_intersections
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_steps,    // batch_size, n_intersections
    const int32_t max_intersections,
    RenderOptions& __restrict__ opt,
    const int32_t n_elements)
{
	const int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 ray_o = make_float3(rays_o[i][0], rays_o[i][1], rays_o[i][2]);
    const float3 ray_d = make_float3(rays_d[i][0], rays_d[i][1], rays_d[i][2]);
    transform_coord(ray_o, t_offset, t_scaling);

    const float delta_scale = _get_delta_scale(t_scaling, ray_d);
    float tmin, tmax;
    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) { return; }

    float delta_t = 0;
    int32_t num_strat_samples = 0;
    float t = tmin;
    for (int32_t j = 0; j < max_intersections; j++) {
        stratified_sample_proposal<scalar_t, branching>(
            t_child, t_icf, ray_o, ray_d, invdir, opt.max_samples_per_node, &num_strat_samples, &delta_t, &t);
        if (t >= tmax) { break; }
        ray_offsets[i][j] = t;
        ray_steps[i][j] = delta_t;
    }
    if (t < tmax) {
        printf("[gen_samples_kernel] Warning: %d samples insufficient to fill cube. (tmin=%f, tmax=%f, t=%f)\n",
            max_intersections, tmin, tmax, t);
    }
}



// Compute RGB output dimension from input dimension & SH degree
__host__ int get_out_data_dim(int format, int basis_dim, int in_data_dim)
{
    if (format != FORMAT_RGBA) {
        return (in_data_dim - 1) / basis_dim;
    } else {
        return in_data_dim - 1;
    }
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> volume_render(
    Octree<scalar_t, branching, data_dim> & tree,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const RenderOptions & opt)
{
    DEVICE_GUARD(tree.data);
    const uint32_t batch_size = rays_o.size(0);

    const uint32_t gen_samples_n_threads = 128;
    const uint32_t render_ray_n_threads = 128;

    // 1. Generate samples
    const int32_t n_intersections = 1024;
    torch::Tensor ray_offsets = torch::full({batch_size, n_intersections}, -1.0,
        torch::dtype(torch::kFloat32).device(tree.data.device).layout(tree.data.layout));
    torch::Tensor ray_steps = torch::full_like(ray_offsets, -1.0);

    gen_samples_kernel<scalar_t, branching>
        <<<n_blocks_linear<uint32_t>(batch_size, gen_samples_n_threads), gen_samples_n_threads>>>(
//            tree.child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tree.child_acc(),
            tree.is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
            tree.offset.data_ptr<float>(),
            tree.scaling.data_ptr<float>(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_offsets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            n_intersections,
            opt,
            (int32_t)batch_size  // TODO: It would be nice without this cast.
    );

    // 2. Forward pass (allocate tensors)
    torch::Tensor output = torch::empty({batch_size, data_dim}, tree.data.options());
    torch::Tensor interp_vals = torch::empty({batch_size, n_intersections, data_dim}, tree.data.options());
    torch::Tensor interp_nids = torch::full({batch_size, n_intersections, 8}, -1,
        torch::dtype(torch::kInt64).device(tree.data.device).layout(tree.data.layout));
    torch::Tensor interp_weights = torch::empty({batch_size, n_intersections, 8},
        torch::dtype(torch::kFloat32).device(tree.data.device).layout(tree.data.layout));
    torch::Tensor neighbor_coo = torch::empty({batch_size, 8, 3}, interp_weights.options());

    // 3. Forward pass (compute)
    render_ray_kernel<scalar_t, branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
            tree.data.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            tree.child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
            tree.is_child_leaf.packed_accessor32<bool, 4, torch::RestrictPtrTraits>(),
            tree.parent_sum,
            tree.offset.data_ptr<float>(),
            tree.scaling.data_ptr<float>(),
            ray_offsets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            interp_nids.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            neighbor_coo.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            opt,
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            (int32_t)batch_size
    );
    return std::make_tuple(output, interp_vals, interp_nids, interp_weights);
}


template <typename scalar_t, int32_t branching, int32_t data_dim>
torch::Tensor volume_render_bwd(
    Octree<scalar_t, branching, data_dim> & tree,
    const torch::Tensor & rays_o,
    const torch::Tensor & rays_d,
    const torch::Tensor & grad_output,
    const torch::Tensor & interp_vals,
    const torch::Tensor & interp_nids,
    const torch::Tensor & interp_weights,
    const RenderOptions & opt)
{
    DEVICE_GUARD(tree.data);
    const uint32_t batch_size = rays_o.size(0);
    const uint32_t render_ray_n_threads = 128;

    torch::Tensor output = torch::zeros_like(tree.data);
    render_ray_bwd_kernel<scalar_t, branching, data_dim>
        <<<n_blocks_linear<uint32_t>(batch_size, render_ray_n_threads), render_ray_n_threads>>>(
            tree.parent.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            tree.offset.data_ptr<float>(),
            tree.scaling.data_ptr<float>(),
            ray_offsets.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            ray_steps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            interp_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            interp_nids.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            interp_weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            opt,
            (int32_t)batch_size
    );
    return output;
}
