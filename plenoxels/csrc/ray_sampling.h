
#include <torch/extension.h>
#include "octree_common.h"


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
        float cube_sz;
        int64_t node_id;
        _dev_query_ninfo<scalar_t, branching>(t_child, t_icf, relpos, &cube_sz, &node_id);

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
    const int32_t max_samples_per_node,
    const int32_t n_elements)
{
	const int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    float3 ray_o = make_float3(rays_o[i][0], rays_o[i][1], rays_o[i][2]);
    float3 ray_d = make_float3(rays_d[i][0], rays_d[i][1], rays_d[i][2]);
    transform_coord(ray_o, t_offset, t_scaling);

    const float delta_scale = _get_delta_scale(t_scaling, ray_d);
    float tmin, tmax;
    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) { return; }

    float delta_t = 0;
    int32_t num_strat_samples = 0;
    float t = tmin;
    for (int j = 0; j < max_intersections; j++) {
        stratified_sample_proposal<scalar_t, branching>(
            t_child, t_icf, ray_o, ray_d, invdir, max_samples_per_node, &num_strat_samples, &delta_t, &t);
        if (t >= tmax) { break; }
        ray_offsets[i][j] = t;
        ray_steps[i][j] = delta_t;
        #ifdef DEBUG
            printf("b=%d, sample=%d/%d t=%f, dt=%f\n", i, j, max_intersections t, delta_t);
        #endif
    }
    if (t < tmax) {
        printf("[gen_samples_kernel] Warning: %d samples insufficient to fill cube. (tmin=%f, tmax=%f, t=%f)\n",
            max_intersections, tmin, tmax, t);
    }
}
