
#include <torch/extension.h>
#include "octree_common.h"


template <int32_t branching>
__device__ __inline__ void stratified_sample_proposal(
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const float3 & __restrict__ ray_o,
    const float3 & __restrict__ ray_d,
    const float3 & __restrict__ invdir,
    const float t_max,
    const int max_samples,
          int    &  __restrict__ n_samples_inout,
          float  &  __restrict__ dt_inout,
          float  &  __restrict__ t_inout
)
{
    /*
        1. Split the current subcube into appropriate number of samples
        2. Go through the samples one by one.
        3. Once there are no more samples go to the next subcube (adding some small amount to tmax) and repeat
    */
    float3 relpos;
    int64_t node_id;
    float s_tmin, s_tmax, s_size, cube_sz;
    if (n_samples_inout == 0) {
        // advance to new sub-cube
        t_inout += dt_inout / 2 + 1e-4;
        if (t_inout >= t_max) {
            n_samples_inout = -1;
            return;
        }
        // new sub-cube position
        relpos = ray_o + t_inout * ray_d;
        // New subcube info pos will hold the current offset in the new subcube
        _dev_query_ninfo<branching>(t_child, relpos, &cube_sz, &node_id);
        _dda_unit(relpos, invdir, &s_tmin, &s_tmax);

        s_size = (s_tmax - s_tmin);
        if (s_size < 1e-4) {
            t_inout += s_tmax + 1e-4;
            dt_inout = 0;
            n_samples_inout = 0;
            printf("s_size too small. Node ID=%ld - s_tmin=%f s_tmax=%f\n", node_id, s_tmin, s_tmax);
            return;
        }

        // Calculate the number of samples needed in the new sub-cube
        n_samples_inout = (int)ceilf(max_samples * s_size / 1.7321);

        //printf("dt=%f - t=%f - relpos=%f %f %f - s_tmin=%f - s_tmax=%f - cube_sz=%f   --  new dt=%f  new t=%f\n",
        //        dt_inout, t_inout, relpos.x, relpos.y, relpos.z, s_tmin, s_tmax, cube_sz, s_size / (cube_sz * (n_samples_inout)),
        //        t_inout + s_tmin / cube_sz + s_size / (cube_sz * (n_samples_inout)));
        // Compute step-size for the new sub-cube
        dt_inout = s_size / (cube_sz * (n_samples_inout));
        // Correct sub-cube start position to be in middle of first delta_t-long segment
        t_inout += s_tmin / cube_sz + dt_inout / 2;
    } else {
        t_inout += dt_inout;
    }
    n_samples_inout--;
}



template <int32_t branching>
__global__ void gen_samples_kernel(
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> t_child,
    const float* __restrict__ t_offset,
    const float* __restrict__ t_scaling,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,     // batch_size, 3
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,     // batch_size, 3
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> ray_steps,        // batch_size, n_intersections
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> num_intersections,  // batch_size
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> intrs_pos,        // batch_size, n_intersections, 3
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d_norm,      // batch_size, 3
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
    rays_d_norm[i][0] = ray_d.x;
    rays_d_norm[i][1] = ray_d.y;
    rays_d_norm[i][2] = ray_d.z;
    float tmin, tmax;
    const float3 invdir = 1.0 / (ray_d + 1e-9);
    _dda_unit(ray_o, invdir, &tmin, &tmax);
    if (tmax < 0 || tmin > tmax) {
        num_intersections[i] = 0;
        return;
    }

    int32_t num_strat_samples = 0;
    float t = tmin,
          t_new = tmin,
          delta_t = 0;
    int j = 0;
    while (j < max_intersections) {
        stratified_sample_proposal<branching>(
            t_child, ray_o, ray_d, invdir, tmax, max_samples_per_node, num_strat_samples, delta_t, t_new);
        if (t_new >= tmax || num_strat_samples < 0) { break; }
        if (t_new - t <= 0) { continue; }
        intrs_pos[i][j][0] = ray_o.x + t * ray_d.x;
        intrs_pos[i][j][1] = ray_o.y + t * ray_d.y;
        intrs_pos[i][j][2] = ray_o.z + t * ray_d.z;
        ray_steps[i][j] = (t_new - t) * delta_scale;  // forward delta t.
        t = t_new;
        j++;
    }
    num_intersections[i] = j;
    if (t_new < tmax) {
        printf("[gen_samples_kernel] Warning (sample %d): %d samples insufficient to fill cube. (tmin=%f, tmax=%f, t=%f, t_new=%f, num_strat_samples=%d, j=%d)\n",
            i, max_intersections, tmin, tmax, t, t_new, num_strat_samples, j);
    }
}
