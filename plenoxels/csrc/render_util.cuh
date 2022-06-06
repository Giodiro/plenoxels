#pragma once

#include "cuda_util.cuh"



template <typename T>
struct Ray {
    Ray() = default;
    __device__ Ray(const T* __restrict__ origin, const T* __restrict__ direction)
        : origin{origin[0], origin[1], origin[2]},
          dir{direction[0], direction[1], direction[2]}
    {}
    __device__ void set(const T* __restrict__ origin, const T* __restrict__ direction)
    {
        #pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            this->origin[i] = origin[i];
            this->dir[i] = direction[i];
        }
    }

    __device__ void update_pos(const T offset) {
        this->pos[0] = myfma(offset, this->dir[0], this->origin[0]);
        this->pos[1] = myfma(offset, this->dir[1], this->origin[1]);
        this->pos[2] = myfma(offset, this->dir[2], this->origin[2]);
    }

    T origin[3];
    T dir[3];
    T pos[3];
    T tmin, tmax, world_step;
};

struct RenderOptions {
    float step_size;
    float sigma_thresh;
    float stop_thresh;
    float near_plane;
};


template <typename T>
__device__ __inline__ void transform_coord(
    T* __restrict__ point,
    const T* __restrict__ scaling,
    const T* __restrict__ offset
)
{
    point[0] = myfma(point[0], scaling[0], offset[0]); // a*b + c
    point[1] = myfma(point[1], scaling[1], offset[1]); // a*b + c
    point[2] = myfma(point[2], scaling[2], offset[2]); // a*b + c
}


// Spherical functions

// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};
__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

template <typename T>
__device__ __inline__ void calc_sh(
    const uint32_t basis_dim,
    const T* __restrict__ dir,
    T* __restrict__ out)
{
    out[0] = C0;
    const T x = dir[0], y = dir[1], z = dir[2];
    const T xx = x * x, yy = y * y, zz = z * z;
    const T xy = x * y, yz = y * z, xz = x * z;
    switch (basis_dim) {
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * y;
            out[2] = C1 * z;
            out[3] = -C1 * x;
    }
}

template <typename T>
__device__ __inline__ void calc_sphfunc(
    const uint32_t basis_dim,
    const T* __restrict__ dir, // Pre-normalized
    T* __restrict__ out
)
{
    calc_sh<T>(basis_dim, dir, out);
}

template <typename T>
__device__ __inline__ void calc_sphfunc_backward(
    const int basis_dim,
    const int lane_id,
    const int ray_id,
    const T* __restrict__ dir, // Pre-normalized
    const T* __restrict__ output_saved,
    const T* __restrict__ grad_output,
    T* __restrict__ grad_basis_data
)
{
    if (grad_basis_data == nullptr) return;
    // nothing needed
}

template <typename T>
__device__ __inline__ T _get_delta_scale(
    const T* __restrict__ scaling,
    T* __restrict__ dir
)
{
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    T delta_scale = vec3_rnorm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template <typename T>
__device__ __inline__ void ray_find_bounds(
    Ray<T>& __restrict__ ray_spec,
    const T* __restrict__ scaling,
    const T* __restrict__ offset,
    const T step_size,
    const T near_plane
)
{
    // Warning: modifies ray.origin, normalizing it to 0-1 range
    transform_coord(ray_spec.origin, scaling, offset);
    // Warning: modifies ray.dir to have unit norm. world_step_out is step-size in world-coordinates.
    ray_spec.world_step = _get_delta_scale(scaling, ray_spec.dir) * step_size;

    ray_spec.tmin = near_plane / ray_spec.world_step * step_size;
    ray_spec.tmax = 2e3f;

    #pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        const T invdir = 1.0f / ray_spec.dir[i];
        // aabb intersection with [0, 1] cube
        const T t1 = (   - ray_spec.origin[i]) * invdir;
        const T t2 = (1  - ray_spec.origin[i]) * invdir;
        if (ray_d_inout[i] != 0.0f) {
            ray_spec.tmin = max(ray_spec.tmin, min(t1, t2));
            ray_spec.tmax = min(ray_spec.tmax, max(t1, t2));
        }
    }
}
