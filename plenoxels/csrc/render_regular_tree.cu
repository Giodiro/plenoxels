#include <cmath>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>
#include <cub/warp/warp_reduce.cuh>

#include "cuda_fp16.h"

#include "render_util.cuh"
#include "cuda_util.cuh"

template <typename T, size_t N>
using Acc32 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int32_t>;
template <typename T, size_t N>
using Acc64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;


#define CUDA_THREADS_PER_BLOCK 128
#define CUDA_WARPS_PER_BLOCK (CUDA_THREADS_PER_BLOCK >> 5)
#define WARP_SIZE 32


__device__ __inline__ int32_t coo2idx(int32_t x, int32_t y, int32_t z, uint32_t grid_size) {
    return x + y * grid_size + z * grid_size * grid_size;
}

__constant__
static const float OFFSET[8][3] = {{-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5},
                                   {0.5, -0.5, -0.5},  {0.5, -0.5, 0.5},  {0.5, 0.5, -0.5},  {0.5, 0.5, 0.5}};

template <int32_t POW2_RF>
__device__ __inline__ void coo_iw_coords(
    const float * __restrict__ p_wcoo,
    int32_t * __restrict__ fn,
    int32_t * __restrict__ cn_wcoo,
    int32_t * __restrict__ fn_wcoo,
    const int32_t coarse_reso,
    const int32_t neighbor_id)
{
    constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
    int32_t cn[3], rfn[3];
    fn[0] = clamp(floor2int(p_wcoo[0] + OFFSET[neighbor_id][0]), 0, fine_reso * coarse_reso - 1);
    fn[1] = clamp(floor2int(p_wcoo[1] + OFFSET[neighbor_id][1]), 0, fine_reso * coarse_reso - 1);
    fn[2] = clamp(floor2int(p_wcoo[2] + OFFSET[neighbor_id][2]), 0, fine_reso * coarse_reso - 1);
    fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
    fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
    fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);
    *cn_wcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
    *fn_wcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);
}

template <typename T, int32_t POW2_RF> struct render_dict_kernels
{
    static __device__ __inline__ void coo_iw(
        const float * __restrict__ p_wcoo,
        const float * __restrict__ iw_multiplier,
        const int32_t coarse_reso,
        const int32_t neighbor_id,
        int32_t * __restrict__ cn_wcoo,
        int32_t * __restrict__ fn_wcoo,
        T       * __restrict__ iw)
    {
        int32_t fn[3];
        coo_iw_coords<POW2_RF>(p_wcoo, fn, cn_wcoo, fn_wcoo, coarse_reso, neighbor_id);
        *iw = static_cast<T>(
            (1.0f - myabs(p_wcoo[0] - static_cast<float>(fn[0]) - 0.5f)) *
            (1.0f - myabs(p_wcoo[1] - static_cast<float>(fn[1]) - 0.5f)) *
            (1.0f - myabs(p_wcoo[2] - static_cast<float>(fn[2]) - 0.5f)) *
            (iw_multiplier == nullptr ? 1.0f : *iw_multiplier)
        );
    }

    static __device__ __inline__ void load_cg_block(
        const T * __restrict__ coarse_grid,
              T * __restrict__ coarse_grid_shmem,
        const int32_t cn_wcoo,
        const int32_t warp_lane,
        const int32_t warp_offset,
        const int32_t S)
    {
        for (int s = warp_lane; s < S; s += 32) {
            cg_shmem[warp_offset + s] = __ldg(coarse_grid + cn_wcoo * S + s);
        }
    }

    static __device__ __inline__ void grad_loop(
        const T * __restrict__ coarse_grid_shmem,
        const T * __restrict__ atoms,
              T * __restrict__ d_coarse_grid,
              T * __restrict__ d_atoms,
        typename cub::WarpReduce<__half2>::TempStorage& __restrict__ cub_storage,
        const __half2 iw,
        const int32_t cn_wcoo,
        const int32_t fn_wcoo,
        const int32_t warp_lane,
        const int32_t warp_offset,
        const int32_t S,
        const int32_t D
    )
    {
        for (int s = 0; s < S; s++) {
            // Gradient wrt atoms
            if (warp_lane < D) {
                atomicAdd(
                    d_atoms + fn_wcoo * S * D + s * D + warp_lane,
                    cg_shmem[warp_offset + s] * iw
                );
            }
            // Gradient wrt coarse-grid
            T tmp = warp_lane < D ? atoms[fn_wcoo * S * D + s * D + warp_lane] : 0.0f;
            tmp = cub::WarpReduce<T>(cub_storage).Sum(tmp * iw);
            if (warp_lane == 0) {
                atomicAdd(d_coarse_grid + cn_wcoo * S + s, tmp);
            }
        }
    }
};

template <int32_t POW2_RF> struct render_dict_kernels<__half2, POW2_RF>
{
    static __device__ __inline__ void coo_iw(
        const float * __restrict__ p_wcoo,
        const float * __restrict__ iw_multiplier,
        const int32_t coarse_reso,
        const int32_t neighbor_id,
        int32_t * __restrict__ cn_wcoo,
        int32_t * __restrict__ fn_wcoo,
        __half2 * __restrict__ iw)
    {
        int32_t fn[3];
        coo_iw_coords<POW2_RF>(p_wcoo, fn, cn_wcoo, fn_wcoo, coarse_reso, neighbor_id);
        *iw = __float2half2_rn(
            (1.0f - myabs(p_wcoo[0] - static_cast<float>(fn[0]) - 0.5f)) *
            (1.0f - myabs(p_wcoo[1] - static_cast<float>(fn[1]) - 0.5f)) *
            (1.0f - myabs(p_wcoo[2] - static_cast<float>(fn[2]) - 0.5f)) *
            (iw_multiplier == nullptr ? 1.0f : *iw_multiplier)
        );
    }

    static __device__ __inline__ void load_cg_block(
        const __half2 * __restrict__ coarse_grid,
              __half2 * __restrict__ coarse_grid_shmem,
        const int32_t cn_wcoo,
        const int32_t warp_lane,
        const int32_t warp_offset,
        const int32_t S)
    {
        for (int s = warp_lane; s < (S >> 1); s += 32) {
            cg_shmem[warp_offset + s] = __ldg(coarse_grid + cn_wcoo * (S >> 1) + s);
        }
    }

    static __device__ __inline__ void grad_loop(
        const __half2 * __restrict__ coarse_grid_shmem,
        const __half2 * __restrict__ atoms,
              __half2 * __restrict__ d_coarse_grid,
              __half2 * __restrict__ d_atoms,
        typename cub::WarpReduce<__half2>::TempStorage& __restrict__ cub_storage,
        const __half2 iw,
        const int32_t cn_wcoo,
        const int32_t fn_wcoo,
        const int32_t warp_lane,
        const int32_t warp_offset,
        const int32_t S,
        const int32_t D
    )
    {
        for (int s = 0; s < (S >> 1); s++) {
            // Gradient wrt atoms
            __half2 tmp2 = __hmul2(cg_shmem[warp_offset + s], iw);
            if (warp_lane < D) {
                atomicAdd(
                    (__half*)d_atoms + fn_wcoo * S * D + s * 2 * D + warp_lane,
                    __low2half(tmp2)
                );
                atomicAdd(
                    (__half*)d_atoms + fn_wcoo * S * D + (s * 2 + 1) * D + warp_lane,
                    __high2half(tmp2)
                );
            }
            // Gradient wrt coarse-grid
            tmp2 = warp_lane < D ?
                __halves2half2(__ldg((__half*)atoms + fn_wcoo * S * D + (s * 2) * D + warp_lane),
                               __ldg((__half*)atoms + fn_wcoo * S * D + (s * 2 + 1) * D + warp_lane))
                : __float2half2_rn(0.0f);
            tmp2 = __hmul2(tmp2, iw);
            tmp2 = cub::WarpReduce<__half2>(cub_storage).Reduce(tmp2, Half2Sum());
            if (warp_lane == 0) {
                atomicAdd(d_coarse_grid + cn_wcoo * (S >> 1) + s, tmp2);
            }
        }
    }
};

template <typename T, int32_t POW2_RF> __device__ __inline__ void static_coo_iw(
        const float * __restrict__ p_wcoo,
        const float * __restrict__ iw_multiplier,
        const int32_t coarse_reso,
        const int32_t neighbor_id,
        int32_t * cn_wcoo,
        int32_t * fn_wcoo,
        T * iw)
{
    return render_dict_kernels<T, POW2_RF>::coo_iw(p_wcoo, iw_multiplier, coarse_reso, neighbor_id, cn_wcoo, fn_wcoo, iw);
}

template <typename T, int32_t POW2_RF> __device__ __inline__ void static_load_cg_block(
        const T * __restrict__ coarse_grid,
              T * __restrict__ coarse_grid_shmem,
        const int32_t cn_wcoo,
        const int32_t warp_lane,
        const int32_t warp_offset,
        const int32_t S)
{
    return render_dict_kernels<T, POW2_RF>::load_cg_block(coarse_grid, coarse_grid_shmem, cn_wcoo, warp_lane, warp_offset, S);
}

template <typename T, int32_t POW2_RF> __device__ __inline__ void static_grad_loop(
        const T * __restrict__ coarse_grid_shmem,
        const T * __restrict__ atoms,
              T * __restrict__ d_coarse_grid,
              T * __restrict__ d_atoms,
        typename cub::WarpReduce<__half2>::TempStorage& __restrict__ cub_storage,
        const __half2 iw,
        const int32_t cn_wcoo,
        const int32_t fn_wcoo,
        const int32_t warp_lane,
        const int32_t warp_offset,
        const int32_t S,
        const int32_t D)
{
    return render_dict_kernels<T, POW2_RF>::grad_loop(coarse_grid_shmem, atoms, d_coarse_grid, d_atoms,
        cub_storage, iw, cn_wcoo, fn_wcoo, warp_lane, warp_offset, S, D);
}

/*
template<int32_t POW2_RF>
__device__ __inline__ void coo_iw(
    const float * __restrict__ p_wcoo,
    const float * __restrict__ iw_multiplier,
    const int32_t fine_reso,
    const int32_t coarse_reso,
    const int32_t neighbor_id,
    int32_t * cn_wcoo,
    int32_t * fn_wcoo,
    __half2 * iw)
{
    int32_t fn[3], cn[3], rfn[3];
    fn[0] = clamp(floor2int(p_wcoo[0] + OFFSET[neighbor_id][0]), 0, fine_reso * coarse_reso - 1);
    fn[1] = clamp(floor2int(p_wcoo[1] + OFFSET[neighbor_id][1]), 0, fine_reso * coarse_reso - 1);
    fn[2] = clamp(floor2int(p_wcoo[2] + OFFSET[neighbor_id][2]), 0, fine_reso * coarse_reso - 1);
    fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
    fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
    fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);
    *iw = __float2half2_rn(
        (1.0f - myabs(p_wcoo[0] - static_cast<float>(fn[0]) - 0.5f)) *
        (1.0f - myabs(p_wcoo[1] - static_cast<float>(fn[1]) - 0.5f)) *
        (1.0f - myabs(p_wcoo[2] - static_cast<float>(fn[2]) - 0.5f)) *
        (iw_multiplier == nullptr ? 1.0f : *iw_multiplier)
    );
    *cn_wcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
    *fn_wcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);
}

template<int32_t POW2_RF>
__device__ __inline__ void coo_iw(
    const float * __restrict__ p_wcoo,
    const float * __restrict__ iw_multiplier,
    const int32_t fine_reso,
    const int32_t coarse_reso,
    const int32_t neighbor_id,
    int32_t * cn_wcoo,
    int32_t * fn_wcoo,
    float   * iw)
{
    int32_t fn[3], cn[3], rfn[3];
    fn[0] = clamp(floor2int(p_wcoo[0] + OFFSET[neighbor_id][0]), 0, fine_reso * coarse_reso - 1);
    fn[1] = clamp(floor2int(p_wcoo[1] + OFFSET[neighbor_id][1]), 0, fine_reso * coarse_reso - 1);
    fn[2] = clamp(floor2int(p_wcoo[2] + OFFSET[neighbor_id][2]), 0, fine_reso * coarse_reso - 1);
    fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
    fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
    fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);
    *iw = (
        (1.0f - myabs(p_wcoo[0] - static_cast<float>(fn[0]) - 0.5f)) *
        (1.0f - myabs(p_wcoo[1] - static_cast<float>(fn[1]) - 0.5f)) *
        (1.0f - myabs(p_wcoo[2] - static_cast<float>(fn[2]) - 0.5f)) *
        (iw_multiplier == nullptr ? 1.0f : *iw_multiplier)
    );
    *cn_wcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
    *fn_wcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);
}
*/


template<int32_t POW2_RF>
__device__ __inline__ void dict_grad_onept(
    const c10::Half * __restrict__ coarse_grid,     // Rc^3, S
    const c10::Half * __restrict__ atoms,           // Rf^3, S, D
          c10::Half * __restrict__ d_coarse_grid,   // Rc^3, S
          c10::Half * __restrict__ d_atoms,         // Rf^3, S, D
    const float                    grad_output,     // 1
    const float     * __restrict__ point,           // 3
          __half2   * __restrict__ cg_shmem,        // V1_WARPS_PER_BLOCK * S / 2
    typename cub::WarpReduce<__half2>::TempStorage& __restrict__ cub_storage,
    const int32_t coarse_reso,
    const int32_t D,
    const int32_t S)
{
    constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
    const int32_t warp_lane = threadIdx.x & 0x1F;
    const int32_t warp_offset = (threadIdx.x >> 5) * (S >> 1);

    const float fp[3] = {
        point[0] * coarse_reso * fine_reso, point[1] * coarse_reso * fine_reso, point[2] * coarse_reso * fine_reso};

    int32_t cn_wcoo, fn_wcoo;
    __half2 iw;
    for (int i = 0; i < 8; i++) {
        static_coo_iw<__half2, POW2_RF>(fp, &grad_output, coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
        static_load_cg_block<__half2, POW2_RF>(coarse_grid, cg_shmem, cn_wcoo, warp_lane, warp_offset, S);
        __syncwarp();
        static_grad_loop<__half2, POW2_RF>(cg_shmem, atoms, d_coarse_grid, d_atoms, cub_storage, iw, cn_wcoo,
            fn_wcoo, warp_lane, warp_offset, S, D);
        __syncwarp();
    }
}


template<int32_t POW2_RF>
__device__ __inline__ void
dict_hlf_singlept(
    const c10::Half * __restrict__ coarse_grid,  // Rc^3, S
    const c10::Half * __restrict__ atoms,        // Rf^3, S, D
    const float     * __restrict__ point,        // 3
          float     * __restrict__ out,          // D
          __half2   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
    const int32_t coarse_reso,
    const int32_t D,
    const int32_t S)
{
    constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
    const int32_t warp_lane = threadIdx.x & 0x1F;
    const int32_t warp_offset = (threadIdx.x >> 5) * S >> 1;

    const float fp[3] = {
        point[0] * coarse_reso * fine_reso, point[1] * coarse_reso * fine_reso, point[2] * coarse_reso * fine_reso};
    __half2 acc2 = __float2half2_rn(0.0f);
    int32_t cn_wcoo, fn_wcoo;
    __half2 iw;
    for (int i = 0; i < 8; i++) {
        static_coo_iw<__half2, POW2_RF>(fp, nullptr, coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
        // load w from coarse_grid to shared mem using all threads in warp
        static_load_cg_block<__half2, POW2_RF>(coarse_grid, cg_shmem, cn_wcoo, warp_lane, warp_offset, S);
        __syncwarp();
        for (int s = 0; s < S; s += 2) {
            __half2 atom_weight = warp_lane >= D ? __float2half2_rn(0.0f) :
                __halves2half2(__ldg((__half*)atoms + fn_wcoo * S * D + s * D + warp_lane),
                               __ldg((__half*)atoms + fn_wcoo * S * D + (s + 1) * D + warp_lane));
            acc2 = __hfma2(cg_shmem[warp_offset + s >> 1], atom_weight, acc2);
        }
        acc2 = __hmul2(iw, acc2);
        __syncwarp();
    }
    if (warp_lane < D) {
        out[warp_lane] = __low2float(acc2) + __high2float(acc2);
    }
}



template<int32_t POW2_RF, int32_t BASIS_DIM>
__global__ void
trace_ray(
    const c10::Half * __restrict__ coarse_grid,  // Rc^3, S
    const c10::Half * __restrict__ atoms,        // Rf^3, S, D
    const float     * __restrict__ rays_o,       // N, 3
    const float     * __restrict__ rays_d,       // N, 3
          float     * __restrict__ out,          // N, 3
    const float     * __restrict__ scaling,
    const float     * __restrict__ offset,
    const int32_t coarse_reso,
    const int32_t N,
    const int32_t S,
    const RenderOptions opt
)
{
    constexpr int32_t D = BASIS_DIM * 3 + 1;
    const int32_t warp_lane = threadIdx.x & 0x1F;
    const int32_t warp_offset = (threadIdx.x >> 5);
    const int32_t ray_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane_colorgrp = warp_lane / BASIS_DIM;
    const int32_t lane_colorgrp_id = warp_lane % BASIS_DIM;
    typedef cub::WarpReduce<float> WarpReduce;
	if (ray_id >= N) return;

    // shared memory.
    __shared__ float sphfunc_val[CUDA_WARPS_PER_BLOCK][9];
    __shared__ Ray<float> ray_spec[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReduce::TempStorage cub_storage[CUDA_WARPS_PER_BLOCK];
    __shared__ float interpolated[CUDA_WARPS_PER_BLOCK][D];
    __half2 * cg_shmem = shared_memory_proxy<__half2>(); // V1_WARPS_PER_BLOCK * S / 2;

    // Setup the ray-spec. Will copy data from rays_o, rays_d
    ray_spec[warp_offset].set(rays_o + ray_id * 3, rays_d + ray_id * 3);
    // Spherical harmonics are computed before ray normalization
    calc_sphfunc(/*basis_dim=*/BASIS_DIM, /*dir=*/ray_spec[warp_offset].dir, /*out=*/sphfunc_val[warp_offset]);
    ray_find_bounds(ray_spec[warp_offset], scaling, offset, opt.step_size, opt.near_plane);

    if (ray_spec[warp_offset].tmin > ray_spec[warp_offset].tmax) {  // Ray doesn't hit box
        out[ray_id * 3 + min(lane_colorgrp, 2)] = 1.0f;
        return;
    }

    float t = ray_spec[warp_offset].tmin, outv = 0.0f, log_light_intensity = 0.0f;
    while (t <= ray_spec[warp_offset].tmax) {
        ray_spec[warp_offset].update_pos(t);
        dict_hlf_singlept<POW2_RF>(
            coarse_grid, atoms, /*point=*/ray_spec[warp_offset].pos, /*out=*/interpolated[warp_offset], /*cg_shmem=*/cg_shmem,
            coarse_reso, D, S);
        __syncwarp();
        if (interpolated[warp_offset][D - 1] > opt.sigma_thresh) {
            float interp_val = warp_lane >= (D - 1) ? 0.0f :
                interpolated[warp_offset][warp_lane] * sphfunc_val[warp_offset][lane_colorgrp_id];
            const float pcnt = ray_spec[warp_offset].world_step * interpolated[warp_offset][D - 1];
            const float weight = myexp(log_light_intensity) * (1.f - myexp(-pcnt));
            log_light_intensity -= pcnt;

            // The reduction will also happen on the last lane which only holds sigma.
            // The value computed there is ignored.
            const float lane_color_total = WarpReduce(cub_storage[warp_offset]).HeadSegmentedSum(
                interp_val, lane_colorgrp_id == 0);
            outv += weight * mymax(lane_color_total + 0.5f, 0.0f);  // clamp [+0, infty)
            if (myexp(log_light_intensity) < opt.stop_thresh) {
                log_light_intensity = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
    outv += myexp(log_light_intensity) * 1.0f;
    if (lane_colorgrp_id == 0 && lane_colorgrp < 3) {
        out[ray_id * 3 + lane_colorgrp] = outv;
    }
}



template <int32_t POW2_RF, int32_t BASIS_DIM>
__global__ void
trace_ray_backward(
        const Acc32<float, 2> grad_output,  // N, 3
        const float * __restrict__ color_cache,  // N, 3
        const c10::Half * __restrict__ coarse_grid,
        const c10::Half * __restrict__ atoms,
        const float * __restrict__ rays_o,
        const float * __restrict__ rays_d,
              c10::Half * __restrict__ d_coarse_grid,
              c10::Half * __restrict__ d_atoms,
        const float * __restrict__ scaling,
        const float * __restrict__ offset,
        const int32_t coarse_reso,
        const int32_t N,
        const int32_t S,
        const RenderOptions opt
)
{
    constexpr int32_t D = BASIS_DIM * 3 + 1;
    const int32_t warp_lane = threadIdx.x & 0x1F;
    const int32_t warp_offset = (threadIdx.x >> 5);
    const int32_t ray_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane_colorgrp = warp_lane / BASIS_DIM;
    const int32_t lane_colorgrp_id = warp_lane % BASIS_DIM;
    // if BASIS_DIM=9, leader_mask=0000 1000 0000 0100 0000 0010 0000 0001 selecting the leaders of the color
    // groups and the final sigma dimension
    const uint32_t leader_mask = 1U | (1U << BASIS_DIM) | (1U << (2 * BASIS_DIM)) | (1U << (3 * BASIS_DIM));
    typedef cub::WarpReduce<float> WarpReducef;
    typedef cub::WarpReduce<__half2> WarpReduceh2;
	if (ray_id >= N) return;

    // shared memory.
    __shared__ float sphfunc_val[CUDA_WARPS_PER_BLOCK][9];
    __shared__ Ray<float> ray_spec[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage cub_storage[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReduceh2::TempStorage cub_storage_h2[CUDA_WARPS_PER_BLOCK];
    __shared__ float interpolated[CUDA_WARPS_PER_BLOCK][D];
    __half2 * cg_shmem = shared_memory_proxy<__half2>(); // V1_WARPS_PER_BLOCK * S / 2;

    // Setup the ray-spec. Will copy data from rays_o, rays_d
    ray_spec[warp_offset].set(rays_o + ray_id * 3, rays_d + ray_id * 3);
    // Spherical harmonics are computed before ray normalization
    calc_sphfunc(/*basis_dim=*/BASIS_DIM, /*dir=*/ray_spec[warp_offset].dir, /*out=*/sphfunc_val[warp_offset]);
    ray_find_bounds(ray_spec[warp_offset], scaling, offset, opt.step_size, opt.near_plane);

    const float c_grad_out[3] = {grad_output[ray_id][0], grad_output[ray_id][1], grad_output[ray_id][2]};
    // const float norm_factor = 2.0f / (3 * N);
    //for (int i = 0; i < 3; ++i) {
        // TODO: Figure out what the commented-out normalization did (from svox).
    //    c_grad_out[i] = grad_output[ray_id][i];//(color_cache[ray_id][i] - grad_output[ray_id][i]) * norm_factor;
    //}
    float accum = fmaf(color_cache[ray_id * 3], c_grad_out[0],
                      fmaf(color_cache[ray_id * 3 + 1], c_grad_out[1],
                           color_cache[ray_id * 3 + 2] * c_grad_out[2]));

    if (ray_spec[warp_offset].tmin > ray_spec[warp_offset].tmax) { return; }

    float t = ray_spec[warp_offset].tmin;
    const float gout = lane_colorgrp < 3 ? c_grad_out[lane_colorgrp] : 0.0f;  // avoid out-of-bounds on sigma thread
    float log_light_intensity = 0.0f;
    while (t <= ray_spec[warp_offset].tmax) {
        ray_spec[warp_offset].update_pos(t);

        dict_hlf_singlept<POW2_RF>(
            coarse_grid, atoms, /*point=*/ray_spec[warp_offset].pos, /*out=*/interpolated[warp_offset], /*cg_shmem=*/cg_shmem,
            coarse_reso, D, S);
        __syncwarp();

        if (interpolated[warp_offset][D - 1] > opt.sigma_thresh) {
            float weighted_lane_color = warp_lane >= (D - 1) ? 0.0f :
                interpolated[warp_offset][warp_lane] * sphfunc_val[warp_offset][lane_colorgrp_id];
            const float pcnt = ray_spec[warp_offset].world_step * interpolated[warp_offset][D - 1];
            const float weight = myexp(log_light_intensity) * (1.f - myexp(-pcnt));
            log_light_intensity -= pcnt;

            // Sum over all dimensions for the color of lane_colorgrp_id. Only valid in the head.
            const float lane_color_total = WarpReducef(cub_storage[warp_offset]).HeadSegmentedSum(
                weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = mymax(lane_color_total, 0.0f);  // Clamp to [+0, infty)
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout;  // the multiplication zeroes out total_color for the lanes >= D - 1

            // For each 'leader' thread (first thread in a colorgroup), sum the values in the other leaders.
            total_color += __shfl_up_sync(leader_mask, total_color, /*delta=*/BASIS_DIM);
            total_color += __shfl_up_sync(leader_mask, total_color, /*delta=*/2 * BASIS_DIM);

            // for sigma thread (and all lanes >= D - 1) this will be something random
            color_in_01 = __shfl_sync(0xffffffff, color_in_01, /*srcLane=*/lane_colorgrp * BASIS_DIM);  // this will be 0 or 1.
            const float curr_grad_color = sphfunc_val[warp_offset][lane_colorgrp_id] * (weight * color_in_01 * gout);

            accum -= weight * total_color;
            const float curr_grad_sigma = ray_spec[warp_offset].world_step * (total_color * myexp(log_light_intensity) - accum);

            dict_grad_onept<POW2_RF>(
                coarse_grid, atoms, d_coarse_grid, d_atoms,
                /*grad_output=*/warp_lane < D - 1 ? curr_grad_color : curr_grad_sigma,
                /*point=*/ray_spec[warp_offset].pos, cg_shmem,
                cub_storage_h2[warp_offset], coarse_reso, D, S);
            if (myexp(log_light_intensity) < opt.stop_thresh) { break; }
        }
        t += opt.step_size;
    }
}





using torch::autograd::variable_list;
using torch::autograd::tensor_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::Tensor;


class DictTreeRender : public Function<DictTreeRender> {
    public:
        static Tensor forward(AutogradContext *ctx,
                              Tensor coarse_grid,   // Rc^3, S
                              Tensor atoms,         // Rf^3, S, D
                              Tensor rays_o,        // N, 3
                              Tensor rays_d,        // N, 3
                              int64_t fine_reso,
                              int64_t coarse_reso,
                              double scaling,
                              double offset,
                              double step_size,
                              double sigma_thresh,
                              double stop_thresh)
        {
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            // Size checks
            if (coarse_grid.size(0) != coarse_reso * coarse_reso * coarse_reso) {
                throw std::invalid_argument("Coarse-grid has wrong first dimension");
            }
            if (coarse_grid.size(1) != atoms.size(1)) {
                throw std::invalid_argument("Coarse-grid and atoms dimension 1 doesn't match");
            }
            if (atoms.size(0) != fine_reso * fine_reso * fine_reso) {
                throw std::invalid_argument("Atoms has wrong first dimension");
            }
            if (atoms.size(2) > 32) {
                throw std::invalid_argument("Data dimension must be at most 32");
            }
            RenderOptions opt = {
                .step_size = (float)step_size,
                .sigma_thresh = (float)sigma_thresh,
                .stop_thresh = (float)stop_thresh,
                .near_plane = 0.0f,
                .last_sample_opaque = false
            };
            const int32_t N = rays_o.size(0);
            const int32_t D = atoms.size(2);
            const int32_t S = atoms.size(1);

            auto out = torch::zeros({N, 3}, rays_o.options());
            auto scaling_t = torch::tensor({scaling, scaling, scaling}, rays_o.options());
            auto offset_t = torch::tensor({offset, offset, offset}, rays_o.options());

            const dim3 grid_size(div_round_up(N, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const int32_t shared_mem = CUDA_WARPS_PER_BLOCK * S;

            //std::cout << coarse_grid << std::endl;

            #define CALL_KERNEL(T, RF, BD)                                                                              \
                trace_ray<RF, BD><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(                  \
                    coarse_grid.data_ptr<T>(), atoms.data_ptr<T>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), \
                    out.data_ptr<float>(), scaling_t.data_ptr<float>(), offset_t.data_ptr<float>(),                     \
                    coarse_reso, N, S, opt)
            if (coarse_grid.scalar_type() == at::ScalarType::Half) {
                switch (fine_reso) {
                case 2:
                    switch (D) {
                        case 4: CALL_KERNEL(c10::Half, 1, 1); break;
                        case 13: CALL_KERNEL(c10::Half, 1, 4); break;
                        case 28: CALL_KERNEL(c10::Half, 1, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 4:
                    switch (D) {
                        case 4: CALL_KERNEL(c10::Half, 2, 1); break;
                        case 13: CALL_KERNEL(c10::Half, 2, 4); break;
                        case 28: CALL_KERNEL(c10::Half, 2, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 8:
                    switch (D) {
                        case 4: CALL_KERNEL(c10::Half, 3, 1); break;
                        case 13: CALL_KERNEL(c10::Half, 3, 4); break;
                        case 28: CALL_KERNEL(c10::Half, 3, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                default: throw std::invalid_argument("fine-resolution must be 2, 4 or 8.");
                }
            } else {
                throw std::invalid_argument("Input data must be float16.");
            }
            #undef CALL_KERNEL
            ctx->save_for_backward({coarse_grid, atoms, out});
            ctx->saved_data["rays_o"] = rays_o;
            ctx->saved_data["rays_d"] = rays_d;
            ctx->saved_data["fine_reso"] = fine_reso;
            ctx->saved_data["coarse_reso"] = coarse_reso;
            ctx->saved_data["scaling"] = scaling;
            ctx->saved_data["offset"] = offset;
            ctx->saved_data["step_size"] = step_size;
            ctx->saved_data["sigma_thresh"] = sigma_thresh;
            ctx->saved_data["stop_thresh"] = stop_thresh;
            return out;
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            const auto saved = ctx->get_saved_variables();
            const Tensor coarse_grid = saved[0];
            const Tensor atoms = saved[1];
            const Tensor fwd_output = saved[2];

            const Tensor rays_o = ctx->saved_data["rays_o"].toTensor();
            const Tensor rays_d = ctx->saved_data["rays_d"].toTensor();
            const int64_t coarse_reso = ctx->saved_data["coarse_reso"].toInt();
            const int64_t fine_reso = ctx->saved_data["fine_reso"].toInt();
            const double scaling = ctx->saved_data["scaling"].toDouble();
            const double offset = ctx->saved_data["offset"].toDouble();
            const RenderOptions opt = {
                .step_size = (float)ctx->saved_data["step_size"].toDouble(),
                .sigma_thresh = (float)ctx->saved_data["sigma_thresh"].toDouble(),
                .stop_thresh = (float)ctx->saved_data["stop_thresh"].toDouble(),
                .near_plane = 0.0f,
                .last_sample_opaque = false
            };
            const int32_t N = rays_o.size(0);
            const int32_t D = atoms.size(2);
            const int32_t S = atoms.size(1);
            const Tensor grad_output = grad_outputs[0];
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();

            Tensor d_coarse_grid = torch::zeros_like(coarse_grid);
            Tensor d_atoms = torch::zeros_like(atoms);
            auto scaling_t = torch::tensor({scaling, scaling, scaling}, rays_o.options());
            auto offset_t = torch::tensor({offset, offset, offset}, rays_o.options());

            const dim3 grid_size(div_round_up(N, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const int32_t shared_mem = CUDA_WARPS_PER_BLOCK * S;

            #define CALL_KERNEL(T, RF, BD)                                                                              \
                trace_ray_backward<RF, BD><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(         \
                    grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                            \
                    fwd_output.data_ptr<float>(), coarse_grid.data_ptr<c10::Half>(),     \
                    atoms.data_ptr<c10::Half>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(),                    \
                    d_coarse_grid.data_ptr<c10::Half>(), d_atoms.data_ptr<c10::Half>(), scaling_t.data_ptr<float>(),    \
                    offset_t.data_ptr<float>(), coarse_reso, N, S, opt)
            if (coarse_grid.scalar_type() == at::ScalarType::Half) {
                switch (fine_reso) {
                case 2:
                    switch (D) {
                        case 4: CALL_KERNEL(c10::Half, 1, 1); break;
                        case 13: CALL_KERNEL(c10::Half, 1, 4); break;
                        case 28: CALL_KERNEL(c10::Half, 1, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 4:
                    switch (D) {
                        case 4: CALL_KERNEL(c10::Half, 2, 1); break;
                        case 13: CALL_KERNEL(c10::Half, 2, 4); break;
                        case 28: CALL_KERNEL(c10::Half, 2, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 8:
                    switch (D) {
                        case 4: CALL_KERNEL(c10::Half, 3, 1); break;
                        case 13: CALL_KERNEL(c10::Half, 3, 4); break;
                        case 28: CALL_KERNEL(c10::Half, 3, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                default: throw std::invalid_argument("fine-resolution must be 2, 4 or 8.");
                }
            } else {
                throw std::invalid_argument("Input data must be float16.");
            }
            #undef CALL_KERNEL
            return {d_coarse_grid, d_atoms, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }
};


Tensor dict_tree_render(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &rays_o, const Tensor &rays_d,
                        const int64_t fine_reso, const int64_t coarse_reso, const double scaling, const double offset,
                        const double step_size, const double sigma_thresh, const double stop_thresh)
{
    return DictTreeRender::apply(coarse_grid, atoms, rays_o, rays_d, fine_reso, coarse_reso, scaling, offset,
                                 step_size, sigma_thresh, stop_thresh);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::dict_tree_render", &dict_tree_render);

