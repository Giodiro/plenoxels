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



inline at::ScalarType scalar_type(at::ScalarType s) {
      return s;
}

// Same as the pytorch version, but dispatches to __half instead of c10::Half
#define AT_DISPATCH_FLOATING_TYPES_AND_CUHALF(TYPE, NAME, ...)                  \
    [&] {                                                                       \
      const auto& the_type = TYPE;                                              \
      /* don't use TYPE again in case it is an expensive or side-effect op */   \
      at::ScalarType _st = scalar_type(the_type);                               \
      RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                  \
      switch (_st) {                                                            \
        AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
        AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)   \
        AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, __half, __VA_ARGS__) \
        default:                                                                \
          AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");        \
      }                                                                         \
}()


__device__ __inline__ int32_t coo2idx(int32_t x, int32_t y, int32_t z, uint32_t grid_size) {
    return x + y * grid_size + z * grid_size * grid_size;
}

__constant__
static const float OFFSET[8][3] = {{-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5},
                                   {0.5, -0.5, -0.5},  {0.5, -0.5, 0.5},  {0.5, 0.5, -0.5},  {0.5, 0.5, 0.5}};

struct DimensionParams {
    int32_t coarse_reso;
    int32_t D;
    int32_t S;
    int32_t G;
    uint32_t warp_lane;
    // the data-dimension group which this warp-lane belongs to. 0 <= l < G
    uint32_t lane_colorgrp;
    uint32_t lane_colorgrp_id;
};


template <int32_t POW2_RF>
class DictRendererKernels
{
private:
    template<class TT>
    struct Proxy { };

public:
    template <typename T>
    __device__ __inline__ void single_point_fwd(const T   * __restrict__ coarse_grid,  // Rc^3, S
                                                const T   * __restrict__ atoms,        // Rf^3, S, D
                                                const float     * __restrict__ point,        // 3
                                                      float     * __restrict__ out,          // 1
                                                      T   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
                                                const DimensionParams& __restrict__ dims,
//                                                const int32_t coarse_reso,
//                                                const int32_t D,
//                                                const int32_t S,
//                                                const int32_t G,
//                                                const int32_t warp_lane,
//                                                const int32_t lane_colorgrp,
                                                const bool efficient_dict) const
    {
//        single_point_fwd_impl(Proxy<T>(), coarse_grid, atoms, point, out, cg_shmem, coarse_reso, D, S, G, warp_lane, lane_colorgrp, efficient_dict);
        single_point_fwd_impl(Proxy<T>(), coarse_grid, atoms, point, out, cg_shmem, dims, efficient_dict);
    }

    template <typename T>
    __device__ __inline__ void single_point_bwd(const T * __restrict__ coarse_grid,     // Rc^3, S
                                                const T * __restrict__ atoms,           // Rf^3, S, D
                                                      T * __restrict__ d_coarse_grid,   // Rc^3, S
                                                      T * __restrict__ d_atoms,         // Rf^3, S, D
                                                const float            grad_output,     // 1
                                                const float * __restrict__ point,           // 3
                                                      T * __restrict__ cg_shmem,        // S / 2
                                                typename cub::WarpReduce<T>::TempStorage& __restrict__ cub_storage,
                                                const DimensionParams& __restrict__ dims,
//                                                const int32_t coarse_reso,
//                                                const int32_t D,
//                                                const int32_t S,
//                                                const int32_t G,
//                                                const int32_t warp_lane,
//                                                const int32_t lane_colorgrp,    // the data-dimension group which this warp-lane belongs to. 0 <= l < G
//                                                const int32_t lane_colorgrp_id,
                                                const bool efficient_dict) const
    {
//        single_point_bwd_impl(Proxy<T>(), coarse_grid, atoms, d_coarse_grid, d_atoms, grad_output, point, cg_shmem,
//            cub_storage, coarse_reso, D, S, G, warp_lane, lane_colorgrp, lane_colorgrp_id, efficient_dict);
        single_point_bwd_impl(Proxy<T>(), coarse_grid, atoms, d_coarse_grid, d_atoms, grad_output, point, cg_shmem,
            cub_storage, dims, efficient_dict);
    }

private:
    __device__ __inline__ void coo_iw(const float * __restrict__ p_wcoo,
                                      const float * __restrict__ iw_multiplier,
                                      const int32_t coarse_reso,
                                      const int32_t neighbor_id,
                                      int32_t * __restrict__ cn_wcoo,
                                      int32_t * __restrict__ fn_wcoo,
                                      float   * __restrict__ iw) const
    {
        constexpr int32_t fine_reso = POW2_RF <= 0 ? 1 : 2 << (POW2_RF - 1);
        int32_t fn[3], cn[3], rfn[3];
        fn[0] = clamp(floor2int(p_wcoo[0] + OFFSET[neighbor_id][0]), 0, fine_reso * coarse_reso - 1);
        fn[1] = clamp(floor2int(p_wcoo[1] + OFFSET[neighbor_id][1]), 0, fine_reso * coarse_reso - 1);
        fn[2] = clamp(floor2int(p_wcoo[2] + OFFSET[neighbor_id][2]), 0, fine_reso * coarse_reso - 1);
        fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
        fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
        fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);
        *cn_wcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
        *fn_wcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);
        *iw = (1.0f - myabs(p_wcoo[0] - static_cast<float>(fn[0]) - 0.5f)) *
              (1.0f - myabs(p_wcoo[1] - static_cast<float>(fn[1]) - 0.5f)) *
              (1.0f - myabs(p_wcoo[2] - static_cast<float>(fn[2]) - 0.5f)) *
              (iw_multiplier == nullptr ? 1.0f : *iw_multiplier);
    }

    template<typename T>
    __device__ __inline__ void load_cg_block(const T * __restrict__ cg,
                                                   T * __restrict__ cg_out,
                                             const int32_t cn_wcoo,
                                             const int32_t warp_lane,
                                             const int32_t S) const
    {
        for (int s = warp_lane; s < S; s += 32) {
            cg_out[s] = __ldg(cg + cn_wcoo * S + s);
        }
    }

    template<typename T>
    __device__ __inline__ void load_cg_block_edict(const T * __restrict__ cg,
                                                         T * __restrict__ cg_out,
                                                   const int32_t cn_wcoo,
                                                   const int32_t warp_lane,
                                                   const int32_t S,
                                                   const int32_t G) const
    {
        // global: G, S -> shared: S, G
        int g, s;
        for (int sg = warp_lane; sg < S * G; sg += 32) {
            g = sg / S; s = sg % S;
            cg_out[s * G + g] = __ldg(cg + cn_wcoo * S * G + sg);
        }
    }

    template<typename T>
    __device__ __inline__ void single_point_fwd_impl(Proxy<T> p,
                                                     const T   * __restrict__ coarse_grid,  // Rc^3, G?, S
                                                     const T   * __restrict__ atoms,        // Rf^3, S, D
                                                     const float     * __restrict__ point,        // 3
                                                           float     * __restrict__ out,          // 1
                                                           T   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
                                                     const DimensionParams& __restrict__ dims,
                                                     const bool efficient_dict) const
    {
        constexpr int32_t fine_reso = POW2_RF <= 0 ? 1 : 2 << (POW2_RF - 1);
        const float fp[3] = {
            point[0] * dims.coarse_reso * fine_reso,
            point[1] * dims.coarse_reso * fine_reso,
            point[2] * dims.coarse_reso * fine_reso};
        T acc = 0.0f;
        float iw;
        int32_t cn_wcoo, fn_wcoo;
        for (int i = 0; i < 8; i++) {
            coo_iw(fp, nullptr, dims.coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
            if (efficient_dict) {
                load_cg_block_edict(coarse_grid, cg_shmem, cn_wcoo, dims.warp_lane, dims.S, dims.G);
            } else {
                load_cg_block(coarse_grid, cg_shmem, cn_wcoo, dims.warp_lane, dims.S);
            }
            __syncwarp();
            for (int s = 0; s < dims.S; s++) {
                T atom_weight = dims.warp_lane < dims.D ? atoms[fn_wcoo * dims.S * dims.D + s * dims.D + dims.warp_lane] : 0.0f;
                if (efficient_dict) {
                    acc = myfma(cg_shmem[s * dims.G + dims.lane_colorgrp], atom_weight * static_cast<T>(iw), acc);
                } else {
                    acc = myfma(cg_shmem[s], atom_weight * static_cast<T>(iw), acc);
                }
            }
            __syncwarp();
        }
        *out = acc;
    }
    __device__ __inline__ void single_point_fwd_impl(Proxy<__half>,
                                                     const __half   * __restrict__ coarse_grid,  // Rc^3, G?, S
                                                     const __half   * __restrict__ atoms,        // Rf^3, S, D
                                                     const float    * __restrict__ point,        // 3
                                                           float    * __restrict__ out,          // D
                                                           __half   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
                                                     const DimensionParams& __restrict__ dims,
                                                     const bool efficient_dict) const
    {
        constexpr int32_t fine_reso = POW2_RF <= 0 ? 1 : 2 << (POW2_RF - 1);

        const float fp[3] = {
            point[0] * dims.coarse_reso * fine_reso, point[1] * dims.coarse_reso * fine_reso, point[2] * dims.coarse_reso * fine_reso};
        __half2* cg_shmem2 = reinterpret_cast<__half2*>(cg_shmem);
        __half2 acc_h2 = __float2half2_rn(0.0f), iw_h2;
        float iw;
        int32_t cn_wcoo, fn_wcoo;
        for (int i = 0; i < 8; i++) {
            coo_iw(fp, nullptr, dims.coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
            iw_h2 = __float2half2_rn(iw);
            load_cg_block(reinterpret_cast<const __half2*>(coarse_grid), cg_shmem2, cn_wcoo, dims.warp_lane, (dims.S * dims.G) >> 1);
            __syncwarp();
            for (int s = 0; s < dims.S; s += 2) {
                __half2 atom_weight = dims.warp_lane >= dims.D ? __float2half2_rn(0.0f) :
                    __halves2half2(__ldg(atoms + fn_wcoo * dims.S * dims.D + s * dims.D + dims.warp_lane),
                                   __ldg(atoms + fn_wcoo * dims.S * dims.D + (s + 1) * dims.D + dims.warp_lane));
                acc_h2 = __hfma2(cg_shmem2[s >> 1], __hmul2(iw_h2, atom_weight), acc_h2);
            }
            __syncwarp();
        }
        *out = __low2float(acc_h2) + __high2float(acc_h2);
    }

    template<typename T>
    __device__ __inline__ void single_point_bwd_impl(Proxy<T> p,
                                                     const T * __restrict__ coarse_grid,     // Rc^3, G?, S
                                                     const T * __restrict__ atoms,           // Rf^3, S, D
                                                           T * __restrict__ d_coarse_grid,   // Rc^3, G?, S
                                                           T * __restrict__ d_atoms,         // Rf^3, S, D
                                                     const float            grad_output,     // 1
                                                     const float * __restrict__ point,           // 3
                                                           T * __restrict__ cg_shmem,        // S / 2
                                                     typename cub::WarpReduce<T>::TempStorage& __restrict__ cub_storage,
                                                     const DimensionParams& __restrict__ dims,
                                                     const bool efficient_dict) const
    {
        constexpr int32_t fine_reso = POW2_RF <= 0 ? 1 : 2 << (POW2_RF - 1);
        const float fp[3] = {
            point[0] * dims.coarse_reso * fine_reso, point[1] * dims.coarse_reso * fine_reso, point[2] * dims.coarse_reso * fine_reso};
        int32_t cn_wcoo, fn_wcoo;
        float iw;
        for (int i = 0; i < 8; i++) {
            coo_iw(fp, &grad_output, dims.coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
            load_cg_block(coarse_grid, cg_shmem, cn_wcoo, dims.warp_lane, dims.S * dims.G);
            __syncwarp();
            for (int s = 0; s < dims.S; s++) {
                // Gradient wrt atoms
                if (dims.warp_lane < dims.D) {
                    if (efficient_dict) {
                        atomicAdd(
                            d_atoms + fn_wcoo * dims.S * dims.D + s * dims.D + dims.warp_lane,
                            cg_shmem[dims.lane_colorgrp * dims.S + s] * static_cast<T>(iw));
                    } else {
                        atomicAdd(
                            d_atoms + fn_wcoo * dims.S * dims.D + s * dims.D + dims.warp_lane,
                            cg_shmem[s] * static_cast<T>(iw));
                    }
                }
                // Gradient wrt coarse-grid
                T tmp = dims.warp_lane < dims.D ? atoms[fn_wcoo * dims.S * dims.D + s * dims.D + dims.warp_lane] : 0.0f;
                if (efficient_dict) {
                    tmp = cub::WarpReduce<T>(cub_storage).HeadSegmentedSum(tmp * static_cast<T>(iw), dims.lane_colorgrp_id == 0);
                    if (dims.lane_colorgrp_id == 0) {
                        atomicAdd(d_coarse_grid + cn_wcoo * dims.G * dims.S + dims.lane_colorgrp * dims.S + s, tmp);  // TODO: This is not great for coalescing atomicAdds.
                    }
                } else {
                    tmp = cub::WarpReduce<T>(cub_storage).Sum(tmp * static_cast<T>(iw));
                    if (dims.warp_lane == 0) {
                        atomicAdd(d_coarse_grid + cn_wcoo * dims.S + s, tmp);
                    }
                }
            }
            __syncwarp();
        }
    }
    __device__ __inline__ void single_point_bwd_impl(Proxy<__half>,
                                                     const __half * __restrict__ coarse_grid,     // Rc^3, S
                                                     const __half * __restrict__ atoms,           // Rf^3, S, D
                                                           __half * __restrict__ d_coarse_grid,   // Rc^3, S
                                                           __half * __restrict__ d_atoms,         // Rf^3, S, D
                                                     const float                 grad_output,     // 1
                                                     const float  * __restrict__ point,           // 3
                                                           __half * __restrict__ cg_shmem,        // S / 2
                                                     typename cub::WarpReduce<__half>::TempStorage& __restrict__ cub_storage,
                                                     const DimensionParams& __restrict__ dims,
//                                                     const int32_t coarse_reso,
//                                                     const int32_t D,
//                                                     const int32_t S,
//                                                     const int32_t G,
//                                                     const int32_t warp_lane,
//                                                     const int32_t lane_colorgrp,    // the data-dimension group which this warp-lane belongs to. 0 <= l < G
//                                                     const int32_t lane_colorgrp_id,
                                                     const bool efficient_dict) const
    {
        constexpr int32_t fine_reso = POW2_RF <= 0 ? 1 : 2 << (POW2_RF - 1);
        const float fp[3] = {
            point[0] * dims.coarse_reso * fine_reso, point[1] * dims.coarse_reso * fine_reso, point[2] * dims.coarse_reso * fine_reso};

        int32_t cn_wcoo, fn_wcoo;
        float iw;
        __half2 iw_h2;
        __half2* cg_shmem2 = reinterpret_cast<__half2*>(cg_shmem);
        for (int i = 0; i < 8; i++) {
            coo_iw(fp, &grad_output, dims.coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
            iw_h2 = __float2half2_rn(iw);
            load_cg_block(reinterpret_cast<const __half2*>(coarse_grid), cg_shmem2, cn_wcoo, dims.warp_lane, (dims.S * dims.G) >> 1);
            __syncwarp();
            for (int s = 0; s < (dims.S >> 1); s++) {
                // Gradient wrt atoms
                __half2 tmp2 = __hmul2(cg_shmem2[dims.lane_colorgrp * dims.S + s], iw_h2);
                if (dims.warp_lane < dims.D) {
                    atomicAdd(
                        d_atoms + fn_wcoo * dims.S * dims.D + (s * 2) * dims.D + dims.warp_lane,
                        __low2half(tmp2));
                    atomicAdd(
                        d_atoms + fn_wcoo * dims.S * dims.D + (s * 2 + 1) * dims.D + dims.warp_lane,
                        __high2half(tmp2));
                }
                // Gradient wrt coarse-grid
                tmp2 = dims.warp_lane < dims.D ?
                    __halves2half2(__ldg(atoms + fn_wcoo * dims.S * dims.D + (s * 2) * dims.D + dims.warp_lane),
                                   __ldg(atoms + fn_wcoo * dims.S * dims.D + (s * 2 + 1) * dims.D + dims.warp_lane))
                    : __float2half2_rn(0.0f);
                tmp2 = __hmul2(tmp2, iw_h2);
                tmp2 = cub::WarpReduce<__half2>(reinterpret_cast<typename cub::WarpReduce<__half2>::TempStorage&>(cub_storage)).Reduce(tmp2, Half2Sum());  // reduce over the D dimension
                if (dims.warp_lane == 0) {
                    atomicAdd(reinterpret_cast<__half2*>(d_coarse_grid) + cn_wcoo * ((dims.S * dims.G) >> 1) + dims.lane_colorgrp * (dims.S >> 1) + s, tmp2);
                }
            }
            __syncwarp();
        }
    }
};


template<typename scalar_t, int32_t POW2_RF, int32_t BASIS_DIM>
__global__ void
trace_ray(
    const scalar_t * __restrict__ coarse_grid,  // Rc^3, S
    const scalar_t * __restrict__ atoms,        // Rf^3, S, D
    const Acc32<float, 2> rays_o,       // N, 3
    const Acc32<float, 2> rays_d,       // N, 3
          Acc32<float, 2> out,          // N, 3
    const float     * __restrict__ scaling,
    const float     * __restrict__ offset,
    const int32_t coarse_reso,
    const int32_t N,
    const int32_t S,
    const bool efficient_dict,
    const RenderOptions opt
)
{
    const DimensionParams dims {
        .coarse_reso = coarse_reso,
        .D = BASIS_DIM * 3 + 1,
        .S = S,
        .G = efficient_dict ? 4 : 1,
        .warp_lane = threadIdx.x & 0x1F,
        .lane_colorgrp = min((threadIdx.x & 0x1F) / BASIS_DIM, 3),
        .lane_colorgrp_id = (threadIdx.x & 0x1F) % BASIS_DIM
    };
    const int32_t warp_offset = (threadIdx.x >> 5);
    const int32_t ray_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const DictRendererKernels<POW2_RF> inner_renderer = DictRendererKernels<POW2_RF>();
    typedef cub::WarpReduce<float> WarpReduce;
	if (ray_id >= N) return;

    // shared memory.
    __shared__ float sphfunc_val[CUDA_WARPS_PER_BLOCK][9];
    __shared__ Ray<float> ray_spec[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReduce::TempStorage cub_storage[CUDA_WARPS_PER_BLOCK];
    scalar_t * cg_shmem = shared_memory_proxy<scalar_t>(); // V1_WARPS_PER_BLOCK * S / 2;

    // Setup the ray-spec. Will copy data from rays_o, rays_d
    ray_spec[warp_offset].set(rays_o[ray_id].data(), rays_d[ray_id].data());
    // Spherical harmonics are computed before ray normalization
    calc_sphfunc(/*basis_dim=*/BASIS_DIM, /*dir=*/ray_spec[warp_offset].dir, /*out=*/sphfunc_val[warp_offset]);
    ray_find_bounds(ray_spec[warp_offset], scaling, offset, opt.step_size, opt.near_plane);

    if (ray_spec[warp_offset].tmin > ray_spec[warp_offset].tmax && dims.lane_colorgrp < 3) {  // Ray doesn't hit box
        out[ray_id][dims.lane_colorgrp] = 1.0f;
        return;
    }

    float t = ray_spec[warp_offset].tmin;
    float outv = 0.0f, log_light_intensity = 0.0f;
    float sigma, interp_val;
    while (t <= ray_spec[warp_offset].tmax) {
        ray_spec[warp_offset].update_pos(t);

        inner_renderer.template single_point_fwd<scalar_t>(
            coarse_grid, atoms, /*point=*/ray_spec[warp_offset].pos, /*out=*/&interp_val,
            /*cg_shmem=*/cg_shmem + warp_offset * dims.S * dims.G, dims, efficient_dict);
        sigma = interp_val;  // This has an effect only in last thread in active warp.
        // broadcast sigma (stored in last coordinate) to other threads in warp
        sigma = __shfl_sync(0xffffffff, sigma, /*srcLane=*/dims.D - 1);
        if (sigma > opt.sigma_thresh) {
            interp_val *= sphfunc_val[warp_offset][dims.lane_colorgrp_id];
            const float pcnt = ray_spec[warp_offset].world_step * sigma;
            const float weight = myexp(log_light_intensity) * (1.f - myexp(-pcnt));
            log_light_intensity -= pcnt;

            // The reduction will also happen on the last lane which only holds sigma.
            // The value computed there is ignored.
            const float lane_color_total = WarpReduce(cub_storage[warp_offset]).HeadSegmentedSum(
                interp_val, dims.lane_colorgrp_id == 0);
            outv += weight * mymax(lane_color_total + 0.5f, 0.0f);  // clamp [+0, infty)
            if (myexp(log_light_intensity) < opt.stop_thresh) {
                log_light_intensity = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
    outv += myexp(log_light_intensity) * 1.0f;
    if (dims.lane_colorgrp_id == 0 && dims.lane_colorgrp < 3) {
        out[ray_id][dims.lane_colorgrp] = outv;
    }
}


template <typename scalar_t, int32_t POW2_RF, int32_t BASIS_DIM>
__global__ void
trace_ray_backward(
        const Acc32<float, 2> grad_output,  // N, 3
        const Acc32<float, 2> color_cache,  // N, 3
        const scalar_t * __restrict__ coarse_grid,
        const scalar_t * __restrict__ atoms,
        const Acc32<float, 2> rays_o,       // N, 3
        const Acc32<float, 2> rays_d,       // N, 3
              scalar_t * __restrict__ d_coarse_grid,
              scalar_t * __restrict__ d_atoms,
        const float * __restrict__ scaling,
        const float * __restrict__ offset,
        const int32_t coarse_reso,
        const int32_t N,
        const int32_t S,
        const bool efficient_dict,
        const RenderOptions opt
)
{
    const DimensionParams dims {
        .coarse_reso = coarse_reso,
        .D = BASIS_DIM * 3 + 1,
        .S = S,
        .G = efficient_dict ? 4 : 1,
        .warp_lane = threadIdx.x & 0x1F,
        .lane_colorgrp = min((threadIdx.x & 0x1F) / BASIS_DIM, 3),
        .lane_colorgrp_id = (threadIdx.x & 0x1F) % BASIS_DIM
    };
    const int32_t warp_offset = (threadIdx.x >> 5);
    const int32_t ray_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const DictRendererKernels<POW2_RF> inner_renderer = DictRendererKernels<POW2_RF>();
    // if BASIS_DIM=9, leader_mask=0000 1000 0000 0100 0000 0010 0000 0001 selecting the leaders of the color
    // groups and the final sigma dimension
    const uint32_t leader_mask = 1U | (1U << BASIS_DIM) | (1U << (2 * BASIS_DIM)) | (1U << (3 * BASIS_DIM));
    typedef cub::WarpReduce<float> WarpReducef;
    typedef cub::WarpReduce<scalar_t> WarpReduceh2;
	if (ray_id >= N) return;

    // shared memory.
    __shared__ float sphfunc_val[CUDA_WARPS_PER_BLOCK][9];
    __shared__ Ray<float> ray_spec[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage cub_storage[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReduceh2::TempStorage cub_storage_h2[CUDA_WARPS_PER_BLOCK];
    scalar_t * cg_shmem = shared_memory_proxy<scalar_t>(); // V1_WARPS_PER_BLOCK * S / 2;

    // Setup the ray-spec. Will copy data from rays_o, rays_d
    ray_spec[warp_offset].set(rays_o[ray_id].data(), rays_d[ray_id].data());
    // Spherical harmonics are computed before ray normalization
    calc_sphfunc(/*basis_dim=*/BASIS_DIM, /*dir=*/ray_spec[warp_offset].dir, /*out=*/sphfunc_val[warp_offset]);
    ray_find_bounds(ray_spec[warp_offset], scaling, offset, opt.step_size, opt.near_plane);

    const float c_grad_out[3] = {grad_output[ray_id][0], grad_output[ray_id][1], grad_output[ray_id][2]};
    // const float norm_factor = 2.0f / (3 * N);
    //for (int i = 0; i < 3; ++i) {
        // TODO: Figure out what the commented-out normalization did (from svox).
    //    c_grad_out[i] = grad_output[ray_id][i];//(color_cache[ray_id][i] - grad_output[ray_id][i]) * norm_factor;
    //}
    float accum = fmaf(color_cache[ray_id][0], c_grad_out[0],
                      fmaf(color_cache[ray_id][1], c_grad_out[1],
                           color_cache[ray_id][2] * c_grad_out[2]));

    if (ray_spec[warp_offset].tmin > ray_spec[warp_offset].tmax) { return; }

    float t = ray_spec[warp_offset].tmin;
    const float gout = dims.lane_colorgrp < 3 ? c_grad_out[dims.lane_colorgrp] : 0.0f;  // avoid out-of-bounds on sigma thread
    float log_light_intensity = 0.0f;
    float sigma, interp_val;
    while (t <= ray_spec[warp_offset].tmax) {
        ray_spec[warp_offset].update_pos(t);

        inner_renderer.template single_point_fwd<scalar_t>(
            coarse_grid, atoms,
            /*point=*/ray_spec[warp_offset].pos, /*out=*/&interp_val,
            /*cg_shmem=*/cg_shmem + warp_offset * dims.S * dims.G, dims, efficient_dict);
        sigma = interp_val;  // This has an effect only in last thread in active warp.
        // broadcast sigma (stored in last coordinate) to other threads in warp
        sigma = __shfl_sync(0xffffffff, sigma, /*srcLane=*/dims.D - 1);

        if (sigma > opt.sigma_thresh) {
            const float weighted_lane_color = interp_val * sphfunc_val[warp_offset][dims.lane_colorgrp_id];
            const float pcnt = ray_spec[warp_offset].world_step * sigma;
            const float weight = myexp(log_light_intensity) * (1.f - myexp(-pcnt));
            log_light_intensity -= pcnt;

            // Sum over all dimensions for the color of lane_colorgrp_id. Only valid in the head.
            const float lane_color_total = WarpReducef(cub_storage[warp_offset]).HeadSegmentedSum(
                weighted_lane_color, dims.lane_colorgrp_id == 0) + 0.5f;
            float total_color = mymax(lane_color_total, 0.0f);  // Clamp to [+0, infty)
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout;  // the multiplication zeroes out total_color for the lanes >= D - 1

            // For each 'leader' thread (first thread in a colorgroup), sum the values in the other leaders.
            total_color += __shfl_up_sync(leader_mask, total_color, /*delta=*/BASIS_DIM);
            total_color += __shfl_up_sync(leader_mask, total_color, /*delta=*/2 * BASIS_DIM);

            // for sigma thread (and all lanes >= D - 1) this will be something random
            color_in_01 = __shfl_sync(0xffffffff, color_in_01, /*srcLane=*/dims.lane_colorgrp * BASIS_DIM);  // this will be 0 or 1.
            const float curr_grad_color = sphfunc_val[warp_offset][dims.lane_colorgrp_id] * (weight * color_in_01 * gout);

            accum -= weight * total_color;
            const float curr_grad_sigma = ray_spec[warp_offset].world_step * (total_color * myexp(log_light_intensity) - accum);

            inner_renderer.template single_point_bwd<scalar_t>(
                coarse_grid, atoms, d_coarse_grid, d_atoms,
                /*grad_output=*/dims.warp_lane < dims.D - 1 ? curr_grad_color : curr_grad_sigma,
                /*point=*/ray_spec[warp_offset].pos, cg_shmem + warp_offset * dims.S * dims.G,
                cub_storage_h2[warp_offset], dims, efficient_dict);
            if (myexp(log_light_intensity) < opt.stop_thresh) { break; }
        }
        t += opt.step_size;
    }
}



template<typename scalar_t, int32_t POW2_RF>
__global__ void
dict_interp(const scalar_t * __restrict__ coarse_grid,  // Rc^3, S
            const scalar_t * __restrict__ atoms,  // Rf^3, S, D
            const float     * __restrict__ points,  // N, 3
                  float     * __restrict__ out,  // N, D
            const int32_t coarse_reso,
            const int32_t D,
            const int32_t N,
            const int32_t S)
{
    const int32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t warp_offset = (threadIdx.x >> 5);
    const DimensionParams dims {
        .coarse_reso = coarse_reso,
        .D = D,
        .S = S,
        .G = 1,
        .warp_lane = threadIdx.x & 0x1F,
        .lane_colorgrp = 0,
        .lane_colorgrp_id = 0
    };
    if (point_id >= N) { return; }
    const DictRendererKernels<POW2_RF> inner_renderer = DictRendererKernels<POW2_RF>();
    scalar_t * cg_shmem = shared_memory_proxy<scalar_t>(); // V1_WARPS_PER_BLOCK * S / 2;
    scalar_t reg_out;
    inner_renderer.template single_point_fwd<scalar_t>(
        coarse_grid, atoms, /*point=*/points + point_id * 3, /*out=*/&reg_out,
        /*cg_shmem=*/cg_shmem + warp_offset * dims.S, dims, /*efficient_dict=*/false);
    if (dims.warp_lane < D) {
        out[point_id * dims.D + dims.warp_lane] = reg_out;
    }
}

template <typename scalar_t, int32_t POW2_RF>
__global__ void
dict_interp_backward(const Acc32<float, 2> grad_output,             // N, D
                     const scalar_t * __restrict__ coarse_grid,
                     const scalar_t * __restrict__ atoms,
                     const float * __restrict__ points,
                           scalar_t * __restrict__ d_coarse_grid,
                           scalar_t * __restrict__ d_atoms,
                     const int32_t coarse_reso, const int32_t D, const int32_t N, const int32_t S)
{
    const int32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t warp_offset = threadIdx.x >> 5;
    const DimensionParams dims {
        .coarse_reso = coarse_reso,
        .D = D,
        .S = S,
        .G = 1,
        .warp_lane = threadIdx.x & 0x1F,
        .lane_colorgrp = 0,
        .lane_colorgrp_id = 0
    };
    typedef cub::WarpReduce<scalar_t> WarpReduce;
    if (point_id >= N) { return; }

    __shared__ typename WarpReduce::TempStorage cub_storage[CUDA_WARPS_PER_BLOCK];
    const DictRendererKernels<POW2_RF> inner_renderer = DictRendererKernels<POW2_RF>();

    scalar_t * cg_shmem = shared_memory_proxy<scalar_t>(); // V1_WARPS_PER_BLOCK * S / 2;
    const float c_grad_out = dims.warp_lane < dims.D ? grad_output[point_id][dims.warp_lane] : 0.0f;
    inner_renderer.template single_point_bwd<scalar_t>(
        coarse_grid, atoms, d_coarse_grid, d_atoms,
        /*grad_output=*/c_grad_out, /*point=*/points + point_id * 3, cg_shmem + warp_offset * dims.S,
        cub_storage[warp_offset], dims, /*efficient_dict=*/false);
}



using torch::autograd::variable_list;
using torch::autograd::tensor_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::Tensor;


template <typename scalar_t> 
scalar_t * tensor2ptr(Tensor tensor) {
    return tensor.data_ptr<scalar_t>();
}
template <>
__half * tensor2ptr<__half>(Tensor tensor) {
    return (__half*)tensor.data_ptr<c10::Half>();
}


class DictTreeRender : public Function<DictTreeRender> {
    public:
        static Tensor forward(AutogradContext *ctx,
                              Tensor coarse_grid,   // Rc^3, G?, S
                              Tensor atoms,         // Rf^3, S, D
                              Tensor rays_o,        // N, 3
                              Tensor rays_d,        // N, 3
                              int64_t fine_reso,
                              int64_t coarse_reso,
                              double scaling,
                              double offset,
                              double step_size,
                              double sigma_thresh,
                              double stop_thresh,
                              bool efficient_dict)
        {
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            // Size checks
            if (coarse_grid.size(0) != coarse_reso * coarse_reso * coarse_reso) {
                throw std::invalid_argument("Coarse-grid has wrong first dimension");
            }
            if (efficient_dict && coarse_grid.dim() != 3) {
                throw std::invalid_argument("Coarse-grid must be a 3D tensor when efficient-dict option is True.");
            }
            if (coarse_grid.dim() == 3 && coarse_grid.size(1) != 4) {
                throw std::invalid_argument("The 'group' dimension of coarse-grid must be 4.");
            }
            if (coarse_grid.size(-1) != atoms.size(1)) {
                throw std::invalid_argument("Coarse-grid and atoms have different dictionary sizes.");
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
            const int32_t G = efficient_dict ? 4 : 1;

            auto out = torch::zeros({N, 3}, rays_o.options());
            auto scaling_t = torch::tensor({scaling, scaling, scaling}, rays_o.options());
            auto offset_t = torch::tensor({offset, offset, offset}, rays_o.options());

            const dim3 grid_size(div_round_up(N, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const int32_t shared_mem = CUDA_WARPS_PER_BLOCK * S * G;

            #define CALL_KERNEL(T, RF, BD)                                                                              \
                trace_ray<T, RF, BD><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(               \
                    tensor2ptr<T>(coarse_grid), tensor2ptr<T>(atoms),                                                   \
                    rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                     \
                    rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                     \
                    out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                        \
                    scaling_t.data_ptr<float>(), offset_t.data_ptr<float>(),                                            \
                    (int32_t)coarse_reso, N, S, efficient_dict, opt)
            AT_DISPATCH_FLOATING_TYPES_AND_CUHALF(coarse_grid.scalar_type(), "trace_ray", [&] {
                switch (fine_reso) {
                case 1:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 0, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 0, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 0, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 2:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 1, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 1, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 1, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 4:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 2, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 2, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 2, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 8:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 3, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 3, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 3, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                default: throw std::invalid_argument("fine-resolution must be 1, 2, 4 or 8.");
                }
            }); 
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
            ctx->saved_data["efficient_dict"] = efficient_dict;
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
            const bool efficient_dict = ctx->saved_data["efficient_dict"].toBool();
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
            const int32_t G = efficient_dict ? 4 : 1;
            const Tensor grad_output = grad_outputs[0];
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();

            Tensor d_coarse_grid = torch::zeros_like(coarse_grid);
            Tensor d_atoms = torch::zeros_like(atoms);
            auto scaling_t = torch::tensor({scaling, scaling, scaling}, rays_o.options());
            auto offset_t = torch::tensor({offset, offset, offset}, rays_o.options());

            const dim3 grid_size(div_round_up(N, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const int32_t shared_mem = CUDA_WARPS_PER_BLOCK * S * G;

            #define CALL_KERNEL(T, RF, BD)                                                                              \
                trace_ray_backward<T, RF, BD><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(      \
                    grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                \
                    fwd_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                 \
                    tensor2ptr<T>(coarse_grid), tensor2ptr<T>(atoms),                                                   \
                    rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                     \
                    rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                     \
                    tensor2ptr<T>(d_coarse_grid), tensor2ptr<T>(d_atoms), scaling_t.data_ptr<float>(),                  \
                    offset_t.data_ptr<float>(), (int32_t)coarse_reso, N, S, efficient_dict, opt)
            AT_DISPATCH_FLOATING_TYPES_AND_CUHALF(coarse_grid.scalar_type(), "trace_ray_backward", [&] {
                switch (fine_reso) {
                case 1:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 0, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 0, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 0, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 2:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 1, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 1, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 1, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 4:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 2, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 2, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 2, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                case 8:
                    switch (D) {
                        case 4: CALL_KERNEL(scalar_t, 3, 1); break;
                        case 13: CALL_KERNEL(scalar_t, 3, 4); break;
                        case 28: CALL_KERNEL(scalar_t, 3, 9); break;
                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
                    }
                    break;
                default: throw std::invalid_argument("fine-resolution must be 1, 2, 4 or 8.");
                }
            });
            #undef CALL_KERNEL
            return {d_coarse_grid, d_atoms, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }
};


class DictInterpolate : public Function<DictInterpolate> {
    public:
        static Tensor forward(AutogradContext *ctx,
                              Tensor coarse_grid,   // Rc^3, S
                              Tensor atoms,         // Rf^3, S, D
                              Tensor points,        // N, 3
                              int64_t fine_reso,
                              int64_t coarse_reso)
        {
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            // Size checks
            if (coarse_grid.size(0) != coarse_reso * coarse_reso * coarse_reso)
                throw std::invalid_argument("Coarse-grid has wrong first dimension");
            if (coarse_grid.size(1) != atoms.size(1))
                throw std::invalid_argument("Coarse-grid and atoms dimension 1 doesn't match");
            if (atoms.size(0) != fine_reso * fine_reso * fine_reso)
                throw std::invalid_argument("Atoms has wrong first dimension");
            if (atoms.size(2) > 32)
                throw std::invalid_argument("Data dimension must be at most 32");
            ctx->save_for_backward({coarse_grid, atoms});
            ctx->saved_data["points"] = points;
            ctx->saved_data["fine_reso"] = fine_reso;
            ctx->saved_data["coarse_reso"] = coarse_reso;
            const int32_t D = atoms.size(2);
            const int32_t S = atoms.size(1);
            const int32_t N = points.size(0);
            Tensor out = torch::zeros({N, D}, torch::dtype(torch::kFloat32).device(coarse_grid.device()));

            const dim3 grid_size(div_round_up(N, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const int32_t shared_mem = CUDA_WARPS_PER_BLOCK * S;
            #define CALL_KERNEL(T, RF)                                                                                  \
                dict_interp<T, RF><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(                 \
                    tensor2ptr<T>(coarse_grid), tensor2ptr<T>(atoms), points.data_ptr<float>(), out.data_ptr<float>(),  \
                    (int32_t)coarse_reso, D, N, S)
            AT_DISPATCH_FLOATING_TYPES_AND_CUHALF(coarse_grid.scalar_type(), "dict_interpolate_fwd", [&] {
                switch(fine_reso) {
                    case 1: CALL_KERNEL(scalar_t, 0); break;
                    case 2: CALL_KERNEL(scalar_t, 1); break;
                    case 4: CALL_KERNEL(scalar_t, 2); break;
                    case 8: CALL_KERNEL(scalar_t, 3); break;
                    default: throw std::invalid_argument("fine resolution must be 1, 2, 4, or 8.");
                }
            });
            #undef CALL_KERNEL
            return out;
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            const auto saved = ctx->get_saved_variables();
            const Tensor coarse_grid = saved[0];
            const Tensor atoms = saved[1];

            const Tensor points = ctx->saved_data["points"].toTensor();
            const int64_t coarse_reso = ctx->saved_data["coarse_reso"].toInt();
            const int64_t fine_reso = ctx->saved_data["fine_reso"].toInt();
            const Tensor grad_output = grad_outputs[0];
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            const int32_t D = atoms.size(2);
            const int32_t S = atoms.size(1);
            const int32_t N = points.size(0);
            Tensor d_coarse_grid = torch::zeros_like(coarse_grid);
            Tensor d_atoms = torch::zeros_like(atoms);
            const dim3 grid_size(div_round_up(N, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const int32_t shared_mem = CUDA_WARPS_PER_BLOCK * S;
            #define CALL_KERNEL(T, RF)                                                                                  \
                dict_interp_backward<T, RF><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(        \
                    grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                                \
                    tensor2ptr<T>(coarse_grid), tensor2ptr<T>(atoms), points.data_ptr<float>(),                         \
                    tensor2ptr<T>(d_coarse_grid), tensor2ptr<T>(d_atoms), (int32_t)coarse_reso, D, N, S)
            AT_DISPATCH_FLOATING_TYPES_AND_CUHALF(coarse_grid.scalar_type(), "dict_interpolate_bwd", [&] {
                switch(fine_reso) {
                    case 1: CALL_KERNEL(scalar_t, 0); break;
                    case 2: CALL_KERNEL(scalar_t, 1); break;
                    case 4: CALL_KERNEL(scalar_t, 2); break;
                    case 8: CALL_KERNEL(scalar_t, 3); break;
                    default: throw std::invalid_argument("fine resolution must be 1, 2, 4, or 8.");
                }
            });
            #undef CALL_KERNEL
            return {d_coarse_grid, d_atoms, Tensor(), Tensor(), Tensor()};
        }
};


Tensor dict_tree_render(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &rays_o, const Tensor &rays_d,
                        const int64_t fine_reso, const int64_t coarse_reso, const double scaling, const double offset,
                        const double step_size, const double sigma_thresh, const double stop_thresh, const bool efficient_dict)
{
    return DictTreeRender::apply(coarse_grid, atoms, rays_o, rays_d, fine_reso, coarse_reso, scaling, offset,
                                 step_size, sigma_thresh, stop_thresh, efficient_dict);
}

Tensor dict_interpolate(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &points,
                        const int64_t fine_reso, const int64_t coarse_reso)
{
    return DictInterpolate::apply(coarse_grid, atoms, points, fine_reso, coarse_reso);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::dict_tree_render", &dict_tree_render)
                        .op("plenoxels::dict_interpolate", &dict_interpolate);
