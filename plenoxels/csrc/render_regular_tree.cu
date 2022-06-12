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
                                                      float     * __restrict__ out,          // D
                                                      T   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
                                                const int32_t coarse_reso,
                                                const int32_t D,
                                                const int32_t S) const
    {
        single_point_fwd_impl(Proxy<T>(), coarse_grid, atoms, point, out, cg_shmem, coarse_reso, D, S);
    }

//    template <typename T>
//    __device__ __inline__ void single_point_bwd(
//        const T * __restrict__ coarse_grid,     // Rc^3, S
//        const T * __restrict__ atoms,           // Rf^3, S, D
//              T * __restrict__ d_coarse_grid,   // Rc^3, S
//              T * __restrict__ d_atoms,         // Rf^3, S, D
//        const float                    grad_output,     // 1
//        const float     * __restrict__ point,           // 3
//              T   * __restrict__ cg_shmem,        // S / 2
//        typename cub::WarpReduce<T>::TempStorage& __restrict__ cub_storage,
//        const int32_t coarse_reso,
//        const int32_t D,
//        const int32_t S) const
//    {
//        constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
//        const int32_t warp_lane = threadIdx.x & 0x1F;
//        const float fp[3] = {
//            point[0] * coarse_reso * fine_reso, point[1] * coarse_reso * fine_reso, point[2] * coarse_reso * fine_reso};
//
//        int32_t cn_wcoo, fn_wcoo;
//        T iw;
//        for (int i = 0; i < 8; i++) {
//            coo_iw(Proxy<T>(), fp, &grad_output, coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
//            load_cg_block(Proxy<T>(), coarse_grid, cg_shmem, cn_wcoo, warp_lane, S);
//            __syncwarp();
//            grad_loop(Proxy<T>(), cg_shmem, atoms, d_coarse_grid, d_atoms, cub_storage, iw, cn_wcoo, fn_wcoo, warp_lane, S, D);
//            __syncwarp();
//        }
//    }

private:
    /*
    template<typename T>
    __device__ __inline__ void coo_iw(Proxy<T>,
                                      const float * __restrict__ p_wcoo,
                                      const float * __restrict__ iw_multiplier,
                                      const int32_t coarse_reso,
                                      const int32_t neighbor_id,
                                      int32_t * __restrict__ cn_wcoo,
                                      int32_t * __restrict__ fn_wcoo,
                                      T       * __restrict__ iw) const
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
    __device__ __inline__ void coo_iw(Proxy<__half2>,
                                      const float * __restrict__ p_wcoo,
                                      const float * __restrict__ iw_multiplier,
                                      const int32_t coarse_reso,
                                      const int32_t neighbor_id,
                                      int32_t * __restrict__ cn_wcoo,
                                      int32_t * __restrict__ fn_wcoo,
                                      __half2 * __restrict__ iw) const
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

    template<typename T>
    __device__ __inline__ void load_cg_block(Proxy<T>,
                                            const T * __restrict__ cg,
                                            T * __restrict__ cg_out,
                                            const int32_t cn_wcoo,
                                            const int32_t warp_lane,
                                            const int32_t S) const
    {
        for (int s = warp_lane; s < S; s += 32) {
            cg_out[s] = __ldg(cg + cn_wcoo * S + s);
        }
    }
    __device__ __inline__ void load_cg_block(Proxy<__half2>,
                                            const __half2 * __restrict__ coarse_grid,
                                                  __half2 * __restrict__ coarse_grid_shmem,
                                            const int32_t cn_wcoo,
                                            const int32_t warp_lane,
                                            const int32_t S) const
    {
        for (int s = warp_lane; s < (S >> 1); s += 32) {
            coarse_grid_shmem[s] = __ldg(coarse_grid + cn_wcoo * (S >> 1) + s);
        }
    }

    template<typename T>
    __device__ __inline__ void forward_loop(Proxy<T>,
                                            const T * __restrict__ cg_sh,
                                            const T * __restrict__ atoms,
                                            T * __restrict__ acc,
                                            const int32_t fn_wcoo, const int32_t warp_lane,
                                            const int32_t S, const int32_t D) const
    {
        for (int s = 0; s < S; s ++) {
            T atom_weight = warp_lane < D ? atoms[fn_wcoo * S * D + s * D + warp_lane] : 0.0f;
            *acc = myfma(cg_sh[s], atom_weight, *acc);
        }
    }
    __device__ __inline__ void forward_loop(Proxy<__half2>,
                                                    const __half2 * __restrict__ cg_sh,
                                                    const __half2 * __restrict__ atoms,
                                                    __half2 * __restrict__ acc,
                                                    const int32_t fn_wcoo,
                                                    const int32_t warp_lane,
                                                    const int32_t S, const int32_t D) const
    {
        for (int s = 0; s < S; s += 2) {
            __half2 atom_weight = warp_lane >= D ? __float2half2_rn(0.0f) :
                __halves2half2(__ldg((__half*)atoms + fn_wcoo * S * D + s * D + warp_lane),
                               __ldg((__half*)atoms + fn_wcoo * S * D + (s + 1) * D + warp_lane));
            *acc = __hfma2(cg_sh[s >> 1], atom_weight, *acc);
        }
    }

    template<typename T>
    __device__ __inline__ void grad_loop(Proxy<T>,
                                        const T * __restrict__ coarse_grid_shmem,
                                        const T * __restrict__ atoms,
                                              T * __restrict__ d_coarse_grid,
                                              T * __restrict__ d_atoms,
                                        typename cub::WarpReduce<__half2>::TempStorage& __restrict__ cub_storage,
                                        const __half2 iw,
                                        const int32_t cn_wcoo,
                                        const int32_t fn_wcoo,
                                        const int32_t warp_lane,
                                        const int32_t S,
                                        const int32_t D) const
    {
        for (int s = 0; s < S; s++) {
            // Gradient wrt atoms
            if (warp_lane < D) {
                atomicAdd(
                    d_atoms + fn_wcoo * S * D + s * D + warp_lane,
                    coarse_grid_shmem[s] * iw
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
    __device__ __inline__ void grad_loop(Proxy<__half2>,
        const __half2 * __restrict__ coarse_grid_shmem,
        const __half2 * __restrict__ atoms,
              __half2 * __restrict__ d_coarse_grid,
              __half2 * __restrict__ d_atoms,
        typename cub::WarpReduce<__half2>::TempStorage& __restrict__ cub_storage,
        const __half2 iw,
        const int32_t cn_wcoo, const int32_t fn_wcoo, const int32_t warp_lane, const int32_t S, const int32_t D) const
    {
        for (int s = 0; s < (S >> 1); s++) {
            // Gradient wrt atoms
            __half2 tmp2 = __hmul2(coarse_grid_shmem[s], iw);
            if (warp_lane < D) {
                atomicAdd(
                    ((__half*)d_atoms) + fn_wcoo * S * D + (s * 2) * D + warp_lane,
                    __low2half(tmp2)
                );
                atomicAdd(
                    ((__half*)d_atoms) + fn_wcoo * S * D + (s * 2 + 1) * D + warp_lane,
                    __high2half(tmp2)
                );
            }
            // Gradient wrt coarse-grid
            tmp2 = warp_lane < D ?
                __halves2half2(__ldg(((__half*)atoms) + fn_wcoo * S * D + (s * 2) * D + warp_lane),
                               __ldg(((__half*)atoms) + fn_wcoo * S * D + (s * 2 + 1) * D + warp_lane))
                : __float2half2_rn(0.0f);
            tmp2 = __hmul2(tmp2, iw);
            tmp2 = cub::WarpReduce<__half2>(cub_storage).Reduce(tmp2, Half2Sum());
            if (warp_lane == 0) {
                atomicAdd(d_coarse_grid + cn_wcoo * (S >> 1) + s, tmp2);
            }
        }
    }
    */

    template<typename T>
    __device__ __inline__ void single_point_fwd_impl(Proxy<T> p,
                                                    const T   * __restrict__ coarse_grid,  // Rc^3, S
                                                    const T   * __restrict__ atoms,        // Rf^3, S, D
                                                    const float     * __restrict__ point,        // 3
                                                          float     * __restrict__ out,          // D
                                                          T   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
                                                    const int32_t coarse_reso,
                                                    const int32_t D,
                                                    const int32_t S) const
    {
        T test = 0.0f;
        printf("test %f\n", test);
//        constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
//        const int32_t warp_lane = threadIdx.x & 0x1F;
//
//        const float fp[3] = {
//            point[0] * coarse_reso * fine_reso, point[1] * coarse_reso * fine_reso, point[2] * coarse_reso * fine_reso};
//        T iw, acc = 0.0f;
//        int32_t cn_wcoo, fn_wcoo;
//        for (int i = 0; i < 8; i++) {
//            coo_iw(p, fp, nullptr, coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
//            load_cg_block(p, coarse_grid, cg_shmem, cn_wcoo, warp_lane, S);
//            __syncwarp();
//            forward_loop(p, cg_shmem, atoms, &acc, fn_wcoo, warp_lane, S, D);
//            __syncwarp();
//            acc *= iw;
//        }
//        if (warp_lane < D) {
//            out[warp_lane] = acc;
//        }
    }
    __device__ __inline__ void single_point_fwd_impl(Proxy<__half2> p,
        const __half2   * __restrict__ coarse_grid,  // Rc^3, S
        const __half2   * __restrict__ atoms,        // Rf^3, S, D
        const float     * __restrict__ point,        // 3
              float     * __restrict__ out,          // D
              __half2   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
        const int32_t coarse_reso,
        const int32_t D,
        const int32_t S)
    {
        __half2 test = __float2half2_rn(0.5f);
        printf("test %f\n", __low2float(test));
//        constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
//        const int32_t warp_lane = threadIdx.x & 0x1F;
//        const float fp[3] = {
//            point[0] * coarse_reso * fine_reso, point[1] * coarse_reso * fine_reso, point[2] * coarse_reso * fine_reso};
//        __half2 iw, acc2 = __float2half2_rn(0.0f);
//        int32_t cn_wcoo, fn_wcoo;
//        for (int i = 0; i < 8; i++) {
//            coo_iw(p, fp, nullptr, coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
//            load_cg_block(p, coarse_grid, cg_shmem, cn_wcoo, warp_lane, S);
//            __syncwarp();
//            forward_loop(p, cg_shmem, atoms, &acc2, fn_wcoo, warp_lane, S, D);
//            __syncwarp();
//            acc2 = __hmul2(iw, acc2);
//        }
//        if (warp_lane < D) {
//            out[warp_lane] = __low2float(acc2) + __high2float(acc2);
//        }
    }
};


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
    const DictRendererKernels<POW2_RF> inner_renderer = DictRendererKernels<POW2_RF>();
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

        inner_renderer.single_point_fwd<__half2>(
            reinterpret_cast<const __half2*>(coarse_grid), reinterpret_cast<const __half2*>(atoms), /*point=*/ray_spec[warp_offset].pos, /*out=*/interpolated[warp_offset],
            /*cg_shmem=*/cg_shmem + warp_offset * (S >> 1), coarse_reso, D, S);
        __syncwarp();  // sync to get the `interpolated` array in each thread.
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

            #define CALL_KERNEL(T, RF, BD)                                                                              \
                trace_ray<RF, BD><<<grid_size, block_size, shared_mem * 2, stream.stream()>>>(                  \
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

//            #define CALL_KERNEL(T, RF, BD)                                                                              \
//                trace_ray_backward<RF, BD><<<grid_size, block_size, shared_mem * sizeof(T), stream.stream()>>>(         \
//                    grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),                            \
//                    fwd_output.data_ptr<float>(), coarse_grid.data_ptr<c10::Half>(),     \
//                    atoms.data_ptr<c10::Half>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(),                    \
//                    d_coarse_grid.data_ptr<c10::Half>(), d_atoms.data_ptr<c10::Half>(), scaling_t.data_ptr<float>(),    \
//                    offset_t.data_ptr<float>(), coarse_reso, N, S, opt)
//            if (coarse_grid.scalar_type() == at::ScalarType::Half) {
//                switch (fine_reso) {
//                case 2:
//                    switch (D) {
//                        case 4: CALL_KERNEL(c10::Half, 1, 1); break;
//                        case 13: CALL_KERNEL(c10::Half, 1, 4); break;
//                        case 28: CALL_KERNEL(c10::Half, 1, 9); break;
//                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
//                    }
//                    break;
//                case 4:
//                    switch (D) {
//                        case 4: CALL_KERNEL(c10::Half, 2, 1); break;
//                        case 13: CALL_KERNEL(c10::Half, 2, 4); break;
//                        case 28: CALL_KERNEL(c10::Half, 2, 9); break;
//                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
//                    }
//                    break;
//                case 8:
//                    switch (D) {
//                        case 4: CALL_KERNEL(c10::Half, 3, 1); break;
//                        case 13: CALL_KERNEL(c10::Half, 3, 4); break;
//                        case 28: CALL_KERNEL(c10::Half, 3, 9); break;
//                        default: throw std::invalid_argument("data-dim must be 4, 13 or 28.");
//                    }
//                    break;
//                default: throw std::invalid_argument("fine-resolution must be 2, 4 or 8.");
//                }
//            } else {
//                throw std::invalid_argument("Input data must be float16.");
//            }
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

