#include <cmath>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_reduce.cuh>

#include "cuda_fp16.h"

#include "cuda_util.cuh"


template <typename T, size_t N>
using Acc32 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int32_t>;
template <typename T, size_t N>
using Acc64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;

const int CUDA_THREADS_PER_BLOCK = 512;
const int WARP_SIZE = 32;
const int CUDA_WARPS_PER_BLOCK = CUDA_THREADS_PER_BLOCK / WARP_SIZE;


constexpr uint32_t n_blocks_linear(uint32_t n_elements, uint32_t n_threads_linear) {
    return (uint32_t)(n_elements + n_threads_linear - 1) / n_threads_linear;
}


__device__ __inline__ int32_t coo2idx(int32_t x, int32_t y, int32_t z, uint32_t grid_size) {
    return x + y * grid_size + z * grid_size * grid_size;
}

__constant__
static const float OFFSET[8][3] = {{-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5},
                                   {0.5, -0.5, -0.5},  {0.5, -0.5, 0.5},  {0.5, 0.5, -0.5},  {0.5, 0.5, 0.5}};


#define FWD_BLOCK_SIZE_X 32
#define FWD_BLOCK_SIZE_Y 16
#define NUM_POINTS_PER_THREAD 16
#define INNER_POINTS_PER_THREAD 4

#define V1_FWD_BLOCK_SIZE 512
#define V1_WARPS_PER_BLOCK (V1_FWD_BLOCK_SIZE >> 5)


template<class scalar_t, int32_t S, int32_t POW2_RF>
__global__ void
k_l2_interp_v1(const __half* __restrict__ coarse_grid,  // Rc^3, S
               const __half* __restrict__ atoms,  // Rf^3, S, D
               const float* __restrict__ points,  // N, 3
               float * __restrict__ out,  // N, D
               const uint32_t coarse_reso,
               const uint32_t D,
               const uint32_t N)
{
    constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
    const int32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t warp_lane = threadIdx.x & 0x1F;
    const int32_t warp_offset = (threadIdx.x >> 5) * S / 2;  // warp_block_id * S
    if (point_id >= N) { return; }

    __shared__ __half2 coarse_w[V1_WARPS_PER_BLOCK * S / 2];
    const float fp[3] = {points[point_id * 3] * coarse_reso * fine_reso,
                         points[point_id * 3 + 1] * coarse_reso * fine_reso,
                         points[point_id * 3 + 2] * coarse_reso * fine_reso};

    int32_t cn[3];           // corner of coarse-neighbor cell in 'coarse-grid' coordinates
    int32_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    int32_t rfn[3];          // corner of fine-neighbor cell in 'full-grid' coordinates

    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 iw2;

    int32_t cn_realcoo, fn_realcoo;
    for (int i = 0; i < 8; i++) {
        fn[0] = clamp(floor2int(fp[0] + OFFSET[i][0]), 0, fine_reso * coarse_reso - 1);
        fn[1] = clamp(floor2int(fp[1] + OFFSET[i][1]), 0, fine_reso * coarse_reso - 1);
        fn[2] = clamp(floor2int(fp[2] + OFFSET[i][2]), 0, fine_reso * coarse_reso - 1);
        fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
        fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
        fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);

        float iw = (1.0f - myabs(fp[0] - static_cast<scalar_t>(fn[0]) - 0.5f)) *
                   (1.0f - myabs(fp[1] - static_cast<scalar_t>(fn[1]) - 0.5f)) *
                   (1.0f - myabs(fp[2] - static_cast<scalar_t>(fn[2]) - 0.5f));
        iw2 = __float2half2_rn(iw);

        cn_realcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
        fn_realcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);

        // load w from coarse_grid to shared mem using all threads in warp
        for (int s = warp_lane * 2; s < S; s += 32 * 2) {
            __half2 load_tmp = __ldg(
                reinterpret_cast<const __half2*>(
                    coarse_grid + cn_realcoo * S + s));
            /*if (__low2float(load_tmp) != __low2float(load_tmp) || __high2float(load_tmp) != __high2float(load_tmp)) {
                printf("Nan when loading from coarse-grid: %f %f\n", __low2float(load_tmp), __high2float(load_tmp));
            }*/
            coarse_w[warp_offset + s / 2] = load_tmp;
            //printf("GRID - warp-lane=%d, warp-offset=%d , s=%d, accessing %d\n", warp_lane, warp_offset, s, warp_offset + s / 2);
        }
        __syncwarp();
        for (int s = 0; s < S; s += 2) {
            __half atom_weight1 = warp_lane > D ? __float2half(0.0f) : 
                __ldg(atoms + fn_realcoo * S * D + s * D + warp_lane);
            __half atom_weight2 = warp_lane > D ? __float2half(0.0f) : 
                __ldg(atoms + fn_realcoo * S * D + (s + 1) * D + warp_lane);

            __half2 atom_weight = __halves2half2(atom_weight1, atom_weight2);
            //printf("SECOND - accessing %d\n", warp_offset + s / 2);
            acc2 = __hfma2(coarse_w[warp_offset + s / 2], atom_weight, acc2);
            /*if (__low2float(acc2) != __low2float(acc2) || __high2float(acc2) != __high2float(acc2)) {
                printf("NaN encountered. atom_weight: %f %f - coarse_w: %f %f\n", __low2float(atom_weight), __high2float(atom_weight), __low2float(coarse_w[warp_offset + s/2]), __high2float(coarse_w[warp_offset + s/2]));
            }*/
            //scalar_t atom_weight = warp_lane > D ? 0.0f: atoms[fn_realcoo][s][warp_lane];
            //acc = myfma(coarse_w[warp_offset + s] * interpolation_weight, atom_weight, acc);
            //printf("A: %f, CG: %f, ACC: %f\n", __low2float(atom_weight), __low2float(coarse_w[warp_offset + s / 2]), __low2float(acc2));
        }
        __syncwarp();
        acc2 = __hmul2(iw2, acc2);
    }
    if (warp_lane < D) {
        out[point_id * D + warp_lane] = __low2float(acc2) + __high2float(acc2);
    }
}



template<class scalar_t, int32_t S, int32_t POW2_RF>
__global__ void
__launch_bounds__(FWD_BLOCK_SIZE_X * FWD_BLOCK_SIZE_Y, 2)
k_l2_interp_v2(Acc32<scalar_t, 2> coarse_grid,  // Rc^3, S
               Acc32<scalar_t, 3> atoms,        // Rf^3, D, S
               Acc32<scalar_t, 2> points,       // N, 3
               Acc32<scalar_t, 2> out,          // D, N
               const int32_t coarse_reso)
{
    /*
     * Thread-blocks are of size FWD_BLOCK_SIZE_X, FWD_BLOCK_SIZE_Y
     * The block will handle only FWD_BLOCK_SIZE_Y different fine-resolution points,
     * loading them all cooperatively into a_sh.
     *
     * The thread-block x-dimension is used to subdivide on the S dimension (num-atoms), to coalesce global loads.
     * Each row handles NUM_POINTS_PER_THREAD points in a loop. 
     *
     * The grid is of size:
     *  - num_points / FWD_BLOCK_SIZE_Y / NUM_POINTS_PER_THREAD
     *  - D
     *  - fine_reso^3 / FWD_BLOCK_SIZE_Y  TODO: This should be rounded up
     */
    constexpr int32_t num_s_per_thread = S / FWD_BLOCK_SIZE_X;
    constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
    constexpr int32_t max_rf = fine_reso * fine_reso * fine_reso;
    typedef cub::WarpReduce<scalar_t> WarpReduce;

    const int32_t start_point_id = blockIdx.x * FWD_BLOCK_SIZE_Y * NUM_POINTS_PER_THREAD + threadIdx.y * NUM_POINTS_PER_THREAD;

    const int32_t rf_id = min(blockIdx.z * FWD_BLOCK_SIZE_Y + threadIdx.y, max_rf);
    const int32_t num_valid_rf_positions = min(FWD_BLOCK_SIZE_Y, max_rf - blockIdx.z * FWD_BLOCK_SIZE_Y);
    const int32_t dim_id = blockIdx.y;
    const int32_t s_idx = threadIdx.x;

    int32_t cn[3];           // corner of coarse-neighbor cell in 'coarse-grid' coordinates
    int32_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    int32_t rfn[3];          // corner of fine-neighbor cell in 'full-grid' coordinates

    __shared__ scalar_t a_sh[FWD_BLOCK_SIZE_Y][S];
    __shared__ typename WarpReduce::TempStorage cub_storage[FWD_BLOCK_SIZE_Y];

    scalar_t accs[INNER_POINTS_PER_THREAD];
    scalar_t interpolation_weight;
    scalar_t cg_reg;

    for (int s = 0; s < num_s_per_thread; s++) {
        if (s * FWD_BLOCK_SIZE_X + s_idx < S) {
            a_sh[threadIdx.y][s * FWD_BLOCK_SIZE_X + s_idx] = atoms[rf_id][dim_id][s * FWD_BLOCK_SIZE_X + s_idx];
        } else {
            a_sh[threadIdx.y][s * FWD_BLOCK_SIZE_X + s_idx] = 0.0f;
        }
    }
    __syncthreads();

    int32_t cn_realcoo, fn_realcoo;
    for (int p = 0; p < NUM_POINTS_PER_THREAD; p += INNER_POINTS_PER_THREAD) {
        int j = 0;
        for (; j < INNER_POINTS_PER_THREAD; j++) {
            int point_id = start_point_id + p + j;
            if (point_id >= out.size(1)) break;
            accs[j] = 0.0f;
            scalar_t fp[3] = {points[point_id][0] * fine_reso * coarse_reso, points[point_id][1] * fine_reso * coarse_reso, points[point_id][2] * fine_reso * coarse_reso};
            for (int i = 0; i < 8; i++) {
                fn[0] = clamp(floor2int(fp[0] + OFFSET[i][0]), 0, fine_reso * coarse_reso - 1);
                fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);

                fn[1] = clamp(floor2int(fp[1] + OFFSET[i][1]), 0, fine_reso * coarse_reso - 1);
                fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);

                fn[2] = clamp(floor2int(fp[2] + OFFSET[i][2]), 0, fine_reso * coarse_reso - 1);
                fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);

                cn_realcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
                fn_realcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso) - blockIdx.z * FWD_BLOCK_SIZE_Y;

                if (fn_realcoo >= 0 && fn_realcoo < num_valid_rf_positions) {
                    interpolation_weight = (1.0f - myabs(fp[0] - static_cast<scalar_t>(fn[0]) - 0.5f)) *
                                           (1.0f - myabs(fp[1] - static_cast<scalar_t>(fn[1]) - 0.5f)) *
                                           (1.0f - myabs(fp[2] - static_cast<scalar_t>(fn[2]) - 0.5f));
                    for (int s = 0; s < num_s_per_thread; s++) {
                        // out-of-bounds reads will be zeroed out when multiplied by a_sh
                        cg_reg = coarse_grid[cn_realcoo][min(s * FWD_BLOCK_SIZE_X + s_idx, S)]; 
                        cg_reg *= interpolation_weight;
                        accs[j] = myfma(cg_reg, a_sh[fn_realcoo][s * FWD_BLOCK_SIZE_X + s_idx], accs[j]);
                    }
                }
            }
            accs[j] = WarpReduce(cub_storage[threadIdx.y]).Sum(accs[j]);
        }
        if (s_idx == 0) {
            for (int k = 0; k < j; k++) {
                atomicAdd(&out[dim_id][start_point_id + p + k], accs[k]);
            }
        }
    }
}




using torch::autograd::variable_list;
using torch::autograd::tensor_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::Tensor;


class L2InterpFunctionv1 : public Function<L2InterpFunctionv1> {
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

            ctx->save_for_backward({coarse_grid, atoms});
            ctx->saved_data["points"] = points;
            ctx->saved_data["fine_reso"] = fine_reso;
            ctx->saved_data["coarse_reso"] = coarse_reso;

            const int32_t D = atoms.size(2);
            const int32_t S = atoms.size(1);
            const int32_t N = points.size(0);
            auto out = torch::zeros({N, D}, points.options());

            const dim3 grid_size(n_blocks_linear(N, V1_WARPS_PER_BLOCK));
            const dim3 block_size(V1_FWD_BLOCK_SIZE);

            AT_DISPATCH_FLOATING_TYPES_AND(torch::kHalf, coarse_grid.scalar_type(), "dispatch_l2interpv1_fwd", [&] {
                switch(fine_reso) {
                    case 4:
                        switch (S) {
                            case 64:
                                k_l2_interp_v1<scalar_t, 64, 2><<<grid_size, block_size, 0, stream.stream()>>>(
                                    (__half*)coarse_grid.data_ptr<at::Half>(),
                                    (__half*)atoms.data_ptr<at::Half>(),
                                    points.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    (uint32_t)coarse_reso, D, N
                                );
                                break;
                            case 128:
                                k_l2_interp_v1<scalar_t, 128, 2><<<grid_size, block_size, 0, stream.stream()>>>(
                                    (__half*)coarse_grid.data_ptr<at::Half>(),
                                    (__half*)atoms.data_ptr<at::Half>(),
                                    points.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    (uint32_t)coarse_reso, D, N
                                );
                                break;
                            case 256:
                                k_l2_interp_v1<scalar_t, 256, 2><<<grid_size, block_size, 0, stream.stream()>>>(
                                    (__half*)coarse_grid.data_ptr<at::Half>(),
                                    (__half*)atoms.data_ptr<at::Half>(),
                                    points.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    (uint32_t)coarse_reso, D, N
                                );
                                break;
                        }
                        break;
                    case 8:
                        switch (S) {
                            case 64:
                                k_l2_interp_v1<scalar_t, 64, 3><<<grid_size, block_size, 0, stream.stream()>>>(
                                    (__half*)coarse_grid.data_ptr<at::Half>(),
                                    (__half*)atoms.data_ptr<at::Half>(),
                                    points.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    (uint32_t)coarse_reso, D, N
                                );
                                break;
                            case 128:
                                k_l2_interp_v1<scalar_t, 128, 3><<<grid_size, block_size, 0, stream.stream()>>>(
                                    (__half*)coarse_grid.data_ptr<at::Half>(),
                                    (__half*)atoms.data_ptr<at::Half>(),
                                    points.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    (uint32_t)coarse_reso, D, N
                                );
                                break;
                            case 256:
                                k_l2_interp_v1<scalar_t, 256, 3><<<grid_size, block_size, 0, stream.stream()>>>(
                                    (__half*)coarse_grid.data_ptr<at::Half>(),
                                    (__half*)atoms.data_ptr<at::Half>(),
                                    points.data_ptr<float>(),
                                    out.data_ptr<float>(),
                                    (uint32_t)coarse_reso, D, N
                                );
                                break;
                        }
                        break;
                }
            });
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

            Tensor d_coarse_grid = torch::zeros_like(coarse_grid);
            Tensor d_atoms = torch::zeros_like(atoms);

            const dim3 grid_size_dcg(n_blocks_linear(points.size(0), CUDA_THREADS_PER_BLOCK / WARP_SIZE));
            const dim3 block_size_dcg(CUDA_THREADS_PER_BLOCK);

            const dim3 grid_size_da(points.size(0));
            const dim3 block_size_da(round_up(grad_output.size(1), 32), 8);  // D * S

            return {d_coarse_grid, d_atoms, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }
};


class L2InterpFunctionv2 : public Function<L2InterpFunctionv2> {
    public:
        static Tensor forward(AutogradContext *ctx,
                              Tensor coarse_grid,   // Rc^3, S
                              Tensor atoms,         // Rf^3, D, S
                              Tensor points,        // N, 3
                              int64_t fine_reso,
                              int64_t coarse_reso)
        {
            const at::cuda::CUDAGuard device_guard(coarse_grid.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            // Size checks
            if (coarse_grid.size(0) != coarse_reso * coarse_reso * coarse_reso) {
                throw std::invalid_argument("Coarse-grid has wrong first dimension");
            }
            if (coarse_grid.size(1) != atoms.size(2)) {
                throw std::invalid_argument("Coarse-grid and atoms dimension 1 doesn't match");
            }
            if (atoms.size(0) != fine_reso * fine_reso * fine_reso) {
                throw std::invalid_argument("Atoms has wrong first dimension");
            }
//            if (atoms.size(1) > 32) {
//                throw std::invalid_argument("Data dimension must be at most 32");
//            }

            ctx->save_for_backward({coarse_grid, atoms});
            ctx->saved_data["points"] = points;
            ctx->saved_data["fine_reso"] = fine_reso;
            ctx->saved_data["coarse_reso"] = coarse_reso;

            const int64_t D = atoms.size(1);
            const int64_t S = atoms.size(2);
            auto out = torch::zeros({D, points.size(0)}, atoms.options());

            const dim3 grid_size((uint32_t)(points.size(0) / FWD_BLOCK_SIZE_Y / NUM_POINTS_PER_THREAD), 
                                 (uint32_t)D, 
                                 (uint32_t)(fine_reso * fine_reso * fine_reso) / FWD_BLOCK_SIZE_Y);
            const dim3 block_size(FWD_BLOCK_SIZE_X, FWD_BLOCK_SIZE_Y);
//            const dim3 grid_size(n_blocks_linear(points.size(0), CUDA_WARPS_PER_BLOCK));
//            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
//            const uint32_t shared_mem = CUDA_WARPS_PER_BLOCK * coarse_grid.size(1);

            fast_divmod fast_divmod_fine_reso((int32_t)fine_reso);
            AT_DISPATCH_FLOATING_TYPES(coarse_grid.scalar_type(), "dispatch_l2interpv2_fwd", [&] {
                switch(fine_reso) {
                    case 4:
                        switch (S) {
                            case 64:
                                k_l2_interp_v2<scalar_t, 64, 1><<<grid_size, block_size, 0, stream.stream()>>>(
                                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    (uint32_t)coarse_reso
                                );
                                break;
                            case 128:
                                k_l2_interp_v2<scalar_t, 128, 1><<<grid_size, block_size, 0, stream.stream()>>>(
                                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    (uint32_t)coarse_reso
                                );
                                break;
                            case 256:
                                k_l2_interp_v2<scalar_t, 256, 1><<<grid_size, block_size, 0, stream.stream()>>>(
                                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    (uint32_t)coarse_reso
                                );
                                break;
                        }
                        break;
                    case 8:
                        switch (S) {
                            case 64:
                                k_l2_interp_v2<scalar_t, 64, 3><<<grid_size, block_size, 0, stream.stream()>>>(
                                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    (uint32_t)coarse_reso
                                );
                                break;
                            case 128:
                                k_l2_interp_v2<scalar_t, 128, 3><<<grid_size, block_size, 0, stream.stream()>>>(
                                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    (uint32_t)coarse_reso
                                );
                                break;
                            case 256:
                                k_l2_interp_v2<scalar_t, 256, 3><<<grid_size, block_size, 0, stream.stream()>>>(
                                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                    (uint32_t)coarse_reso
                                );
                                break;
                        }
                        break;
                }
            });
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

            Tensor d_coarse_grid = torch::zeros_like(coarse_grid);
            Tensor d_atoms = torch::zeros_like(atoms);

            const dim3 grid_size_dcg(n_blocks_linear(points.size(0), CUDA_THREADS_PER_BLOCK / WARP_SIZE));
            const dim3 block_size_dcg(CUDA_THREADS_PER_BLOCK);

            const dim3 grid_size_da(points.size(0));
            const dim3 block_size_da(round_up(grad_output.size(1), 32), 8);  // D * S

            return {d_coarse_grid, d_atoms, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }
};



Tensor l2_interp_v1(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &points, const int64_t fine_reso,
                    const int64_t coarse_reso, const double fine_vl, const double coarse_vl)
{
    return L2InterpFunctionv1::apply(coarse_grid, atoms, points, fine_reso, coarse_reso);
}


Tensor l2_interp_v2(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &points, const int64_t fine_reso,
                    const int64_t coarse_reso, const double fine_vl, const double coarse_vl)
{
    return L2InterpFunctionv2::apply(coarse_grid, atoms, points, fine_reso, coarse_reso);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::l2_interp_v2", &l2_interp_v2)
                        .op("plenoxels::l2_interp_v1", &l2_interp_v1);
