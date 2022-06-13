#include <cmath>
#include <stdexcept>
#include <tuple>

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_reduce.cuh>

#include "cuda_fp16.h"

#include "cuda_util.cuh"


#define V1_FWD_BLOCK_SIZE 512
#define V1_WARPS_PER_BLOCK (V1_FWD_BLOCK_SIZE >> 5)


template <typename T, size_t N>
using Acc32 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int32_t>;
template <typename T, size_t N>
using Acc64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;


__device__ __inline__ int32_t coo2idx(int32_t x, int32_t y, int32_t z, uint32_t grid_size) {
    return x + y * grid_size + z * grid_size * grid_size;
}

__constant__
static const float OFFSET[8][3] = {{-0.5, -0.5, -0.5}, {-0.5, -0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5},
                                   {0.5, -0.5, -0.5},  {0.5, -0.5, 0.5},  {0.5, 0.5, -0.5},  {0.5, 0.5, 0.5}};


template<int32_t POW2_RF>
__device__ __inline__ void compute_coo_iw(
    const float * __restrict__ p_wcoo,
    const int32_t fine_reso,
    const int32_t coarse_reso,
    const int32_t neighbor_id,
    int32_t * cn_wcoo,
    int32_t * fn_wcoo,
    float * iw)
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
        (1.0f - myabs(p_wcoo[2] - static_cast<float>(fn[2]) - 0.5f)));
    *cn_wcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
    *fn_wcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);
}


template<int32_t POW2_RF, typename output_t, typename input_t>
__global__ void
k_l2_interp(const input_t* __restrict__ coarse_grid,  // Rc^3, S
            const input_t* __restrict__ atoms,  // Rf^3, S, D
            const float* __restrict__ points,  // N, 3
                  output_t* __restrict__ out,  // N, D
            const int32_t coarse_reso,
            const int32_t D,
            const int32_t N,
            const int32_t S)
{
    constexpr int32_t fine_reso = 2 << (POW2_RF - 1);
    const int32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t warp_lane = threadIdx.x & 0x1F;
    const int32_t warp_offset = (threadIdx.x >> 5) * S;
    if (point_id >= N) { return; }

    input_t * coarse_w = shared_memory_proxy<input_t>(); // V1_WARPS_PER_BLOCK * S;
    const float fp[3] = {points[point_id * 3] * coarse_reso * fine_reso,
                         points[point_id * 3 + 1] * coarse_reso * fine_reso,
                         points[point_id * 3 + 2] * coarse_reso * fine_reso};
    input_t acc = 0.0f;
    int32_t cn_wcoo, fn_wcoo;
    float iw;
    for (int i = 0; i < 8; i++) {
        compute_coo_iw<POW2_RF>(fp, fine_reso, coarse_reso, i, &cn_wcoo, &fn_wcoo, &iw);
        // load w from coarse_grid to shared mem using all threads in warp
        for (int s = warp_lane; s < S; s += 32) {
            //*reinterpret_cast<float2*>(coarse_w + (warp_offset + s)) = *reinterpret_cast<const float2*>(coarse_grid + (cn_realcoo * S + s));
            coarse_w[warp_offset + s] = coarse_grid[cn_wcoo * S + s];
        }
        __syncwarp();
        for (int s = 0; s < S; s ++) {
            input_t atom_weight = warp_lane > D ? (input_t)0.0f : atoms[fn_wcoo * S * D + s * D + warp_lane];
            acc = myfma(coarse_w[warp_offset + s], atom_weight, acc);
        }
        __syncwarp();
        acc *= iw;
    }
    if (warp_lane < D) {
        out[point_id * D + warp_lane] = (output_t)acc;
    }
}


template<int32_t POW2_RF>
__device__ __forceinline__ void
k_l2_interp_hlf_single_pt(
    const c10::Half * __restrict__ coarse_grid,  // Rc^3, S
    const c10::Half * __restrict__ atoms,        // Rf^3, S, D
    const float     * __restrict__ point,        // 3
          float     * __restrict__ out,          // D
          __half2   * __restrict__ cg_shmem,     // V1_WARPS_PER_BLOCK * S / 2
    const int32_t coarse_reso,
    const int32_t D,
    const int32_t S)
{
    const uint32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const uint32_t warp_block_id = threadIdx.x / WARP_SIZE;
    const uint32_t warp_lane = threadIdx.x % WARP_SIZE;
    const uint32_t D = grad_output.size(1);
    const uint32_t S = d_coarse_grid.size(1);
    const uint32_t warp_mask = __ballot_sync(0xffffffff, warp_lane < S);
    if (warp_lane >= S || point_id >= points.size(0)) { return; }

    const scalar_t cp[3] = {points[point_id][0] * coarse_reso, points[point_id][1] * coarse_reso, points[point_id][2] * coarse_reso};
    const scalar_t fp[3] = {cp[0] * fine_reso, cp[1] * fine_reso, cp[2] * fine_reso};

    int32_t cn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    int32_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    scalar_t fn_center[3];   // center of fine-neighbor cell in 'full-grid' coordinates
    scalar_t grad[10];  // TODO: This is hardcoded randomly, allows up to 256 atoms
    scalar_t interp_weights[8];
    calc_interp_weights(interp_weights, fp, fn_center);

    // Load grad_output into shmem
    extern __shared__ __align__(sizeof(double2)) unsigned char smem[];
    scalar_t* shmem_grad_output = reinterpret_cast<scalar_t *>(smem);
    for (int d = warp_lane; d < D; d += min(WARP_SIZE, S)) {
        shmem_grad_output[warp_block_id * D + d] = grad_output[point_id][d];
    }
    __syncwarp(warp_mask);
    //const uint32_t num_atoms_per_warp = (S + WARP_SIZE - 1) / WARP_SIZE;
        fn[0] = clamp(floor2int(fp[0] + OFFSET[i][0]), 0, fine_reso * coarse_reso - 1);
        fn[1] = clamp(floor2int(fp[1] + OFFSET[i][1]), 0, fine_reso * coarse_reso - 1);
        fn[2] = clamp(floor2int(fp[2] + OFFSET[i][2]), 0, fine_reso * coarse_reso - 1);
        fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
        fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
        fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);
        __half2 iw = __float2half2_rn(
            (1.0f - myabs(fp[0] - static_cast<float>(fn[0]) - 0.5f)) *
            (1.0f - myabs(fp[1] - static_cast<float>(fn[1]) - 0.5f)) *
            (1.0f - myabs(fp[2] - static_cast<float>(fn[2]) - 0.5f)));
        int32_t cn_wcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
        int32_t fn_wcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso);
        // load w from coarse_grid to shared mem using all threads in warp
        for (int s = warp_lane * 2; s < S; s += 64) {
            cg_shmem[warp_offset + s >> 1] = __ldg(
                reinterpret_cast<const __half2*>(coarse_grid + cn_wcoo * S + s));
        #pragma unroll 10
        for (int j = 0; j < 10; j++) { grad[j] = 0.0; }
        cn[0] = floor2int(cp[0] + OFFSET[i][0]);
        cn[1] = floor2int(cp[1] + OFFSET[i][1]);
        cn[2] = floor2int(cp[2] + OFFSET[i][2]);
        if (cn[0] < 0 || cn[0] >= coarse_reso ||
            cn[1] < 0 || cn[1] >= coarse_reso ||
            cn[2] < 0 || cn[2] >= coarse_reso) { continue; }
        const int32_t cn_realcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
        // overwrite cn from coarse-grid-coordinates fo full-grid-coordinates
        cn[0] *= fine_reso;
        cn[1] *= fine_reso;
        cn[2] *= fine_reso;
        for (int j = 0; j < 8; j++) {
            fn[0] = floor2int(fp[0] + OFFSET[j][0]);
            fn[1] = floor2int(fp[1] + OFFSET[j][1]);
            fn[2] = floor2int(fp[2] + OFFSET[j][2]);
            fn_center[0] = static_cast<scalar_t>(fn[0]) + 0.5;
            fn_center[1] = static_cast<scalar_t>(fn[1]) + 0.5;
            fn_center[2] = static_cast<scalar_t>(fn[2]) + 0.5;
            // The if-statement also takes care of any out-of-bounds neighbors (ignored)
            if (static_cast<scalar_t>(cn[0]) <= fn_center[0] && static_cast<scalar_t>(cn[0]) + fine_reso >= fn_center[0] &&
                static_cast<scalar_t>(cn[1]) <= fn_center[1] && static_cast<scalar_t>(cn[1]) + fine_reso >= fn_center[1] &&
                static_cast<scalar_t>(cn[2]) <= fn_center[2] && static_cast<scalar_t>(cn[2]) + fine_reso >= fn_center[2])
            {
                const int32_t fn_realcoo = coo2idx(fn[0] - cn[0], fn[1] - cn[1], fn[2] - cn[2], fine_reso);
                for (uint32_t d = 0; d < D; d++) {
                    const scalar_t go_weight = interp_weights[j] * shmem_grad_output[warp_block_id * D + d];
                    for (uint32_t s = warp_lane, s_w = 0; s < S; s += WARP_SIZE, s_w++) {
                        grad[s_w] = myfma(atoms[fn_realcoo][d][s], go_weight, grad[s_w]);
                    }
                }
            }
        }
        __syncwarp();
        for (int s = 0; s < S; s += 2) {
            __half2 atom_weight = warp_lane > D ? __float2half2_rn(0.0f) :
                __halves2half2(__ldg((__half*)atoms + fn_wcoo * S * D + s * D + warp_lane),
                               __ldg((__half*)atoms + fn_wcoo * S * D + (s + 1) * D + warp_lane));
            acc2 = __hfma2(cg_shmem[warp_offset + s >> 1], atom_weight, acc2);
        }
        __syncwarp();
        acc2 = __hmul2(iw, acc2);
    }
    if (warp_lane < D) {
        out[warp_lane] = __low2float(acc2) + __high2float(acc2);
    }
}


template<int32_t POW2_RF>
__global__ void
k_l2_interp_hlf(const c10::Half * __restrict__ coarse_grid,  // Rc^3, S
                const c10::Half * __restrict__ atoms,  // Rf^3, S, D
                const float     * __restrict__ points,  // N, 3
                      float     * __restrict__ out,  // N, D
                const int32_t coarse_reso,
                const int32_t D,
                const int32_t N,
                const int32_t S)
{
    const int32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (point_id >= N) { return; }

    __half2 * coarse_w = shared_memory_proxy<__half2>(); // V1_WARPS_PER_BLOCK * S / 2;

    k_l2_interp_hlf_single_pt<POW2_RF>(
        coarse_grid, atoms, points + point_id * 3, out + point_id * D, coarse_w, coarse_reso, D, S);
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
            Tensor out;
            if (coarse_grid.scalar_type() == at::ScalarType::Half) {
                out = torch::zeros({N, D}, points.options());
            } else {
                out = torch::zeros({N, D}, coarse_grid.options());
            }

            const dim3 grid_size(div_round_up(N, V1_WARPS_PER_BLOCK));
            const dim3 block_size(V1_FWD_BLOCK_SIZE);
            const int32_t shmem = V1_WARPS_PER_BLOCK * S;

            #define CALL_KERNEL(IT, OT, RF)                                                                         \
                k_l2_interp<RF><<<grid_size, block_size, shmem * sizeof(IT), stream.stream()>>>(         \
                    coarse_grid.data_ptr<IT>(),                                                                     \
                    atoms.data_ptr<IT>(),                                                                           \
                    points.data_ptr<float>(),                                                                       \
                    out.data_ptr<OT>(),                                                                             \
                    coarse_reso, D, N, S)
            #define CALL_KERNEL_HALF(IT, OT, RF)                                                                         \
                k_l2_interp_hlf<RF><<<grid_size, block_size, shmem * sizeof(IT), stream.stream()>>>(         \
                    coarse_grid.data_ptr<IT>(),                                                                     \
                    atoms.data_ptr<IT>(),                                                                           \
                    points.data_ptr<float>(),                                                                       \
                    out.data_ptr<OT>(),                                                                             \
                    coarse_reso, D, N, S)

            if (coarse_grid.scalar_type() == at::ScalarType::Half) {
                switch(fine_reso) {
                    case 4:
                        CALL_KERNEL_HALF(c10::Half, float, 2);
                        break;
                    case 8:
                        CALL_KERNEL_HALF(c10::Half, float, 3);
                        break;
                    default:
                        throw std::invalid_argument("fine-resolution must be 4 or 8.");
                }
            } else {
                AT_DISPATCH_FLOATING_TYPES(coarse_grid.scalar_type(), "dispatch_l2interpv1_fwd", [&] {
                    switch(fine_reso) {
                        case 4:
                            CALL_KERNEL(scalar_t, scalar_t, 2);
                            break;
                        case 8:
                            CALL_KERNEL(scalar_t, scalar_t, 3);
                            break;
                        default:
                            throw std::invalid_argument("fine-resolution must be 4 or 8.");
                    }
                });
            }
            #undef CALL_KERNEL
            #undef CALL_KERNEL_HALF
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

            //const dim3 grid_size_dcg(n_blocks_linear(points.size(0), CUDA_THREADS_PER_BLOCK / WARP_SIZE));
            //const dim3 block_size_dcg(CUDA_THREADS_PER_BLOCK);

            //const dim3 grid_size_da(points.size(0));
            //const dim3 block_size_da(round_up(grad_output.size(1), 32), 8);  // D * S

            return {d_coarse_grid, d_atoms, Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }
};



Tensor l2_interp_v1(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &points, const int64_t fine_reso,
                    const int64_t coarse_reso, const double fine_vl, const double coarse_vl)
{
    return L2InterpFunctionv1::apply(coarse_grid, atoms, points, fine_reso, coarse_reso);
}


static auto registry = torch::RegisterOperators()
                        .op("plenoxels::l2_interp_v1", &l2_interp_v1);
