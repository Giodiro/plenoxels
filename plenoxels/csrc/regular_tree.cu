#include <cmath>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_reduce.cuh>

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

template<class scalar_t>
__device__ __inline__ void
calc_interp_weights(scalar_t * __restrict__ weights_out,
                    const scalar_t * __restrict__ point,
                    scalar_t * __restrict__ scratch)
{
    // Interpolation weight for fine coordinates to the center of top-left cell.
    scratch[0] = point[0] - 0.5f - myfloor(point[0] - 0.5f);
    scratch[1] = point[1] - 0.5f - myfloor(point[1] - 0.5f);
    scratch[2] = point[2] - 0.5f - myfloor(point[2] - 0.5f);
    weights_out[7] = scratch[0]         * scratch[1]         * scratch[2];
    weights_out[6] = scratch[0]         * scratch[1]         * (1.0 - scratch[2]);
    weights_out[5] = scratch[0]         * (1.0 - scratch[1]) * scratch[2];
    weights_out[4] = scratch[0]         * (1.0 - scratch[1]) * (1.0 - scratch[2]);
    weights_out[3] = (1.0 - scratch[0]) * scratch[1]         * scratch[2];
    weights_out[2] = (1.0 - scratch[0]) * scratch[1]         * (1.0 - scratch[2]);
    weights_out[1] = (1.0 - scratch[0]) * (1.0 - scratch[1]) * scratch[2];
    weights_out[0] = (1.0 - scratch[0]) * (1.0 - scratch[1]) * (1.0 - scratch[2]);
}


#define FWD_BLOCK_SIZE_X 32
#define FWD_BLOCK_SIZE_Y 16
#define NUM_POINTS_PER_THREAD 16
#define INNER_POINTS_PER_THREAD 4


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

    //int32_t cn[3];           // corner of coarse-neighbor cell in 'coarse-grid' coordinates
    scalar_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    //int32_t rfn[3];          // corner of fine-neighbor cell in 'full-grid' coordinates

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
    int32_t cn_tmp, fn_tmp;
    for (int p = 0; p < NUM_POINTS_PER_THREAD; p += INNER_POINTS_PER_THREAD) {
        int j = 0;
        for (; j < INNER_POINTS_PER_THREAD; j++) {
            int point_id = start_point_id + p * INNER_POINTS_PER_THREAD + j;
            if (point_id >= out.size(1)) break;
            accs[j] = 0.0f;
            scalar_t fp[3] = {points[point_id][0] * fine_reso * coarse_reso, points[point_id][1] * fine_reso * coarse_reso, points[point_id][2] * fine_reso * coarse_reso};
            for (int i = 0; i < 8; i++) {
                fn[0] = clamp(floor2int(fp[0] + OFFSET[i][0]), 0, fine_reso * coarse_reso - 1);
                //fast_divmod_pow2<POW2_RF>(fn[0], cn[0], rfn[0]);
                fast_divmod_pow2<POW2_RF>(fn[0], cn_realcoo, fn_realcoo);
                fn[1] = clamp(floor2int(fp[1] + OFFSET[i][1]), 0, fine_reso * coarse_reso - 1);
                //fast_divmod_pow2<POW2_RF>(fn[1], cn[1], rfn[1]);
                fast_divmod_pow2<POW2_RF>(fn[1], cn_tmp, fn_tmp);
                cn_realcoo += cn_tmp * coarse_reso;
                fn_realcoo += fn_tmp * fine_reso;
                fn[2] = clamp(floor2int(fp[2] + OFFSET[i][2]), 0, fine_reso * coarse_reso - 1);
                //fast_divmod_pow2<POW2_RF>(fn[2], cn[2], rfn[2]);
                fast_divmod_pow2<POW2_RF>(fn[2], cn_tmp, fn_tmp);
                cn_realcoo += cn_tmp * coarse_reso * coarse_reso;
                fn_realcoo += fn_tmp * fine_reso * fine_reso - blockIdx.z * FWD_BLOCK_SIZE_Y;
                //cn_realcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
                //fn_realcoo = coo2idx(rfn[0], rfn[1], rfn[2], fine_reso) - blockIdx.z * FWD_BLOCK_SIZE_Y;

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
                atomicAdd(&out[dim_id][start_point_id + p * INNER_POINTS_PER_THREAD + k], accs[k]);
                //atomicAdd(&out[dim_id][blockIdx.x * FWD_BLOCK_SIZE_Y * NUM_POINTS_PER_THREAD + threadIdx.y * NUM_POINTS_PER_THREAD + p * INNER_POINTS_PER_THREAD + k], accs[k]);
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


Tensor l2_interp_v2(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &points, const int64_t fine_reso,
                    const int64_t coarse_reso, const double fine_vl, const double coarse_vl)
{
    return L2InterpFunctionv2::apply(coarse_grid, atoms, points, fine_reso, coarse_reso);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::l2_interp_v2", &l2_interp_v2);
