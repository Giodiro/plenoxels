#include <cmath>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/warp/warp_reduce.cuh>

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


template<class scalar_t>
__global__ void
__launch_bounds__(CUDA_THREADS_PER_BLOCK)
k_l2_interp_v2(Acc32<scalar_t, 2> coarse_grid,  // Rc^3, S
               Acc32<scalar_t, 3> atoms,        // Rf^3, S, D
               Acc32<scalar_t, 2> points,       // N, 3
               Acc32<scalar_t, 2> out,          // N, D
               const fast_divmod fast_divmod_fine_reso,
               const uint32_t coarse_reso)
{
    extern __shared__ __align__(sizeof(double2)) unsigned char smem[];
    scalar_t* coarse_w = reinterpret_cast<scalar_t *>(smem);  // num warps in block * S
    const uint32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const uint32_t warp_lane = threadIdx.x % WARP_SIZE;
    const uint32_t tot_reso = coarse_reso * fast_divmod_fine_reso.d_;
    const uint32_t D = atoms.size(2);
    const uint32_t S = coarse_grid.size(1);
    const uint32_t warp_offset = (threadIdx.x / WARP_SIZE) * S;  // warp_block_id * S
    const uint32_t warp_mask = __ballot_sync(0xffffffff, warp_lane < D);
    if (warp_lane >= D || point_id >= points.size(0)) { return; }

    const scalar_t fp[3] = {points[point_id][0] * tot_reso, points[point_id][1] * tot_reso, points[point_id][2] * tot_reso};

    int32_t cn[3];           // corner of coarse-neighbor cell in 'coarse-grid' coordinates
    int32_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    int32_t rfn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates

    scalar_t acc = 0.0f;
    scalar_t interpolation_weight;

    int32_t cn_realcoo;
    int32_t fn_realcoo;
    for (int i = 0; i < 8; i++) {
        fn[0] = floor2int(fp[0] + OFFSET[i][0]);
        fn[1] = floor2int(fp[1] + OFFSET[i][1]);
        fn[2] = floor2int(fp[2] + OFFSET[i][2]);
        if (fn[0] < 0 || fn[0] >= tot_reso ||
            fn[1] < 0 || fn[1] >= tot_reso ||
            fn[2] < 0 || fn[2] >= tot_reso) {
            continue;
        }
        fast_divmod_fine_reso.divmod(fn[0], cn[0], rfn[0]);  // fn[0] = fn[0] / fine_reso, cn[0] = fn[0] % fine_reso;
        fast_divmod_fine_reso.divmod(fn[1], cn[1], rfn[1]);
        fast_divmod_fine_reso.divmod(fn[2], cn[2], rfn[2]);
        cn_realcoo = coo2idx(cn[0], cn[1], cn[2], coarse_reso);
        fn_realcoo = coo2idx(rfn[0], rfn[1], rfn[2], fast_divmod_fine_reso.d_);

        interpolation_weight = (1.0f - myabs(fp[0] - static_cast<scalar_t>(fn[0]) - 0.5f)) *
                               (1.0f - myabs(fp[1] - static_cast<scalar_t>(fn[1]) - 0.5f)) *
                               (1.0f - myabs(fp[2] - static_cast<scalar_t>(fn[2]) - 0.5f));
        // load w from coarse_grid to shared mem using all active threads in warp
        for (int s = warp_lane; s < S; s += D) {
            coarse_w[warp_offset + s] = coarse_grid[cn_realcoo][s];
        }
        __syncwarp(warp_mask);
        for (int s = 0; s < S; s++) {
            // pseudo: out += coarse_grid[cn][s] * iw[j] * atoms[fn][s][d]
            acc = myfma(coarse_w[warp_offset + s] * interpolation_weight, atoms[fn_realcoo][s][warp_lane], acc);
        }
        __syncwarp(warp_mask);
    }
    out[point_id][warp_lane] = acc;
}


template<typename scalar_t>
__global__ void k_l2_interp_v2_d_a(Acc32<scalar_t, 2> grad_output,   // N, D
                                    Acc32<scalar_t, 2> coarse_grid,  // Rc^3, S
                                    Acc32<scalar_t, 3> d_atoms,      // Rf^3, S, D
                                    Acc32<scalar_t, 2> points,       // N, 3
                                    const uint32_t fine_reso,
                                    const uint32_t coarse_reso
                                   )
{
    const uint32_t point_id = blockIdx.x;
    const uint32_t dim_start_id = threadIdx.x;
    const uint32_t num_dims_in_block = blockDim.x;
    const uint32_t atom_start_id = threadIdx.y;
    const uint32_t num_atoms_in_block = blockDim.y;

    const uint32_t S = coarse_grid.size(1);
    const uint32_t D = grad_output.size(1);
    if (point_id >= grad_output.size(0) || atom_start_id >= S || dim_start_id >= D) { return; }

    const scalar_t cp[3] = {points[point_id][0] * coarse_reso, points[point_id][1] * coarse_reso, points[point_id][2] * coarse_reso};
    const scalar_t fp[3] = {cp[0] * fine_reso, cp[1] * fine_reso, cp[2] * fine_reso};

    int32_t cn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    int32_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    scalar_t fn_center[3];   // center of fine-neighbor cell in 'full-grid' coordinates
    scalar_t interp_weights[8];
    calc_interp_weights(interp_weights, fp, fn_center);

    for (int i = 0; i < 8; i++) {
        cn[0] = floor2int(cp[0] + OFFSET[i][0]);
        cn[1] = floor2int(cp[1] + OFFSET[i][1]);
        cn[2] = floor2int(cp[2] + OFFSET[i][2]);
        if (cn[0] < 0 || cn[0] >= coarse_reso ||
            cn[1] < 0 || cn[1] >= coarse_reso ||
            cn[2] < 0 || cn[2] >= coarse_reso) {
            continue;
        }
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
            if (static_cast<scalar_t>(cn[0]) <= fn_center[0] && static_cast<scalar_t>(cn[0]) + fine_reso >= fn_center[0] &&
                static_cast<scalar_t>(cn[1]) <= fn_center[1] && static_cast<scalar_t>(cn[1]) + fine_reso >= fn_center[1] &&
                static_cast<scalar_t>(cn[2]) <= fn_center[2] && static_cast<scalar_t>(cn[2]) + fine_reso >= fn_center[2])
            {
                const int32_t fn_realcoo = coo2idx(fn[0] - cn[0], fn[1] - cn[1], fn[2] - cn[2], fine_reso);
                for (uint32_t s = atom_start_id; s < S; s += num_atoms_in_block) {
                    scalar_t cg_s = coarse_grid[cn_realcoo][s] * interp_weights[j];
                    for (uint32_t d = dim_start_id; d < D; d += num_dims_in_block) {
                        atomicAdd(
                            &d_atoms[fn_realcoo][s][d], grad_output[point_id][d] * cg_s
                        );
                    }
                }
            }
        }
    }
}


template<typename scalar_t>
__global__ void k_l2_interp_v2_d_cg2(Acc32<scalar_t, 2> grad_output,   // N, D
                                    Acc32<scalar_t, 2> d_coarse_grid, // Rc^3, S
                                    Acc32<scalar_t, 3> atoms,         // Rf^3, D, S
                                    Acc32<scalar_t, 2> points,        // N, 3
                                    const uint32_t fine_reso,
                                    const uint32_t coarse_reso
                                   )
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
    scalar_t grad[8];  // TODO: This is hardcoded randomly, allows up to 256 atoms
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

    for (int i = 0; i < 8; i++) {
        #pragma unroll 8
        for (int j = 0; j < 8; j++) { grad[j] = 0.0; }
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
        for (uint32_t s = warp_lane, s_w = 0; s < S; s += WARP_SIZE, s_w++) {
            atomicAdd(&d_coarse_grid[cn_realcoo][s], grad[s_w]);
        }
    }
}




/*
 * PyTorch Wrappers
 */


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

            auto out = torch::zeros({points.size(0), atoms.size(2)}, torch::dtype(atoms.dtype()).device(atoms.device()));
            const dim3 grid_size(n_blocks_linear(points.size(0), CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const uint32_t shared_mem = CUDA_WARPS_PER_BLOCK * coarse_grid.size(1);

            fast_divmod fast_divmod_fine_reso((int32_t)fine_reso);
            AT_DISPATCH_FLOATING_TYPES(coarse_grid.scalar_type(), "dispatch_l2interpv2_fwd", [&] {
                k_l2_interp_v2<scalar_t><<<grid_size, block_size, shared_mem * sizeof(scalar_t), stream.stream()>>>(
                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
//                    (uint32_t)fine_reso,
                    fast_divmod_fine_reso,
                    (uint32_t)coarse_reso
                );
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

            Tensor atoms_t = atoms.transpose(1, 2).contiguous();  // Rf^3, D, S
            AT_DISPATCH_FLOATING_TYPES(coarse_grid.scalar_type(), "dispatch_l2interpv2_bwd", [&] {
                const uint32_t shared_mem = CUDA_WARPS_PER_BLOCK * atoms.size(2);  // D
                k_l2_interp_v2_d_cg2<scalar_t><<<grid_size_dcg, block_size_dcg, shared_mem * sizeof(scalar_t), stream.stream()>>>(
                    grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    d_coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    atoms_t.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    (uint32_t)fine_reso, (uint32_t)coarse_reso
                );
                k_l2_interp_v2_d_a<scalar_t><<<grid_size_da, block_size_da, 0, stream.stream()>>>(
                    grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    d_atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    (uint32_t)fine_reso, (uint32_t)coarse_reso
                );
            });
            //std::cout << "grad_output " << grad_output.min() << " " << grad_output.max() << "\n";
            //std::cout << "d_coarse_grid" << d_coarse_grid.min() << " " << d_coarse_grid.max() << "\n";
            //std::cout << "d_atoms " << d_atoms.min() << " " << d_atoms.max() << "\n";
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

