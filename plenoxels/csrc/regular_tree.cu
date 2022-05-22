#include <cmath>
#include <tuple>

#include <torch/torch.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include "regular_tree.h"

template <typename T, size_t N>
using Acc32 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int32_t>;
template <typename T, size_t N>
using Acc64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;


constexpr uint32_t n_blocks_linear(uint32_t n_elements, uint32_t n_threads_linear) {
    return (uint32_t)(n_elements + n_threads_linear - 1) / n_threads_linear;
}

/*
 * Linear interpolation
 * implements (1 - w) * a + w * b via a subtraction and a fused multiply-add.
 * TODO: This only works for floats due to use of fmaf.
 */
template<typename T>
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}

template<typename index_t, typename data_t>
__device__ __inline__ float trilerp_one(const index_t * __restrict__ n_idx,
                                        const float   * __restrict__ pos,
                                        const data_t  * __restrict__ data,
                                        const uint32_t stride,
                                        const uint32_t grid_size,
                                        const uint32_t data_idx)
{
    // data: [x][y][z][sh-data]
    const uint32_t offz = stride;            // stride=stride
    const uint32_t offy = grid_size * offz;  // stride=stride * grid_size
    const uint32_t offx = grid_size * offy;  // stride=stride * grid_size * grid_size

    const data_t * __restrict__ data_ptr = data + offx * n_idx[0] +
                                                  offy * n_idx[1] +
                                                  offz * n_idx[2] +
                                                  data_idx;

    const float ix0y0 = lerp(data_ptr[0], data_ptr[offz], pos[2]);            // (1-z) * (x,y,z) + (z) * (x,y,z+1)
    const float ix0y1 = lerp(data_ptr[offy], data_ptr[offy + offz], pos[2]);  // (1-z) * (x,y+1,z) + (z) * (x,y+1,z+1)
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);                             // (1-y) * ix0y0 + (y) * ix0y1
    const float ix1y0 = lerp(data_ptr[offx], data_ptr[offx + offz], pos[2]);  // (1-z) * (x+1,y,z) + (z) * (x+1,y,z+1)
    const float ix1y1 = lerp(data_ptr[offy + offx], data_ptr[offy + offx + offz], pos[2])  // (1-z)*(x+1,y+1,z)+z*(x+1,y+1,z+1)
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
}

template<typename data_t>
__device__ __inline__ float trilerp_precomputed(const float   * __restrict__ pos,
                                                const data_t  * __restrict__ data)
{
    const float ix0y0 = lerp(data[0], data[1], pos[2]);
    const float ix0y1 = lerp(data[2], data[3], pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1y0 = lerp(data[4], data[5], pos[2]);
    const float ix1y1 = lerp(data[6], data[7], pos[2])
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
}


template<typename query_t, typename sh_t>
__global__ void k_l2_interp(Acc32<query_t, 2> Q,       // N x S
                            Acc32<sh_t, 3> A,          // S x R^3 x D
                            Acc32<float, 2> O,         // N x D
                            Acc32<float, 2> positions,  // N x 3
                            const uint32_t grid_size
                           )
{
    const uint32_t point_id = threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    if (warp_lane >= D || point_id > Q.size(0)) { return; }
    float pos[3] = {positions[point_id][0], positions[point_id][1], positions[point_id][2]};
    int32_t n_idx[3];
    for (j = 0; j < 3; j++) {  // this work is repeated unnecessarily for all threads in warp.
        pos[j] = pos[j] * grid_size;
        pos[j] = min(max(pos[j], 0.0f), grid_size - 1.0f);
        n_idx[j] = min(static_cast<int32_t>(pos[j]), grid_size - 2);
        pos[j] -= static_cast<float>(floor);
    }
    sh_t neighbor_data[8] = {0.};
    const uint32_t offz = 1;                 // stride=stride
    const uint32_t offy = offz * grid_size;  // stride=stride * grid_size
    const uint32_t offx = offy * grid_size;  // stride=stride * grid_size * grid_size
    const uint32_t offdata = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2];
    for (int s = 0; s < S; s++) {
        // Load s-th weight from global (maybe async?)
        query_t weight = Q[point_id][s];
        neighbor_data[0] = fmaf(weight, A[s][offdata][warp_lane], neighbor_data[0]);
        neighbor_data[1] = fmaf(weight, A[s][offdata + offz][warp_lane], neighbor_data[1]);
        neighbor_data[2] = fmaf(weight, A[s][offdata + offy][warp_lane], neighbor_data[2]);
        neighbor_data[3] = fmaf(weight, A[s][offdata + offy + offz][warp_lane], neighbor_data[3]);
        neighbor_data[4] = fmaf(weight, A[s][offdata + offx][warp_lane], neighbor_data[4]);
        neighbor_data[5] = fmaf(weight, A[s][offdata + offx + offz][warp_lane], neighbor_data[5]);
        neighbor_data[6] = fmaf(weight, A[s][offdata + offx + offy][warp_lane], neighbor_data[6]);
        neighbor_data[7] = fmaf(weight, A[s][offdata + offx + offy + offz][warp_lane], neighbor_data[7]);
    }
    O[point_id][warp_lane] = trilerp_precomputed(pos, neighbor_data);
}


template<typename query_t, typename sh_t>
__global__ void k_l2_interp_bwd(Acc32<float, 2> grad_output,  // N x D
                                Acc32<query_t, 2> DQ,  // N x S
                                Acc32<sh_t, 3> DA,    // S x R^3 x D
                                Acc32<float, 2> positions,
                                Acc32<query_t, 2> Q,
                                Acc32<sh_t, 3> A,
                                const uint32_t grid_size
                                )
{
    const uint32_t point_id = threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    if (warp_lane >= D || point_id > Q.size(0)) { return; }
    __shared__ typename cub::WarpReduce<float>::TempStorage temp_storage;

    float pos[3] = {positions[point_id][0], positions[point_id][1], positions[point_id][2]};
    int32_t n_idx[3];
    for (j = 0; j < 3; j++) {  // this work is repeated unnecessarily for all threads in warp.
        pos[j] = pos[j] * grid_size;
        pos[j] = min(max(pos[j], 0.0f), grid_size - 1.0f);
        n_idx[j] = min(static_cast<int32_t>(pos[j]), grid_size - 2);
        pos[j] -= static_cast<float>(floor);
    }
    const float ax = 1.f - pos[0];
    const float ay = 1.f - pos[1];
    const float az = 1.f - pos[2];
    const uint32_t offz = A.stride(1);                 // stride=stride
    const uint32_t offy = offz * grid_size;  // stride=stride * grid_size
    const uint32_t offx = offy * grid_size;  // stride=stride * grid_size * grid_size
    uint32_t A_offset = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2] + warp_lane;

    sh_t* __restrict__ A_ptr = &A[0];
    sh_t* __restrict__ DA_ptr = &DA[0];
    const uint32_t A_stride0 = A.stride(0);

    const float go = grad_output[point_id][warp_lane];
    float iw;     // interpolation weight
    uint32_t il;  // interpolation location within 2nd level grid
    float out;
    for (int s = 0; s < S; s++) {
        // Gradient with respect to atoms (DA) is summed over all points
        // Gradient with respect to queries (DQ) is summed over all dimensions (warp lanes)
        iw = ax * ay * az;
        il = A_offset;
        out = iw * A_ptr[il];
        atomicAdd(&DA_ptr[il], iw * go);

        iw = ax * ay * pos[2];
        il = A_offset + offz;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        iw = ax * pos[1] * az;
        il = A_offset + offy;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        iw = ax * pos[1] * pos[2];
        il = A_offset + offy + offz;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        iw = pos[0] * ay * az;
        il = A_offset + offx;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        iw = pos[0] * ay * pos[2];
        il = A_offset + offx + offz;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        iw = pos[0] * pos[1] * az;
        il = A_offset + offx + offy;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        iw = pos[0] * pos[1] * pos[2];
        il = A_offset + offx + offy + offz;
        out = fmaf(iw, A_ptr[il], out);
        atomicAdd(&DA_ptr[il], iw * go);

        out *= go;
        out = cub::WarpReduce<float>(temp_storage).Sum(out);
        if (warp_lane == 0) {
            DQ[point_id][s] = out;
        }
        A_offset += A_stride0;
    }
}

/*
 * PyTorch Wrappers
 */


using torch::autograd::variable_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::Tensor;


class L2InterpFunction : public Function<L2InterpFunction> {
    public:
        static Tensor forward(AutogradContext *ctx,
                              Tensor queries,
                              Tensor atoms,
                              Tensor points)
        {
            const at::cuda::CUDAGuard device_guard(queries.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            ctx->save_for_backward({queries, atoms});
            ctx->saved_data["points"] = points;


            const uint32_t l2_grid_size = (uint32_t)std::sqrt(atoms.size(1));
            auto out = torch::empty({queries.size(0), atoms.size(2)}, torch::dtype(torch::kFloat32).device(queries.device()));
            const uint32_t threads_per_block = 256;
            k_l2_interp<float, float>
                <<< n_blocks_linear(queries.size(0), threads_per_block), threads_per_block, 0, stream.stream()>>>
                (queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 atoms.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 l2_grid_size);
            return out;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
        {
            const at::cuda::CUDAGuard device_guard(queries.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            const auto saved = ctx->get_saved_variables();
            const auto queries = saved[0];
            const auto atoms = saved[1];
            const auto points = ctx->saved_data["points"].toTensor();
            const auto grad_output = grad_outputs[0];

            const uint32_t l2_grid_size = (uint32_t)std::sqrt(atoms.size(1));
            auto d_atoms = torch::empty_like(atoms);
            auto d_queries = torch::empty_like(queries);
            const uint32_t threads_per_block = 256;
            k_l2_interp_bwd<float, float>
                <<< n_blocks_linear(queries.size(0), threads_per_block), threads_per_block, 0, stream.stream()>>>
                (grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 d_queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 d_atoms.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                 atoms.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                 l2_grid_size);
            return {d_queries, d_atoms, Tensor()};
        }
}


torch::Tensor l2_interp(const Tensor &queries, const Tensor &atoms, const Tensor &points)
    return L2InterpFunction::apply(queries, atoms, points)[0];
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::l2_interp", &l2_interp);


torch::Tensor level2_interp(const torch::Tensor &queries, const torch::Tensor &atoms, const torch::Tensor &points)
{
    const at::cuda::CUDAGuard device_guard(queries.device());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const uint32_t l2_grid_size = (uint32_t)std::sqrt(atoms.size(1));
    auto out = torch::empty({queries.size(0), atoms.size(2)}, torch::dtype(torch::kFloat32).device(queries.device()));
    const uint32_t threads_per_block = 256;
    k_l2_interp<float, float>
        <<< n_blocks_linear(queries.size(0), threads_per_block), threads_per_block, 0, stream.stream()>>>
        (queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         atoms.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
         out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         l2_grid_size);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> level2_interp_bwd(const torch::Tensor &grad_output,
                                                           const torch::Tensor &queries
                                                           const torch::Tensor &atoms,
                                                           const torch::Tensor &points)
{
    const at::cuda::CUDAGuard device_guard(queries.device());
    const auto stream = at::cuda::getCurrentCUDAStream();

    const uint32_t l2_grid_size = (uint32_t)std::sqrt(atoms.size(1));
    auto d_atoms = torch::empty_like(atoms);
    auto d_queries = torch::empty_like(queries);
    const uint32_t threads_per_block = 256;
    k_l2_interp_bwd<float, float>
        <<< n_blocks_linear(queries.size(0), threads_per_block), threads_per_block, 0, stream.stream()>>>
        (grad_output.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         d_queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         d_atoms.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
         points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
         atoms.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
         l2_grid_size);
    return std::make_tuple(d_queries, d_atoms);;
}
