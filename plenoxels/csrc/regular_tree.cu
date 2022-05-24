#include <cmath>
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


constexpr uint32_t n_blocks_linear(uint32_t n_elements, uint32_t n_threads_linear) {
    return (uint32_t)(n_elements + n_threads_linear - 1) / n_threads_linear;
}

__host__ __device__ __forceinline__ float myfma(float a, float b, float c) { return fmaf(a, b, c); }
__host__ __device__ __forceinline__ double myfma(double a, double b, double c) { return fma(a, b, c); }
__host__ __device__ __forceinline__ float myfloor(float a) { return floorf(a); }
__host__ __device__ __forceinline__ double myfloor(double a) { return floor(a); }
/*
 * Linear interpolation
 * implements (1 - w) * a + w * b via a subtraction and a fused multiply-add.
 * TODO: This only works for floats due to use of fmaf.
 */
template<typename T>
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return myfma(w, b - a, a);
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
    const float ix1y1 = lerp(data_ptr[offy + offx], data_ptr[offy + offx + offz], pos[2]);  // (1-z)*(x+1,y+1,z)+z*(x+1,y+1,z+1)
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
}

template<typename data_t, typename out_t>
__device__ __inline__ out_t trilerp_precomputed(const out_t   * __restrict__ pos,
                                                const data_t  * __restrict__ data)
{
    const out_t ix0y0 = lerp(static_cast<out_t>(data[0]), static_cast<out_t>(data[1]), pos[0]);
    const out_t ix0y1 = lerp(static_cast<out_t>(data[2]), static_cast<out_t>(data[3]), pos[0]);
    const out_t ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const out_t ix1y0 = lerp(static_cast<out_t>(data[4]), static_cast<out_t>(data[5]), pos[0]);
    const out_t ix1y1 = lerp(static_cast<out_t>(data[6]), static_cast<out_t>(data[7]), pos[0]);
    const out_t ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[2]);
}

template<typename out_t>
__device__ __inline__ void unnormalize_pos(out_t * __restrict__ pos,
                                           int32_t * __restrict__ idx,
                                           const uint32_t grid_size)
{
    #pragma unroll 3
    for (int j = 0; j < 3; j++) {  // this work is repeated unnecessarily for all threads in warp.
        pos[j] = pos[j] * grid_size - 0.5;
        pos[j] = min(static_cast<out_t>(grid_size - 1), max(pos[j], 0.0));
        idx[j] = static_cast<int32_t>(myfloor(pos[j]));
        //pos[j] = pos[j] * (grid_size - 1);
        //pos[j] = min(max(pos[j], 0.0f), static_cast<out_t>(grid_size - 1));
        //n_idx[j] = min(static_cast<int32_t>(pos[j]), grid_size - 2);
        pos[j] -= static_cast<out_t>(idx[j]);
    }
}


template<typename query_t, typename sh_t, typename out_t>
__global__ void k_l2_interp(Acc64<query_t, 2> Q,       // N x S
                            Acc32<sh_t, 3> A,          // S x R^3 x D
                            Acc32<out_t, 2> O,         // N x D
                            Acc32<out_t, 2> positions,  // N x 3
                            const uint32_t grid_size
                           )
{
    const uint32_t point_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    if (warp_lane >= D || point_id >= Q.size(0)) { return; }
    //printf("blockIdx.x %d, threadIdx.x %d, point %d\n", blockIdx.x, threadIdx.x, point_id);
    out_t pos[3] = {positions[point_id][0], positions[point_id][1], positions[point_id][2]};
    int32_t n_idx[3];
    unnormalize_pos(pos, n_idx, grid_size);
    sh_t neighbor_data[8] = {0.};
    const uint32_t offx = 1;                 // stride=stride
    const uint32_t offy = offx * grid_size;  // stride=stride * grid_size
    const uint32_t offz = offy * grid_size;  // stride=stride * grid_size * grid_size
    const uint32_t offdata = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2];
    for (int s = 0; s < S; s++) {
        // Load s-th weight from global
        const sh_t weight = static_cast<sh_t>(Q[point_id][s]);
        neighbor_data[0] = myfma(weight, A[s][offdata][warp_lane], neighbor_data[0]);
        neighbor_data[1] = myfma(weight, A[s][offdata + offx][warp_lane], neighbor_data[1]);
        neighbor_data[2] = myfma(weight, A[s][offdata + offy][warp_lane], neighbor_data[2]);
        neighbor_data[3] = myfma(weight, A[s][offdata + offy + offx][warp_lane], neighbor_data[3]);
        neighbor_data[4] = myfma(weight, A[s][offdata + offz][warp_lane], neighbor_data[4]);
        neighbor_data[5] = myfma(weight, A[s][offdata + offz + offx][warp_lane], neighbor_data[5]);
        neighbor_data[6] = myfma(weight, A[s][offdata + offz + offy][warp_lane], neighbor_data[6]);
        neighbor_data[7] = myfma(weight, A[s][offdata + offz + offy + offx][warp_lane], neighbor_data[7]);
    }
    O[point_id][warp_lane] = trilerp_precomputed(pos, neighbor_data);
}

template<typename query_t, typename sh_t, typename out_t>
__global__ void k_l2_interp_dA(Acc32<out_t, 2> grad_output,  // N x D
                               Acc32<sh_t, 3> DA,    // S x R^3 x D
                               Acc32<out_t, 2> positions,
                               Acc32<query_t, 2> Q,
                               const uint32_t grid_size
                               )
{
    const uint32_t atom_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t point_id = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = DA.size(0);
    const uint32_t D = DA.size(2);
    const uint32_t N = positions.size(0);
    if (warp_lane >= D || atom_id >= S || point_id >= N) { return; }

    const uint32_t offx = 1;                 // stride=stride
    const uint32_t offy = offx * grid_size;  // stride=stride * grid_size
    const uint32_t offz = offy * grid_size;  // stride=stride * grid_size * grid_size

    out_t pos[3];
    int32_t n_idx[3];
    pos[0] = positions[point_id][0];
    pos[1] = positions[point_id][1];
    pos[2] = positions[point_id][2];
    unnormalize_pos(pos, n_idx, grid_size);
    const uint32_t offdata = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2];
    const out_t ax = 1.f - pos[0];
    const out_t ay = 1.f - pos[1];
    const out_t az = 1.f - pos[2];
    const out_t go = grad_output[point_id][warp_lane];
    const sh_t weight = static_cast<sh_t>(Q[point_id][atom_id]) * go;
    DA[atom_id][offdata][warp_lane] += static_cast<sh_t>(ax * ay * az * weight);
    DA[atom_id][offdata + offx][warp_lane] += static_cast<sh_t>(az * ay * pos[0] * weight);
    DA[atom_id][offdata + offy][warp_lane] += static_cast<sh_t>(az * pos[1] * ax * weight);
    DA[atom_id][offdata + offy + offx][warp_lane] += static_cast<sh_t>(az * pos[1] * pos[0] * weight);
    DA[atom_id][offdata + offz][warp_lane] += static_cast<sh_t>(pos[2] * ay * ax * weight);
    DA[atom_id][offdata + offz + offx][warp_lane] += static_cast<sh_t>(pos[2] * ay * pos[0] * weight);
    DA[atom_id][offdata + offz + offy][warp_lane] += static_cast<sh_t>(pos[2] * pos[1] * ax * weight);
    DA[atom_id][offdata + offz + offy + offx][warp_lane] += static_cast<sh_t>(pos[2] * pos[1] * pos[0] * weight);
}


template<typename query_t, typename sh_t, typename out_t>
__global__ void k_l2_interp_dQ(Acc32<out_t, 2> grad_output,  // N x D
                               Acc32<query_t, 2> DQ,  // N x S
                               Acc32<out_t, 2> positions,
                               Acc32<sh_t, 3> A,
                               const uint32_t grid_size
                               )
{
    const uint32_t point_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    if (warp_lane >= D || point_id >= DQ.size(0)) { return; }
    __shared__ typename cub::WarpReduce<out_t>::TempStorage temp_storage;

    out_t pos[3] = {positions[point_id][0], positions[point_id][1], positions[point_id][2]};
    int32_t n_idx[3];
    unnormalize_pos(pos, n_idx, grid_size);
    const out_t ax = 1.f - pos[0];
    const out_t ay = 1.f - pos[1];
    const out_t az = 1.f - pos[2];
    const uint32_t offx = A.stride(1);                 // stride=stride
    const uint32_t offy = offx * grid_size;  // stride=stride * grid_size
    const uint32_t offz = offy * grid_size;  // stride=stride * grid_size * grid_size
    uint32_t A_offset = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2] + warp_lane;

    sh_t* __restrict__  A_ptr = A.data();
    const uint32_t A_stride0 = A.stride(0);

    const out_t go = grad_output[point_id][warp_lane];
    out_t dq_temp;
    for (int s = 0; s < S; s++) {
        // Gradient with respect to atoms (DA) is summed over all points
        // Gradient with respect to queries (DQ) is summed over all dimensions (warp lanes)
        dq_temp =       ax * ay * az *            A_ptr[A_offset];
        dq_temp = myfma(az * ay * pos[0],         A_ptr[A_offset + offx],               dq_temp);
        dq_temp = myfma(az * pos[1] * ax,         A_ptr[A_offset + offy],               dq_temp);
        dq_temp = myfma(az * pos[1] * pos[0],     A_ptr[A_offset + offy + offx],        dq_temp);
        dq_temp = myfma(pos[2] * ay * ax,         A_ptr[A_offset + offz],               dq_temp);
        dq_temp = myfma(pos[2] * ay * pos[0],     A_ptr[A_offset + offz + offx],        dq_temp);
        dq_temp = myfma(pos[2] * pos[1] * ax,     A_ptr[A_offset + offz + offy],        dq_temp);
        dq_temp = myfma(pos[2] * pos[1] * pos[0], A_ptr[A_offset + offx + offy + offz], dq_temp);
        dq_temp *= go;
        dq_temp = cub::WarpReduce<out_t>(temp_storage).Sum(dq_temp, D);
        if (warp_lane == 0) {
            DQ[point_id][s] = static_cast<query_t>(dq_temp);
        }
        A_offset += A_stride0;
    }
}

template<typename query_t, typename sh_t, typename out_t>
__global__ void k_l2_interp_bwd(Acc32<out_t, 2> grad_output,  // N x D
                                Acc64<query_t, 2> DQ,  // N x S
                                Acc32<sh_t, 3> DA,    // S x R^3 x D
                                Acc32<out_t, 2> positions,
                                Acc32<query_t, 2> Q,
                                Acc32<sh_t, 3> A,
                                const uint32_t grid_size
                                )
{
    const uint32_t point_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    if (warp_lane >= D || point_id >= Q.size(0)) { return; }
    __shared__ typename cub::WarpReduce<out_t>::TempStorage temp_storage;

    out_t pos[3] = {positions[point_id][0], positions[point_id][1], positions[point_id][2]};
    int32_t n_idx[3];
    unnormalize_pos(pos, n_idx, grid_size);
    const out_t ax = 1.f - pos[0];
    const out_t ay = 1.f - pos[1];
    const out_t az = 1.f - pos[2];
    const uint32_t offx = A.stride(1);                 // stride=stride
    const uint32_t offy = offx * grid_size;  // stride=stride * grid_size
    const uint32_t offz = offy * grid_size;  // stride=stride * grid_size * grid_size
    uint32_t A_offset = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2] + warp_lane;

    sh_t* __restrict__  A_ptr = A.data();
    sh_t* __restrict__ DA_ptr = DA.data();
    const uint32_t A_stride0 = A.stride(0);

    const out_t go = grad_output[point_id][warp_lane];
    out_t iw;     // interpolation weight
    uint32_t il;  // interpolation location within 2nd level grid
    out_t dq_temp;
    for (int s = 0; s < S; s++) {
        // Gradient with respect to atoms (DA) is summed over all points
        // Gradient with respect to queries (DQ) is summed over all dimensions (warp lanes)
        const out_t weight = static_cast<out_t>(Q[point_id][s]) * go;
        iw = ax * ay * az;
        il = A_offset;
        dq_temp = iw * A_ptr[il];
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = az * ay * pos[0];
        il = A_offset + offx;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = az * pos[1] * ax;
        il = A_offset + offy;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = az * pos[1] * pos[0];
        il = A_offset + offy + offx;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = pos[2] * ay * ax;
        il = A_offset + offz;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = pos[2] * ay * pos[0];
        il = A_offset + offz + offx;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = pos[2] * pos[1] * ax;
        il = A_offset + offz + offy;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        iw = pos[2] * pos[1] * pos[0];
        il = A_offset + offx + offy + offz;
        dq_temp = myfma(iw, A_ptr[il], dq_temp);
        atomicAdd(&DA_ptr[il], static_cast<sh_t>(iw * weight));

        dq_temp *= go;
        dq_temp = cub::WarpReduce<out_t>(temp_storage).Sum(dq_temp);
        if (warp_lane == 0) {
            DQ[point_id][s] = static_cast<query_t>(dq_temp);
        }
        A_offset += A_stride0;
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


            const uint32_t l2_grid_size = (uint32_t)std::cbrt(atoms.size(1));
            auto out = torch::zeros({queries.size(0), atoms.size(2)}, torch::dtype(queries.dtype()).device(queries.device()));
            const uint32_t threads_per_block = 512;
            AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "dispatch_l2interp_fwd", [&] {
                k_l2_interp<scalar_t, scalar_t, scalar_t>
                    <<< n_blocks_linear(queries.size(0), threads_per_block / 32), threads_per_block, 0, stream.stream()>>>
                    (queries.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                     out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     l2_grid_size);
            });
            return out;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
        {
            const auto saved = ctx->get_saved_variables();
            const Tensor queries = saved[0];
            const Tensor atoms = saved[1];
            const Tensor points = ctx->saved_data["points"].toTensor();
            const Tensor grad_output = grad_outputs[0];

            const at::cuda::CUDAGuard device_guard(queries.device());
            const auto stream = at::cuda::getCurrentCUDAStream();

            const uint32_t l2_grid_size = (uint32_t)std::cbrt(atoms.size(1));
            Tensor d_queries = torch::zeros_like(queries);
            Tensor d_atoms = torch::zeros_like(atoms);
            const uint32_t threads_per_block = 512;
            AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "dispatch_l2interp_bwd", [&] {
                k_l2_interp_bwd<scalar_t, scalar_t, scalar_t>
                    <<< n_blocks_linear(queries.size(0), threads_per_block / 32), threads_per_block, 0, stream.stream()>>>
                    (grad_output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     d_queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     d_atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                     points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                     l2_grid_size);
/*                k_l2_interp_dA<scalar_t, scalar_t, scalar_t>
                    <<< n_blocks_linear(atoms.size(0), 32 / 32), 32, 0, stream.stream()>>>
                    (grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     d_atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                     points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     l2_grid_size);
                k_l2_interp_dQ<scalar_t, scalar_t, scalar_t>
                    <<< n_blocks_linear(queries.size(0), threads_per_block / 32), threads_per_block, 0, stream.stream()>>>
                    (grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     d_queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                     atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                     l2_grid_size);*/
            });
            return {d_queries, d_atoms, Tensor()};
        }
};


Tensor l2_interp(const Tensor &queries, const Tensor &atoms, const Tensor &points) 
{
    return L2InterpFunction::apply(queries, atoms, points);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::l2_interp", &l2_interp);

