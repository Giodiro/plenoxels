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

__device__ __forceinline__ float myfma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ double myfma(double a, double b, double c) { return fma(a, b, c); }
__device__ __forceinline__ float myfloor(float a) { return floorf(a); }
__device__ __forceinline__ double myfloor(double a) { return floor(a); }
__device__ __forceinline__ float myabs(float a) { return fabsf(a); }
__device__ __forceinline__ double myabs(double a) { return fabs(a); }
__device__ __forceinline__ int32_t floor2int(float a) { return __float2int_rd(a); }
__device__ __forceinline__ int32_t floor2int(double a) { return __double2int_rd(a); }

__host__ __device__ int round_up(int num, int multiple) 
{
    return ((num + multiple - 1) / multiple) * multiple;
}


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
        pos[j] = pos[j] * (grid_size - 1);
        pos[j] = min(static_cast<out_t>(grid_size - 1.00001), max(pos[j], 0.0));
        idx[j] = floor2int(pos[j]);
        //pos[j] = pos[j] * (grid_size - 1);
        //pos[j] = min(max(pos[j], 0.0f), static_cast<out_t>(grid_size - 1));
        //n_idx[j] = min(static_cast<int32_t>(pos[j]), grid_size - 2);
        pos[j] -= static_cast<out_t>(idx[j]);
    }
}


__device__ __inline__ int32_t coo2idx(int32_t x, int32_t y, int32_t z, uint32_t grid_size) {
    return x + y * grid_size + z * grid_size * grid_size;//z + y * grid_size + x * grid_size * grid_size;
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

// The code below is based on section 4 Unsigned division of paper https://gmplib.org/~tege/divcnst-pldi94.pdf
// In current ORT, fast_divmod is used for calculating the position of a element in tensor,
// so unsigned integer division from the paper is good enough for ORT. The advantage is that div is very simple,
// then GPU compiler can do loop unroll easilly when divmod is called in a loop.
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h
struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    for (l_ = 0; l_ < 32; l_++)
      if ((1U << l_) >= d_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
  }

  __device__ inline int div(int n) const {
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
  }

  __device__ inline int mod(int n) const {
    return n - div(n) * d_;
  }

  __device__ inline void divmod(int n, int& q, int& r) const {
    q = div(n);
    r = n - q * d_;
  }

  uint32_t d_;  // divisor
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};


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



/* v1 */

template<typename query_t, typename sh_t, typename out_t>
__global__ void k_l2_interp(Acc32<query_t, 2> Q,       // N x S
                            Acc32<sh_t, 3> A,          // S x R^3 x D
                            Acc32<out_t, 2> O,         // N x D
                            Acc32<out_t, 2> positions,  // N x 3
                            const uint32_t grid_size
                           )
{
    const uint32_t point_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t warp_lane = threadIdx.x % 32;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    const uint32_t N = Q.size(0);
    if (warp_lane >= D || point_id >= N) { return; }
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


template<typename query_t, typename sh_t, typename out_t, bool verbose>
__global__ void k_l2_interp_bwd(Acc32<out_t, 2> grad_output,  // N x D
                                Acc32<query_t, 2> DQ,  // N x S
                                Acc32<sh_t, 3> DA,    // S x R^3 x D
                                Acc32<out_t, 2> positions,
                                Acc32<query_t, 2> Q,
                                Acc32<sh_t, 3> A,
                                const int32_t grid_size
                                )
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t point_id = tid / WARP_SIZE;
    const uint32_t warp_block_id = threadIdx.x / WARP_SIZE;
    const uint32_t warp_lane = threadIdx.x % WARP_SIZE;
    const uint32_t S = A.size(0);
    const uint32_t D = A.size(2);
    const uint32_t N = Q.size(0);
    if (warp_lane >= D || point_id >= N) { return; }
    __shared__ typename cub::WarpReduce<out_t>::TempStorage temp_storage[CUDA_WARPS_PER_BLOCK];

    out_t pos[3] = {positions[point_id][0], positions[point_id][1], positions[point_id][2]};
    int32_t n_idx[3];
    unnormalize_pos(pos, n_idx, grid_size);
    const out_t ax = 1.f - pos[0];
    const out_t ay = 1.f - pos[1];
    const out_t az = 1.f - pos[2];
    const uint32_t offx = A.stride(1);
    const uint32_t offy = offx * grid_size;
    const uint32_t offz = offy * grid_size;
    uint32_t A_offset = offx * n_idx[0] + offy * n_idx[1] + offz * n_idx[2] + warp_lane;
    if (verbose) {
        printf("Starting from A_offset=%ld, n_idx[0]=%d, n_idx[1]=%d, n_idx[2]=%d, pos[0]=%f, pos[1]=%f, pos[2]=%f\n", A_offset, n_idx[0], n_idx[1], n_idx[2], positions[point_id][0], positions[point_id][1], positions[point_id][2]);
    }

    sh_t*  A_ptr = A.data();
    sh_t*  DA_ptr = DA.data();
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
        if (verbose) {
            if ((A_offset + offx + offy + offz) >= A.size(0) * A.size(1) * A.size(2)) {
                printf("A_offset out of bounds %ld %ld %ld %ld\n", A_offset, offx, offy, offz);
            }
        }

        dq_temp *= go;
        dq_temp = cub::WarpReduce<out_t>(temp_storage[warp_block_id]).Sum(dq_temp);
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


class L2InterpFunction : public Function<L2InterpFunction> {
    public:
        static Tensor forward(AutogradContext *ctx,
                              Tensor queries,
                              Tensor atoms,
                              Tensor points,
                              bool verbose)
        {
            const at::cuda::CUDAGuard device_guard(queries.device());
            const auto stream = at::cuda::getCurrentCUDAStream();
            ctx->save_for_backward({queries, atoms});
            ctx->saved_data["points"] = points;
            ctx->saved_data["verbose"] = verbose;

            const uint32_t l2_grid_size = (uint32_t)std::cbrt(atoms.size(1));
            auto out = torch::zeros({queries.size(0), atoms.size(2)}, torch::dtype(queries.dtype()).device(queries.device()));
            AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "dispatch_l2interp_fwd", [&] {
                k_l2_interp<scalar_t, scalar_t, scalar_t>
                    <<< n_blocks_linear(queries.size(0), CUDA_THREADS_PER_BLOCK / WARP_SIZE), CUDA_THREADS_PER_BLOCK, 0, stream.stream()>>>
                    (queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
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
            const bool verbose = ctx->saved_data["verbose"].toBool();
            const Tensor grad_output = grad_outputs[0];

            const at::cuda::CUDAGuard device_guard(queries.device());
            const auto stream = at::cuda::getCurrentCUDAStream();

            const int32_t l2_grid_size = (int32_t)std::cbrt(atoms.size(1));
            Tensor d_queries = torch::zeros_like(queries);
            Tensor d_atoms = torch::zeros_like(atoms);
            AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "dispatch_l2interp_bwd", [&] {
                if (verbose) {
                    k_l2_interp_bwd<scalar_t, scalar_t, scalar_t, true>
                        <<< n_blocks_linear(queries.size(0), CUDA_THREADS_PER_BLOCK / WARP_SIZE), CUDA_THREADS_PER_BLOCK, 0, stream.stream()>>>
                        (grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         d_queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         d_atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                         points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                         l2_grid_size);
                } else {
                    k_l2_interp_bwd<scalar_t, scalar_t, scalar_t, false>
                        <<< n_blocks_linear(queries.size(0), CUDA_THREADS_PER_BLOCK / WARP_SIZE), CUDA_THREADS_PER_BLOCK, 0, stream.stream()>>>
                        (grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         d_queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         d_atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                         points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         queries.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                         atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                         l2_grid_size);
                }
            });
            return {d_queries, d_atoms, Tensor(), Tensor()};
        }
};


Tensor l2_interp(const Tensor &queries, const Tensor &atoms, const Tensor &points, const bool verbose) 
{
    return L2InterpFunction::apply(queries, atoms, points, verbose);
}

Tensor l2_interp_v2(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &points, const int64_t fine_reso,
                    const int64_t coarse_reso, const double fine_vl, const double coarse_vl)
{
    return L2InterpFunctionv2::apply(coarse_grid, atoms, points, fine_reso, coarse_reso);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::l2_interp", &l2_interp)
                        .op("plenoxels::l2_interp_v2", &l2_interp_v2);

