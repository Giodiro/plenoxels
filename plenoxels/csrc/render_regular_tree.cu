#include <cmath>
#include <stdexcept>
#include <tuple>

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/warp/warp_reduce.cuh>

#include "render_util.cuh"
#include "cuda_util.cuh"

template <typename T, size_t N>
using Acc32 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int32_t>;
template <typename T, size_t N>
using Acc64 = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, int64_t>;


const int CUDA_THREADS_PER_BLOCK = 128;
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


template<class scalar_t, uint32_t BASIS_DIM>
__global__ void
trace_ray(
    const Acc32<scalar_t, 2> coarse_grid,
    const Acc32<scalar_t, 3> atoms,
    const Acc32<scalar_t, 2> rays_o,
    const Acc32<scalar_t, 2> rays_d,
    Acc32<scalar_t, 2> out,
    const fast_divmod fast_divmod_fine_reso,
    const uint32_t coarse_reso,
    const uint32_t n_rays,
    const scalar_t * __restrict__ scaling,
    const scalar_t * __restrict__ offset,
    const RenderOptions opt
)
{
    const uint32_t ray_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const uint32_t warp_offset = threadIdx.x / WARP_SIZE;
    const uint32_t warp_lane = threadIdx.x % WARP_SIZE;
    const uint32_t lane_colorgrp = warp_lane / BASIS_DIM;
    const uint32_t lane_colorgrp_id = warp_lane % BASIS_DIM;
    const uint32_t D = atoms.size(2);  // D is also BASIS_DIM * 3 + 1
    typedef cub::WarpReduce<scalar_t> WarpReduce;
	if (ray_id >= n_rays || warp_lane >= D) return;

    // shared memory. This is done to save some register space, no actual sharing occurs.
    __shared__ scalar_t sphfunc_val[CUDA_WARPS_PER_BLOCK][9];
    __shared__ Ray<scalar_t> ray_spec[CUDA_WARPS_PER_BLOCK];
    __shared__ typename WarpReduce::TempStorage cub_storage[CUDA_WARPS_PER_BLOCK];
    // dynamically allocated shmem. This is actually shared
    scalar_t* coarse_w_smem = shared_memory_proxy<scalar_t>();  // CUDA_WARPS_PER_BLOCK * S

    // Setup the ray-spec. Will copy data from rays_o, rays_d
    ray_spec[warp_offset].set(rays_o[ray_id].data(), rays_d[ray_id].data());
    // Spherical harmonics are computed before ray normalization
    calc_sphfunc(/*basis_dim=*/BASIS_DIM, /*dir=*/ray_spec[warp_offset].dir, /*out=*/sphfunc_val[warp_offset]);
    // Finish ray-spec initialization
    ray_find_bounds(ray_spec[warp_offset], scaling, offset, (scalar_t)opt.step_size, (scalar_t)opt.near_plane);
    __syncwarp((1U << D) - 1);

    if (ray_spec[warp_offset].tmin > ray_spec[warp_offset].tmax) {  // Ray doesn't hit box
        out[ray_id][lane_colorgrp] = 1.0f;
        return;
    }

    scalar_t t = ray_spec[warp_offset].tmin;
    scalar_t outv = 0.0f;
    scalar_t log_light_intensity = 0.0f;
    scalar_t sigma, interp_val;
    while (t <= ray_spec[warp_offset].tmax) {
        ray_spec[warp_offset].update_pos(t);

        dictionary_interp<scalar_t>(
            coarse_grid, atoms, /*point=*/ray_spec[warp_offset].pos, /*coarse_w_smem=*/coarse_w_smem,
            /*out=*/&interp_val, fast_divmod_fine_reso, coarse_reso, warp_lane, warp_offset);
        sigma = interp_val;
        // broadcast sigma (stored in last coordinate) to other threads in warp
        __shfl_sync((1U << D) - 1, sigma, /*srcLane=*/D - 1);
        if (sigma > opt.sigma_thresh) {
            interp_val *= sphfunc_val[warp_offset][lane_colorgrp_id]; // bank conflict
            const scalar_t pcnt = ray_spec[warp_offset].world_step * sigma;
            const scalar_t weight = myexp(log_light_intensity) * (1.f - myexp(-pcnt));
            log_light_intensity -= pcnt;

            // The reduction will also happen on the last lane which only holds sigma.
            // The value computed there is ignored.
            scalar_t lane_color_total = WarpReduce(cub_storage[warp_offset]).HeadSegmentedSum(
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
        out[ray_id][lane_colorgrp] = outv;
    }
}


template<class scalar_t>
__device__ void
dictionary_interp(const Acc32<scalar_t, 2> coarse_grid,  // Rc^3, S
                  const Acc32<scalar_t, 3> atoms,        // Rf^3, S, D
                  const scalar_t __restrict__ *point,    // 3
                        scalar_t __restrict__ *coarse_w_smem,  // num warps in block * S
                        scalar_t __restrict__ *out,      // 1
                  const fast_divmod fast_divmod_fine_reso,
                  const uint32_t coarse_reso,
                  const uint32_t warp_lane,
                  const uint32_t warp_offset)
{
    const uint32_t tot_reso = coarse_reso * fast_divmod_fine_reso.d_;
    const uint32_t D = atoms.size(2);
    const uint32_t S = coarse_grid.size(1);
    const scalar_t fp[3] = {point[0] * tot_reso, point[1] * tot_reso, point[2] * tot_reso};

    int32_t cn[3];           // corner of coarse-neighbor cell in 'coarse-grid' coordinates
    int32_t fn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates
    int32_t rfn[3];           // corner of fine-neighbor cell in 'full-grid' coordinates

    scalar_t interpolation_weight;
    *sigma = 0.0f;
    *lane_color = 0.0f;

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

        interp_weight = (1.0f - myabs(fp[0] - static_cast<scalar_t>(fn[0]) - 0.5f)) *
                        (1.0f - myabs(fp[1] - static_cast<scalar_t>(fn[1]) - 0.5f)) *
                        (1.0f - myabs(fp[2] - static_cast<scalar_t>(fn[2]) - 0.5f));
        // load w from coarse_grid to shared mem using all active threads in warp
        for (int s = warp_lane; s < S; s += D) {
            coarse_w_smem[warp_offset * S + s] = coarse_grid[cn_realcoo][s];
        }
        __syncwarp((1U << D) - 1);
        for (int s = 0; s < S; s++) {
            // pseudo: out += coarse_grid[cn][s] * iw[j] * atoms[fn][s][d]
            myfma(coarse_w_smem[warp_offset * S + s] * interp_weight,
                  atoms[fn_realcoo][s][warp_lane],
                  out);
        }
        __syncwarp((1U << D) - 1);
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
                              double offset)
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
            RenderOptions opt = {.step_size = 0.1, .sigma_thresh = 1e-4, .stop_thresh = 1e-4, .near_plane = 0.0};

            ctx->save_for_backward({coarse_grid, atoms});
            ctx->saved_data["rays_o"] = rays_o;
            ctx->saved_data["rays_d"] = rays_o;
            ctx->saved_data["fine_reso"] = fine_reso;
            ctx->saved_data["coarse_reso"] = coarse_reso;

            const uint32_t num_rays = rays_o.size(0);

            auto out = torch::zeros({num_rays, 3}, coarse_grid.options());
            auto scaling_t = torch::tensor({scaling, scaling, scaling}, coarse_grid.options());
            auto offset_t = torch::tensor({offset, offset, offset}, coarse_grid.options());

            const dim3 grid_size(n_blocks_linear(num_rays, CUDA_WARPS_PER_BLOCK));
            const dim3 block_size(CUDA_THREADS_PER_BLOCK);
            const uint32_t shared_mem = CUDA_WARPS_PER_BLOCK * coarse_grid.size(1);

            fast_divmod fast_divmod_fine_reso((int32_t)fine_reso);

            AT_DISPATCH_FLOATING_TYPES(coarse_grid.scalar_type(), "dispatch_l2interpv2_fwd", [&] {
                trace_ray<scalar_t, 1><<<grid_size, block_size, shared_mem * sizeof(scalar_t), stream.stream()>>>(
                    coarse_grid.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    atoms.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    fast_divmod_fine_reso,
                    (uint32_t)coarse_reso,
                    num_rays,
                    scaling_t.data_ptr<scalar_t>(),
                    offset_t.data_ptr<scalar_t>(),
                    opt
                );
            });
            return out;
        }

        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            return {Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        }
};


Tensor dict_tree_render(const Tensor &coarse_grid, const Tensor &atoms, const Tensor &rays_o, const Tensor &rays_d,
                        const int64_t fine_reso, const int64_t coarse_reso, const double scaling, const double offset)
{
    return DictTreeRender::apply(coarse_grid, atoms, rays_o, rays_d, fine_reso, coarse_reso, scaling, offset);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::dict_tree_render", &dict_tree_render);

