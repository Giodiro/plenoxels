#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using torch::autograd::tensor_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::Tensor;
using at::TensorBase;

#define CHECK_BIT(var,pos) (((var)>>(pos)) & 1)

namespace {
  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_4d_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,   // [N, C, Hi, Wi, ...]
      TensorInfo<scalar_t, index_t> grid,    // [N, Ho, Wo, O]
      TensorInfo<scalar_t, index_t> output,  // [N, C, Ho, Wo]
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners)
  {
    const uint_t DIM = static_cast<uint_t>(input.dim - 2);
    index_t C = input.sizes[1];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t grid_sN = grid.strides[0];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t grid_sCoor = grid.strides[DIM + 1];
    const uint_t DIM_MASK = ~((~0) << DIM);
    const uint_t N_COMBOS = 1 << DIM;

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      auto out_ptr = output.data;         // pointer to output tensor at 'index'
      index_t grid_offset = n * grid_sN;  // offset for grid-pointer at 'index'
      // calculate out_ptr and grid_offset by looping over the spatial dimensions (from last to first)
      index_t mul = 1;
      for (int d = 0; d < DIM; d++) {
        const index_t gridcoo_d = ((index / mul) % grid.sizes[DIM - d]);
        grid_offset += gridcoo_d * grid.strides[DIM - d];
        out_ptr += gridcoo_d * output.strides[DIM - d + 1];
        mul *= grid.sizes[DIM - d];
      }
      const index_t n = index / mul;
      grid_offset += n * grid_sN;
      out_ptr += n * out_sN;
      auto inp_ptr = input.data + n * inp_sN;  // pointer to input tensor for the current 'batch' (first-dimension)

      scalar_t coo[DIM];  // the coordinate at 'index' loaded from grid
      index_t c0[DIM];    // floored `coo`
      for (uint_t d = 0; d < DIM; d++) {
        coo[d] = at::native::grid_sampler_compute_source_index(
          grid.data[grid_offset + d * grid_sCoor],
          input.sizes[DIM + 1 - d],
          padding_mode,
          align_corners);
        c0[d] = static_cast<index_t>(::floor(coo[d]));
      }

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        const scalar_t minus_one = static_cast<scalar_t>(-1.0);
        const scalar_t plus_one = static_cast<scalar_t>(1.0);
        // get surfaces to each neighbor:
        scalar_t neighbor_weights[N_COMBOS];
        for (uint_t i = 0; i < N_COMBOS; i++) {
          neighbor_weights[i] = plus_one;
          for (uint_t d = 0; d < DIM; d++) {
            // consider coordinates xyzq, where the base-corner is x_0000,y_0000,z_0000,q_0000.
            // the other corners can be obtained by simply adding 1 to the appropriate coordinate. For example:
            // x_1010 = x_0000 + 1; x_0110 = x_0000 and so on.
            // w_0000 = (x_1111 - x) * (y_1111 - y) * (z_1111 - y) * (q_1111 - q)
            // w_0010 = (x_1101 - x) * (y_1101 - y) * (z - z_1101) * (q_1101 - q)
            // w_1110 = (x - x_0001) * (y - y_0001) * (z - z_0001) * (q_0001 - q)
            // the sign of term d for each weight is determined by whether bit d is set
            neighbor_weights[i] *=
              (CHECK_BIT(i, d) ? minus_one : plus_one)  // if bit d is set in i then we have a minus sign
              * (static_cast<scalar_t>((c0[d] + CHECK_BIT(~i, d))) - coo[d]);   // the coordinate is obtained by adding 1 if the right bit is set in ~i.
          }
        }

        for (index_t c = 0; c < C; ++c, inp_ptr += inp_sC, out_ptr += out_sC) {
          for (uint_t i = 0; i < N_COMBOS; i++) {
            // the first loop is done to calculate the input pointer
            index_t inp_ptr_offset = 0;
            bool in_bounds = true;
            for (int d = 0; d < DIM; d++) {
              index_t coo_i_d = c0[d] + CHECK_BIT(i, d);
              if (coo_i_d < 0 || coo_i_d >= input.sizes[DIM + 1 - d]) {
                in_bounds = false;
                break;
              }
              inp_ptr_offset += coo_i_d * input.strides[DIM + 1 - d];
            }
            if (in_bounds) {
              *out_ptr += inp_ptr[inp_ptr_offset] * neighbor_weights[i];
            }
          }
        }
      }
    }
  }
}


void launch_grid_sampler_nd_forward_kernel(
    const Tensor &output, const Tensor &input, const Tensor &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
//  at::native::check_grid_sampler_common(input, grid);
//  check_grid_sampler_4d(input, grid, interpolation_mode);

  auto N = input.size(0);
  int64_t count = N;
  for (int d = 1; d < grid.dim - 1; d++) {
    count *= grid.size(d);
  }

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_nd_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_nd_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_4d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::grid_sample_nd", &launch_grid_sampler_nd_forward_kernel);
