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


namespace {
  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_premul_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,   // [N, C, H, W]
      TensorInfo<scalar_t, index_t> grid,    // [N, H, W, 2]
      TensorInfo<scalar_t, index_t> output,  // [C, H, W] initialized to 1
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {
    index_t N = input.sizes[0];
    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t out_sC = output.strides[0];
    index_t out_sH = output.strides[1];
    index_t out_sW = output.strides[2];

    // nthreads : H * W
    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = index / out_W;

      for (index_t n = 0; n < N; n++) {
        const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        scalar_t x = grid.data[grid_offset];
        scalar_t y = grid.data[grid_offset + grid_sCoor];

        scalar_t ix = at::native::grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
        scalar_t iy = at::native::grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          // get NE, NW, SE, SW pixel values from (x, y)
          index_t ix_nw = static_cast<index_t>(::floor(ix));
          index_t iy_nw = static_cast<index_t>(::floor(iy));
          index_t ix_ne = ix_nw + 1;
          index_t iy_ne = iy_nw;
          index_t ix_sw = ix_nw;
          index_t iy_sw = iy_nw + 1;
          index_t ix_se = ix_nw + 1;
          index_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

          // calculate bilinear weighted pixel value and set output pixel
          auto inp_ptr_NC = input.data + n * inp_sN;
          // The output pointer doesn't change within the loop over N.
          auto out_ptr_CHW = output.data + h * out_sH + w * out_sW;
          scalar_t out_val = static_cast<scalar_t>(0);
          for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_CHW += out_sC) {
            if (at::native::within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
              out_val += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
            }
            if (at::native::within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
              out_val += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
            }
            if (at::native::within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
              out_val += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
            }
            if (at::native::within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
              out_val += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
            }
          }
          *out_ptr_CHW *= out_val;
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
          index_t ix_nearest = static_cast<index_t>(::round(ix));
          index_t iy_nearest = static_cast<index_t>(::round(iy));

          // assign nearest neighor pixel value to output pixel
          auto inp_ptr_NC = input.data + n * inp_sN;
          auto out_ptr_CHW = output.data + h * out_sH + w * out_sW;
          for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_CHW += out_sC) {
            if (at::native::within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
              *out_ptr_CHW *= inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
            } else {
              *out_ptr_CHW = static_cast<scalar_t>(0);
            }
          }
        }
      }
    }
  }


// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
// including batch * channel offset (NC_offset).

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_premul_backward_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad_output, // [C, H, W]
      TensorInfo<scalar_t, index_t> input,       // [N, C, H, W]
      TensorInfo<scalar_t, index_t> grid,        // [N, H, W, 2]
      TensorInfo<scalar_t, index_t> grad_input,  // [N, C, H, W] initialized to zeros (or unused if input_requires_grad is false)
      TensorInfo<scalar_t, index_t> grad_grid,   // [N, H, W, 2] initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const index_t grad_input_memory_span,
      const bool input_requires_grad) {

    index_t N = input.sizes[0];
    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t gOut_sC = grad_output.strides[0];
    index_t gOut_sH = grad_output.strides[1];
    index_t gOut_sW = grad_output.strides[2];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    index_t gInp_sN;
    index_t gInp_sC;
    index_t gInp_sH;
    index_t gInp_sW;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sH = grad_input.strides[2];
      gInp_sW = grad_input.strides[3];
    }
    index_t gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = index / out_W;

      for (index_t n = 0; n < N; n++) {
        const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y co-ordinates from grid
        scalar_t x = grid.data[grid_offset];
        scalar_t y = grid.data[grid_offset + grid_sCoor];

        // multipliers for gradients on ix and iy
        scalar_t gix_mult, giy_mult;
        scalar_t ix = at::native::grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
        scalar_t iy = at::native::grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
          // get NE, NW, SE, SW pixel values from (x, y)
          index_t ix_nw = static_cast<index_t>(::floor(ix));
          index_t iy_nw = static_cast<index_t>(::floor(iy));
          index_t ix_ne = ix_nw + 1;
          index_t iy_ne = iy_nw;
          index_t ix_sw = ix_nw;
          index_t iy_sw = iy_nw + 1;
          index_t ix_se = ix_nw + 1;
          index_t iy_se = iy_nw + 1;

          // get surfaces to each neighbor:
          scalar_t nw = (ix_se - ix)    * (iy_se - iy);
          scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
          scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
          scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

          scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
          scalar_t *gOut_ptr_CHW = grad_output.data + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          scalar_t *inp_ptr_NC = input.data + n * inp_sN;
          for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_CHW += gOut_sC) {
            scalar_t gOut = *gOut_ptr_CHW;

            if (input_requires_grad) {
              // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
              at::native::safe_add_2d(grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
              at::native::safe_add_2d(grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
              at::native::safe_add_2d(grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
              at::native::safe_add_2d(grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
            }

            // calculate grad_grid
            if (at::native::within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
              scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
              gix -= nw_val * (iy_se - iy) * gOut;
              giy -= nw_val * (ix_se - ix) * gOut;
            }
            if (at::native::within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
              scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
              gix += ne_val * (iy_sw - iy) * gOut;
              giy -= ne_val * (ix - ix_sw) * gOut;
            }
            if (at::native::within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
              scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
              gix -= sw_val * (iy - iy_ne) * gOut;
              giy += sw_val * (ix_ne - ix) * gOut;
            }
            if (at::native::within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
              scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
              gix += se_val * (iy - iy_nw) * gOut;
              giy += se_val * (ix - ix_nw) * gOut;
            }
          }

          // assuming grad_grid is contiguous
          // thus we can
          //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
          //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
          scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
          gGrid_ptr_NHW[0] = gix_mult * gix;
          gGrid_ptr_NHW[1] = giy_mult * giy;
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
          if (input_requires_grad) {
            index_t ix_nearest = static_cast<index_t>(::round(ix));
            index_t iy_nearest = static_cast<index_t>(::round(iy));

            // assign nearest neighor pixel value to output pixel
            scalar_t *gOut_ptr_CHW = grad_output.data + h * gOut_sH + w * gOut_sW;
            index_t NC_offset = n * gInp_sN;
            for (index_t c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_CHW += gOut_sC) {
              // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
              at::native::safe_add_2d(grad_input.data, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_CHW, NC_offset, grad_input_memory_span);
            }
          }

          // assuming grad_grid is contiguous
          // thus we can
          //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
          //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
          scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
          gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
          gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
        }
      }
    }
  }
}


void launch_grid_sampler_2d_premul_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  at::native::check_grid_sampler_common(input, grid);
  at::native::check_grid_sampler_2d(input, grid);

  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_premul_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) && canUse32BitIndexMath(output)) {
        grid_sampler_2d_premul_kernel<scalar_t>
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
        grid_sampler_2d_premul_kernel<scalar_t>
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



void launch_grid_sampler_2d_premul_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool,2> output_mask) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  at::native::check_grid_sampler_common(input, grid);
  at::native::check_grid_sampler_2d(input, grid);

  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  auto H = grid.size(1);
  auto W = grid.size(2);

  // If `input` gradient is not required, we skip computing it -- not needing to create
  // the tensor to hold the gradient can markedly increase performance. (`grid` gradient
  // is always computed.)
  auto input_requires_grad = output_mask[0];

  int64_t count = H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_premul_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) && canUse32BitIndexMath(grad_output)) {
        grid_sampler_2d_premul_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_premul_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_input) : TensorInfo<scalar_t, int64_t>(),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}


class GridSample2dPremul : public Function<GridSample2dPremul> {
  public:
    static Tensor forward(AutogradContext *ctx,
                          Tensor &input,
                          Tensor &grid,
                          int64_t interpolation_mode,
                          int64_t padding_mode,
                          bool align_corners
    ) {
      auto in_size = input.sizes();
      auto grid_size = grid.sizes();
      auto output = at::ones(
        {in_size[1], in_size[2], in_size[3]},  input.options());  // C, H, W
      launch_grid_sampler_2d_premul_forward_kernel(
        output, input, grid, interpolation_mode, padding_mode, align_corners);
      ctx->save_for_backward({input, grid});
      ctx->saved_data["interpolation_mode"] = interpolation_mode;
      ctx->saved_data["padding_mode"] = padding_mode;
      ctx->saved_data["align_corners"] = align_corners;
      ctx->saved_data["input_needs_grad"] = input.requires_grad();
      ctx->saved_data["grid_needs_grad"] = grid.requires_grad();
      return output;
    }

    static tensor_list backward(AutogradContext *ctx,
                                tensor_list grad_outputs
    ) {
      const auto saved = ctx->get_saved_variables();
      const Tensor input = saved[0];
      const Tensor grid = saved[1];


      auto input_requires_grad = ctx->saved_data["input_needs_grad"].toBool();
      Tensor grad_input = ([&]() {
        if (input_requires_grad) {
          return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        } else {
          return Tensor();
        }
      })();
      auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      launch_grid_sampler_2d_premul_backward_kernel(
          grad_input, grad_grid, grad_outputs[0], input,
          grid, ctx->saved_data["interpolation_mode"].toInt(), ctx->saved_data["padding_mode"].toInt(),
          ctx->saved_data["align_corners"].toBool(), {input_requires_grad, ctx->saved_data["grid_needs_grad"].toBool()});

      return {grad_input, grad_grid, Tensor(), Tensor(), Tensor()};
    }
};


Tensor grid_sample_2d_premul(Tensor &input,
                      Tensor &grid,
                      int64_t interpolation_mode,
                      int64_t padding_mode,
                      bool align_corners) {
  return GridSample2dPremul::apply(input, grid, interpolation_mode, padding_mode, align_corners);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::grid_sample_2d_premul", &grid_sample_2d_premul);
