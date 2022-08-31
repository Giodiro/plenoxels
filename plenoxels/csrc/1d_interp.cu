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
  static __forceinline__ __device__
  bool within_bounds_1d(int w, int W) {
    return w >= 0 && w < W;
  }

  template<typename scalar_t, typename index_t>
  static __forceinline__ __device__
  void safe_add_1d(scalar_t *data, int w, int sW, int W,
                   scalar_t delta,
                   const index_t NC_offset,
                   const index_t memory_span) {
    if (within_bounds_1d(w, W)) {
      at::native::fastAtomicAdd(data, NC_offset + w * sW, memory_span, delta, true);
    }
  }

  // See NOTE [ grid_sampler Native Functions ].
  void check_grid_sampler_1d(
    const TensorBase& input,
    const TensorBase& grid,
    int64_t interpolation_mode
  ) {
    TORCH_CHECK(
      input.dim() == 3 && input.dim() == grid.dim(),
      "grid_sampler(): expected 3D input and grid with same number of "
      "dimensions, but got input with sizes ", input.sizes(),
      " and grid with sizes ", grid.sizes());
    TORCH_CHECK(
      !(input.dim() == 3 &&
        static_cast<GridSamplerInterpolation>(interpolation_mode) ==
          GridSamplerInterpolation::Bicubic),
      "grid_sampler(): bicubic interpolation only supports 4D input");
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_1d_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {
    index_t C = input.sizes[1];
    index_t inp_W = input.sizes[2];
    index_t out_W = grid.sizes[1];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sW = input.strides[2];
    index_t grid_sN = grid.strides[0];
    index_t grid_sW = grid.strides[1];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sW = output.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t n = index / out_W;
      const index_t grid_offset = n * grid_sN + w * grid_sW;

      // get the corresponding input x co-ordinate from grid
      scalar_t x = grid.data[grid_offset];

      scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get E, W pixel values from x
        index_t ix_w = static_cast<index_t>(::floor(ix));
        index_t ix_e = ix_w + 1;

        // get surfaces to each neighbor:
        scalar_t w = (ix_e - ix);
        scalar_t e = (ix - ix_w);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCW = output.data + n * out_sN + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCW += out_sC) {
          *out_ptr_NCW = static_cast<scalar_t>(0);
          if (within_bounds_1d(ix_w, inp_W)) {
            *out_ptr_NCW += inp_ptr_NC[ix_w * inp_sW] * w;
          }
          if (within_bounds_1d(ix_e, inp_W)) {
            *out_ptr_NCW += inp_ptr_NC[ix_e * inp_sW] * e;
          }
        }
      }
      else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(::round(ix));

        // assign nearest neighor pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCW = output.data + n * out_sN + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCW += out_sC) {
          if (within_bounds_1d(ix_nearest, inp_W)) {
            *out_ptr_NCW = inp_ptr_NC[ix_nearest * inp_sW];
          } else {
            *out_ptr_NCW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_1d_backward_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad_output,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
      TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const index_t grad_input_memory_span,
      const bool input_requires_grad) {
    index_t C = input.sizes[1];
    index_t inp_W = input.sizes[2];
    index_t out_W = grid.sizes[1];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sW = input.strides[2];
    index_t grid_sN = grid.strides[0];
    index_t grid_sW = grid.strides[1];
    index_t gout_sN = grad_output.strides[0];
    index_t gout_sC = grad_output.strides[1];
    index_t gout_sW = grad_output.strides[2];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    index_t gInp_sN;
    index_t gInp_sC;
    index_t gInp_sW;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sW = grad_input.strides[2];
    }
    index_t gGrid_sW = grad_grid.strides[1];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t n = index / out_W;
      const index_t grid_offset = n * grid_sN + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get E, W pixel values from x
        index_t ix_w = static_cast<index_t>(::floor(ix));
        index_t ix_e = ix_w + 1;

        // get surfaces to each neighbor:
        scalar_t w = (ix_e - ix);
        scalar_t e = (ix - ix_w);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCW = grad_output.data + n * gOut_sN + w * gOut_sW;
        index_t NC_offset = n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCW += gOut_sC) {
          scalar_t gOut = *gOut_ptr_NCW;

          if (input_requires_grad) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_1d(grad_input.data, ix_w, gInp_sW, inp_W, w * gOut, NC_offset, grad_input_memory_span);
            safe_add_1d(grad_input.data, ix_e, gInp_sW, inp_W, e * gOut, NC_offset, grad_input_memory_span);
          }

          // calculate grad_grid
          if (within_bounds_1d(ix_w, inp_W)) {
            scalar_t w_val = inp_ptr_NC[ix_w * inp_sW];
            gix -= w_val * gOut;
          }
          if (within_bounds_1d(ix_e, inp_W)) {
            scalar_t e_val = inp_ptr_NC[ix_e * inp_sW];
            gix += e_val * gOut;
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NW
        //   2. directly assign to gGrid_ptr_NW[0]
        scalar_t *gGrid_ptr_NW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NW[0] = gix_mult * gix;
      }
      else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        if (input_requires_grad) {
          index_t ix_nearest = static_cast<index_t>(::round(ix));

          // assign nearest neighor pixel value to output pixel
          scalar_t *gOut_ptr_NCW = grad_output.data + n * gOut_sN + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_NCW += gOut_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_1d(grad_input.data, ix_nearest, gInp_sW, inp_W, *gOut_ptr_NCW, NC_offset, grad_input_memory_span);
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NW
        //   2. directly assign to gGrid_ptr_NW[0]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
      }
    }
  }
}


void launch_grid_sampler_1d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  at::native::check_grid_sampler_common(input, grid);
  check_grid_sampler_1d(input, grid, interpolation_mode);

  auto N = input.size(0);
  auto W = grid.size(1);
  int64_t count = N * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_1d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) && canUse32BitIndexMath(output)) {
        grid_sampler_1d_kernel<scalar_t>
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


void launch_grid_sampler_1d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase& grad_output, const TensorBase& input,
    const TensorBase& grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask)
{
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  at::native::check_grid_sampler_common(input, grid);
  check_grid_sampler_1d(input, grid, interpolation_mode);

  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  //globalContext().alertNotDeterministic("grid_sampler_4d_backward_cuda");
  auto N = input.size(0);
  auto W = grid.size(1);

  // If `input` gradient is not required, we skip computing it -- not needing to create
  // the tensor to hold the gradient can markedly increase performance. (`grid` gradient
  // is always computed.)
  auto input_requires_grad = output_mask[0];

  int64_t count = N * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_1d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) && canUse32BitIndexMath(grad_output)) {
        grid_sampler_1d_backward_kernel<scalar_t>
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
        grid_sampler_1d_backward_kernel<scalar_t>
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


class GridSample1d : public Function<GridSample1d> {
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
      auto output = at::empty(
        {in_size[0], in_size[1], grid_size[1]},  input.options());  // N, C, W
      launch_grid_sampler_1d_forward_kernel(
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
      launch_grid_sampler_1d_backward_kernel(
          grad_input, grad_grid, grad_outputs[0], input,
          grid, ctx->saved_data["interpolation_mode"].toInt(), ctx->saved_data["padding_mode"].toInt(),
          ctx->saved_data["align_corners"].toBool(), {input_requires_grad, ctx->saved_data["grid_needs_grad"].toBool()});

      return {grad_input, grad_grid, Tensor(), Tensor(), Tensor()};
    }
};
