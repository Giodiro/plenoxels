#include <functional>
#include <map>
#include <vector>

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
  template <typename scalar_t, typename index_t, int dims>
  C10_LAUNCH_BOUNDS_1(512)
  __global__ void grid_sampler_nd_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,   // [N, C, Hi, Wi, ...]
      TensorInfo<scalar_t, index_t> grid,    // [N, Ho, Wo, ..., dims]
      TensorInfo<scalar_t, index_t> output,  // [N, C, Ho, Wo, ...]
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners)
  {
    constexpr uint32_t N_COMBOS = 1 << dims;
    index_t C = input.sizes[1];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t grid_sN = grid.strides[0];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t grid_sCoor = grid.strides[dims + 1];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      auto out_ptr = output.data;         // pointer to output tensor at 'index'
      auto grid_ptr = grid.data;          // pointer to grid tensor at 'index'
      // calculate pointers by looping over the spatial dimensions (from last to first)
      index_t mul = 1;
      #pragma unroll dims
      for (int d = 0; d < dims; d++) {
        const index_t gridcoo_d = (index / mul) % grid.sizes[dims - d];
        grid_ptr += gridcoo_d * grid.strides[dims - d];
        out_ptr += gridcoo_d * output.strides[dims - d + 1];
        mul *= grid.sizes[dims - d];
      }
      const index_t n = index / mul;
      grid_ptr += n * grid_sN;
      out_ptr += n * out_sN;
      auto inp_ptr = input.data + n * inp_sN;  // pointer to input tensor for the current 'batch' (first-dimension)

      scalar_t coo[dims];  // the coordinate at 'index' loaded from grid
      index_t c0[dims];    // floored `coo`
      #pragma unroll dims
      for (int d = 0; d < dims; d++) {
        coo[d] = at::native::grid_sampler_compute_source_index(
          *(grid_ptr + d * grid_sCoor),
          input.sizes[dims + 1 - d],
          padding_mode,
          align_corners);
        c0[d] = static_cast<index_t>(::floor(coo[d]));
      }

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get surfaces to each neighbor:
        // consider coordinates xyzq, where the base-corner is x_0000,y_0000,z_0000,q_0000.
        // the other corners can be obtained by simply adding 1 to the appropriate coordinate. For example:
        // x_1010 = x_0000 + 1; x_0110 = x_0000 and so on.
        // w_0000 = (x_1111 - x) * (y_1111 - y) * (z_1111 - y) * (q_1111 - q)
        // w_0010 = (x_1101 - x) * (y_1101 - y) * (z - z_1101) * (q_1101 - q)
        // w_1110 = (x - x_0001) * (y - y_0001) * (z - z_0001) * (q_0001 - q)
        // the sign of term d for each weight is determined by whether bit d is set
        // hence the global sign is determined by the parity of the bitset of i.
        scalar_t neighbor_weights[N_COMBOS];
        for (uint32_t i = 0; i < N_COMBOS; i++) {
          // set bits indicate a minus sign. If an odd number of bits are set the overall sign is -, otherwise it is +.
          neighbor_weights[i] = (__popc(i) & 1) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(+1);
          #pragma unroll dims
          for (int d = 0; d < dims; d++) {
            // the coordinate is obtained by adding 1 if the d-th bit is set in ~i.
            neighbor_weights[i] *= static_cast<scalar_t>((c0[d] + CHECK_BIT(~i, d))) - coo[d];
          }
        }

        for (index_t c = 0; c < C; ++c, inp_ptr += inp_sC, out_ptr += out_sC) {
          *out_ptr = static_cast<scalar_t>(0);
          for (uint32_t i = 0; i < N_COMBOS; i++) {
            // the first loop is done to calculate the input pointer
            index_t inp_ptr_offset = 0;
            bool in_bounds = true;
            #pragma unroll dims
            for (int d = 0; d < dims; d++) {
              index_t coo_i_d = c0[d] + CHECK_BIT(i, d);
              if (coo_i_d < 0 || coo_i_d >= input.sizes[dims + 1 - d]) {
                in_bounds = false;
                break;
              }
              inp_ptr_offset += coo_i_d * input.strides[dims + 1 - d];
            }
            if (in_bounds) {
              *out_ptr += inp_ptr[inp_ptr_offset] * neighbor_weights[i];
            }
          }
        }
      }
    }
  }


  template <typename scalar_t, typename index_t, int dims>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_nd_backward_kernel(
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
    const bool input_requires_grad)
  {
    constexpr uint32_t N_COMBOS = 1 << dims;
    index_t C = input.sizes[1];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t grid_sN = grid.strides[0];
    index_t grid_sCoor = grid.strides[dims + 1];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];

    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
    }
    index_t gGrid_sW = grad_grid.strides[dims];  // Here W is taken to be the last spatial dimension

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      /* Calculate pointers to the data-tensors at the current 'index' position. */
      auto gout_ptr = grad_output.data;        // pointer to grad-output tensor at 'index'
      auto grid_ptr = grid.data;          // pointer to grid tensor at 'index'
      // calculate pointers by looping over the spatial dimensions (from last to first)
      index_t mul = 1;
      #pragma unroll dims
      for (int d = 0; d < dims; d++) {
        const index_t gridcoo_d = (index / mul) % grid.sizes[dims - d];
        grid_ptr += gridcoo_d * grid.strides[dims - d];
        gout_ptr += gridcoo_d * grad_output.strides[dims - d + 1];
        mul *= grid.sizes[dims - d];
      }
      const index_t n = index / mul;
      grid_ptr += n * grid_sN;
      gout_ptr += n * gOut_sN;
      index_t gInpNC_offset;
      index_t inpNC_offset = n * inp_sN;
      if (input_requires_grad) {
        gInpNC_offset = n * gInp_sN;
      }

      scalar_t coo[dims];  // the coordinate at 'index' loaded from grid
      scalar_t gcoo_mult[dims];
      index_t c0[dims];    // floored `coo`
      #pragma unroll dims
      for (int d = 0; d < dims; d++) {
        coo[d] = at::native::grid_sampler_compute_source_index_set_grad(
          *(grid_ptr + d * grid_sCoor),
          input.sizes[dims + 1 - d],
          padding_mode,
          align_corners,
          gcoo_mult + d);
        c0[d] = static_cast<index_t>(::floor(coo[d]));
      }

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get surfaces to each neighbor:
        scalar_t neighbor_weights[N_COMBOS];
        for (uint32_t i = 0; i < N_COMBOS; i++) {
          // set bits indicate a minus sign. If an odd number of bits are set the overall sign is -, otherwise it is +.
          neighbor_weights[i] = (__popc(i) & 1) ? static_cast<scalar_t>(-1) : static_cast<scalar_t>(+1);
          #pragma unroll dims
          for (int d = 0; d < dims; d++) {
            // the coordinate is obtained by adding 1 if the d-th bit is set in ~i.
            neighbor_weights[i] *= static_cast<scalar_t>((c0[d] + CHECK_BIT(~i, d))) - coo[d];
          }
        }

        scalar_t gcoo[dims] = { static_cast<scalar_t>(0) };
        for (index_t c = 0; c < C; ++c, gout_ptr += gOut_sC, gInpNC_offset += gInp_sC, inpNC_offset += inp_sC) {
          scalar_t gOut = *gout_ptr;
          for (int i = 0; i < N_COMBOS; i++) {
            // calculate the pointer to grad-input, then if in-bounds do atomic-add.
            index_t ginp_ptr_offset = gInpNC_offset;
            index_t inp_ptr_offset = inpNC_offset;
            bool in_bounds = true;
            #pragma unroll dims
            for (int d = 0; d < dims; d++) {
              index_t coo_i_d = c0[d] + CHECK_BIT(i, d);
              if (coo_i_d < 0 || coo_i_d >= grad_input.sizes[dims + 1 - d]) {
                in_bounds = false;
                break;
              }
              if (input_requires_grad) {
                ginp_ptr_offset += coo_i_d * grad_input.strides[dims + 1 - d];
              }
              inp_ptr_offset += coo_i_d * input.strides[dims + 1 - d];
            }
            // input-grad
            if (input_requires_grad && in_bounds) {
              at::native::fastAtomicAdd(grad_input.data,
                                        ginp_ptr_offset,
                                        grad_input_memory_span,
                                        gOut * neighbor_weights[i],
                                        true);
            }
            // grid-grad
            if (in_bounds) {
              scalar_t inp_val = input.data[inp_ptr_offset];
              for (int d = 0; d < dims; d++) {
                scalar_t gCooD = (CHECK_BIT(i, d) ? +1 : -1) * gOut * inp_val;
                #pragma unroll dims
                for (int d2 = 0; d2 < dims; d2++) {
                  if (d2 == d) continue;
                  gCooD *= (CHECK_BIT(i, d2) ? -1 : +1) * (static_cast<scalar_t>((c0[d2] + CHECK_BIT(~i, d2))) - coo[d2]);
                }
                gcoo[d] += gCooD;
              }
            }
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NLDHW
        //   2. directly assign to gGrid_ptr_NLDHW[0], gGrid_ptr_NLDHW[1], gGrid_ptr_NLDHW[2], gGrid_ptr_NLDHW[3]
        scalar_t *gGrid_ptr_NdimsW = grad_grid.data + index * gGrid_sW;
        #pragma unroll dims
        for (int d = 0; d < dims; d++) {
          gGrid_ptr_NdimsW[d] = gcoo_mult[d] * gcoo[d];
        }
      }
    }
  }
}


template<typename scalar_t, typename index_t, int ... Ds>
void call_fwd_kernel(
      std::integer_sequence<int, Ds...>, 
      const index_t d,
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,   // [N, C, Hi, Wi, ...]
      TensorInfo<scalar_t, index_t> grid,    // [N, Ho, Wo, O]
      TensorInfo<scalar_t, index_t> output,  // [N, C, Ho, Wo]
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {
  void (*fs[])(const index_t, TensorInfo<scalar_t, index_t>, TensorInfo<scalar_t, index_t>, TensorInfo<scalar_t, index_t>, const GridSamplerInterpolation, const GridSamplerPadding, bool) = {&grid_sampler_nd_kernel<scalar_t, index_t, Ds>...};
  fs[d]<<<GET_BLOCKS(nthreads, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(nthreads, input, grid, output, interpolation_mode, padding_mode, align_corners);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename scalar_t, typename index_t, int ... Ds>
void call_bwd_kernel(
    std::integer_sequence<int, Ds...>,
    const index_t d,
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
  void (*fs[])(const index_t, TensorInfo<scalar_t, index_t>, TensorInfo<scalar_t, index_t>,
               TensorInfo<scalar_t, index_t>, TensorInfo<scalar_t, index_t>, TensorInfo<scalar_t, index_t>,
               const GridSamplerInterpolation, const GridSamplerPadding, bool, const index_t, const bool) = {&grid_sampler_nd_backward_kernel<scalar_t, index_t, Ds>...};
  fs[d]<<<GET_BLOCKS(nthreads, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
    nthreads, grad_output, input, grid, grad_input, grad_grid, interpolation_mode, padding_mode, align_corners, grad_input_memory_span, input_requires_grad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void launch_grid_sampler_nd_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.

  auto N = input.size(0);
  auto dims = grid.dim() - 2;
  int64_t count = N;
  for (int d = 1; d < grid.dim() - 1; d++) {
    count *= grid.size(d);
  }

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_nd_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        call_fwd_kernel<scalar_t>(
            std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7>{}, 
            static_cast<int>(dims) - 1,
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
      } else {
        call_fwd_kernel<scalar_t>(
            std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7>{}, 
            dims - 1,
            count,
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
      }
    });
  }
}

void launch_grid_sampler_nd_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase& grad_output, const TensorBase& input,
    const TensorBase& grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool,2> output_mask) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.

  auto N = input.size(0);
  auto dims = grid.dim() - 2;
  int64_t count = N;
  for (int d = 1; d < grid.dim() - 1; d++) {
    count *= grid.size(d);
  }
  auto input_requires_grad = output_mask[0];

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_nd_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) && canUse32BitIndexMath(grad_output)) {
        call_bwd_kernel<scalar_t>(
            std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7>{},
            static_cast<int>(dims) - 1,
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
      } else {
        call_bwd_kernel<scalar_t>(
            std::integer_sequence<int, 1, 2, 3, 4, 5, 6, 7>{},
            dims - 1,
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
      }
    });
  }
}


class GridSampleNd : public Function<GridSampleNd> {
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
      std::vector<int64_t> out_shape = grid.sizes().vec();
      out_shape.erase(out_shape.begin());
      out_shape.erase(out_shape.end() - 1);
      out_shape.insert(out_shape.begin(), {in_size[0], in_size[1]}); // N, C, sizes
      auto output = at::empty(out_shape, input.options());
      launch_grid_sampler_nd_forward_kernel(
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
      launch_grid_sampler_nd_backward_kernel(
          grad_input, grad_grid, grad_outputs[0], input,
          grid, ctx->saved_data["interpolation_mode"].toInt(), ctx->saved_data["padding_mode"].toInt(),
          ctx->saved_data["align_corners"].toBool(), {input_requires_grad, ctx->saved_data["grid_needs_grad"].toBool()});

      return {grad_input, grad_grid, Tensor(), Tensor(), Tensor()};
    }
};

Tensor grid_sample_nd(Tensor &input,
                      Tensor &grid,
                      int64_t interpolation_mode,
                      int64_t padding_mode,
                      bool align_corners)
{
  return GridSampleNd::apply(input, grid, interpolation_mode, padding_mode, align_corners);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::grid_sample_nd", &grid_sample_nd);
