#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/native/cuda/GridSampler.cuh>
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
using torch::autograd::variable_list;
using torch::autograd::tensor_list;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::Tensor;


namespace {
  static __forceinline__ __device__
  bool within_bounds_4d(int l, int d, int h, int w, int L, int D, int H, int W) {
    return l >= 0 && l < L && d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }

  template<typename scalar_t, typename index_t>
  static __forceinline__ __device__
  void safe_add_4d(scalar_t *data, int l, int d, int h, int w,
                   int sL, int sD, int sH, int sW, int L, int D, int H, int W,
                   scalar_t delta,
                   const index_t NC_offset,
                   const index_t memory_span) {
    if (within_bounds_4d(l, d, h, w, L, D, H, W)) {
      fastAtomicAdd(data,
                    NC_offset + l * sL + d * sD + h * sH + w * sW,
                    memory_span,
                    delta,
                    true);
    }
  }

  // See NOTE [ grid_sampler Native Functions ].
  void check_grid_sampler_4d(
    const TensorBase& input,
    const TensorBase& grid,
    int64_t interpolation_mode
  ) {
    TORCH_CHECK(
      input.dim() == 6 && input.dim() == grid.dim(),
      "grid_sampler(): expected 6D input and grid with same number of "
      "dimensions, but got input with sizes ", input.sizes(),
      " and grid with sizes ", grid.sizes());
    TORCH_CHECK(
      !(input.dim() == 6 &&
        static_cast<GridSamplerInterpolation>(interpolation_mode) ==
          GridSamplerInterpolation::Bicubic),
      "grid_sampler(): bicubic interpolation only supports 4D input");
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(512)
  __global__ void grid_sampler_4d_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners)
  {
    index_t C = input.sizes[1];
    index_t inp_L = input.sizes[2];
    index_t inp_D = input.sizes[3];
    index_t inp_H = input.sizes[4];
    index_t inp_W = input.sizes[5];
    index_t out_L = grid.sizes[1];
    index_t out_D = grid.sizes[2];
    index_t out_H = grid.sizes[3];
    index_t out_W = grid.sizes[4];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sL = input.strides[2];
    index_t inp_sD = input.strides[3];
    index_t inp_sH = input.strides[4];
    index_t inp_sW = input.strides[5];
    index_t grid_sN = grid.strides[0];
    index_t grid_sL = grid.strides[1];
    index_t grid_sD = grid.strides[2];
    index_t grid_sH = grid.strides[3];
    index_t grid_sW = grid.strides[4];
    index_t grid_sCoor = grid.strides[5];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sL = output.strides[2]
    index_t out_sD = output.strides[3];
    index_t out_sH = output.strides[4];
    index_t out_sW = output.strides[5];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t l = (index / (out_H * out_W * out_D)) % out_L;
      const index_t n = index / (out_D * out_H * out_W * out_L);
      const auto grid_offset = n * grid_sN + l * grid_sL + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z, q coordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];
      scalar_t iq = grid.data[grid_offset + 3 * grid_sCoor];

      ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
      iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
      iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);
      iq = grid_sampler_compute_source_index(iq, inp_L, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z, q)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        // for 6d, we add left-right
        index_t ix_ltnw = static_cast<index_t>(::floor(ix));
        index_t iy_ltnw = static_cast<index_t>(::floor(iy));
        index_t iz_ltnw = static_cast<index_t>(::floor(iz));
        index_t iq_ltnw = static_cast<index_t>(::floor(iq));

        index_t ix_ltne = ix_ltnw + 1;
        index_t iy_ltne = iy_ltnw;
        index_t iz_ltne = iz_ltnw;
        index_t iq_ltne = iq_ltnw;

        index_t ix_ltsw = ix_ltnw;
        index_t iy_ltsw = iy_ltnw + 1;
        index_t iz_ltsw = iz_ltnw;
        index_t iq_ltsw = iq_ltnw;

        index_t ix_ltse = ix_ltnw + 1;
        index_t iy_ltse = iy_ltnw + 1;
        index_t iz_ltse = iz_ltnw;
        index_t iq_ltse = iq_ltnw;

        index_t ix_lbnw = ix_ltnw;
        index_t iy_lbnw = iy_ltnw;
        index_t iz_lbnw = iz_ltnw + 1;
        index_t iq_lbnw = iq_ltnw;

        index_t ix_lbne = ix_ltnw + 1;
        index_t iy_lbne = iy_ltnw;
        index_t iz_lbne = iz_ltnw + 1;
        index_t iq_lbne = iq_ltnw;

        index_t ix_lbsw = ix_ltnw;
        index_t iy_lbsw = iy_ltnw + 1;
        index_t iz_lbsw = iz_ltnw + 1;
        index_t iq_lbsw = iq_ltnw;

        index_t ix_lbse = ix_ltnw + 1;
        index_t iy_lbse = iy_ltnw + 1;
        index_t iz_lbse = iz_ltnw + 1;
        index_t iq_lbse = iq_ltnw;

        index_t ix_rtnw = ix_ltnw;
        index_t iy_rtnw = iy_ltnw;
        index_t iz_rtnw = iz_ltnw;
        index_t iq_rtnw = iq_ltnw + 1;

        index_t ix_rtne = ix_ltnw + 1;
        index_t iy_rtne = iy_ltnw;
        index_t iz_rtne = iz_ltnw;
        index_t iq_rtne = iq_ltnw + 1;

        index_t ix_rtsw = ix_ltnw;
        index_t iy_rtsw = iy_ltnw + 1;
        index_t iz_rtsw = iz_ltnw;
        index_t iq_rtsw = iq_ltnw + 1;

        index_t ix_rtse = ix_ltnw + 1;
        index_t iy_rtse = iy_ltnw + 1;
        index_t iz_rtse = iz_ltnw;
        index_t iq_rtse = iq_ltnw + 1;

        index_t ix_rbnw = ix_ltnw;
        index_t iy_rbnw = iy_ltnw;
        index_t iz_rbnw = iz_ltnw + 1;
        index_t iq_rbnw = iq_ltnw + 1;

        index_t ix_rbne = ix_ltnw + 1;
        index_t iy_rbne = iy_ltnw;
        index_t iz_rbne = iz_ltnw + 1;
        index_t iq_rbne = iq_ltnw + 1;

        index_t ix_rbsw = ix_ltnw;
        index_t iy_rbsw = iy_ltnw + 1;
        index_t iz_rbsw = iz_ltnw + 1;
        index_t iq_rbsw = iq_ltnw + 1;

        index_t ix_rbse = ix_ltnw + 1;
        index_t iy_rbse = iy_ltnw + 1;
        index_t iz_rbse = iz_ltnw + 1;
        index_t iq_rbse = iq_ltnw + 1;

        // get surfaces to each neighbor:
        scalar_t ltnw = (ix_rbse - ix) * (iy_rbse - iy) * (iz_rbse - iz) * (iq_rbse - iq);
        scalar_t ltne = (ix - ix_rbsw) * (iy_rbsw - iy) * (iz_rbsw - iz) * (iq_rbsw - iq);
        scalar_t ltsw = (ix_rbne - ix) * (iy - iy_rbne) * (iz_rbne - iz) * (iq_rbne - iq);
        scalar_t ltse = (ix - ix_rbnw) * (iy - iy_rbnw) * (iz_rbnw - iz) * (iq_rbnw - iq);
        scalar_t lbnw = (ix_rtse - ix) * (iy_rtse - iy) * (iz - iz_rtse) * (iq_rtse - iq);
        scalar_t lbne = (ix - ix_rtsw) * (iy_rtsw - iy) * (iz - iz_rtsw) * (iq_rtsw - iq);
        scalar_t lbsw = (ix_rtne - ix) * (iy - iy_rtne) * (iz - iz_rtne) * (iq_rtne - iq);
        scalar_t lbse = (ix - ix_rtnw) * (iy - iy_rtnw) * (iz - iz_rtnw) * (iq_rtnw - iq);
        scalar_t rtnw = (ix_lbse - ix) * (iy_lbse - iy) * (iq_lbse - iz) * (iq - iq_lbse);
        scalar_t rtne = (ix - ix_lbsw) * (iy_lbsw - iy) * (iq_lbsw - iz) * (iq - iq_lbsw);
        scalar_t rtsw = (ix_lbne - ix) * (iy - iy_lbne) * (iq_lbne - iz) * (iq - iq_lbne);
        scalar_t rtse = (ix - ix_lbnw) * (iy - iy_lbnw) * (iq_lbnw - iz) * (iq - iq_lbnw);
        scalar_t rbnw = (ix_ltse - ix) * (iy_ltse - iy) * (iz - iz_ltse) * (iq - iq_ltse);
        scalar_t rbne = (ix - ix_ltsw) * (iy_ltsw - iy) * (iz - iz_ltsw) * (iq - iq_ltsw);
        scalar_t rbsw = (ix_ltne - ix) * (iy - iy_ltne) * (iz - iz_ltne) * (iq - iq_ltne);
        scalar_t rbse = (ix - ix_ltnw) * (iy - iy_ltnw) * (iz - iz_ltnw) * (iq - iq_ltnw);

        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCLDHW = output.data + n * out_sN + l * out_sL + d * out_sD + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCLDHW += out_sC) {
          //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
          // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
          // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
          // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
          *out_ptr_NCLDHW = static_cast<scalar_t>(0);
          if (within_bounds_4d(iq_ltnw, iz_ltnw, iy_ltnw, ix_ltnw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_ltnw * inp_sL + iz_ltnw * inp_sD + iy_ltnw * inp_sH + ix_ltnw * inp_sW] * ltnw;
          }
          if (within_bounds_4d(iq_ltne, iz_ltne, iy_ltne, ix_ltne, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_ltne * inp_sL + iz_ltne * inp_sD + iy_ltne * inp_sH + ix_ltne * inp_sW] * ltne;
          }
          if (within_bounds_4d(iq_ltsw, iz_ltsw, iy_ltsw, ix_ltsw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_ltsw * inp_sL + iz_ltsw * inp_sD + iy_ltsw * inp_sH + ix_ltsw * inp_sW] * ltsw;
          }
          if (within_bounds_4d(iq_ltse, iz_ltse, iy_ltse, ix_ltse, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_ltse * inp_sL + iz_ltse * inp_sD + iy_ltse * inp_sH + ix_ltse * inp_sW] * ltse;
          }
          if (within_bounds_4d(iq_lbnw, iz_lbnw, iy_lbnw, ix_lbnw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_lbnw * inp_sL + iz_lbnw * inp_sD + iy_lbnw * inp_sH + ix_lbnw * inp_sW] * lbnw;
          }
          if (within_bounds_4d(iq_lbne, iz_lbne, iy_lbne, ix_lbne, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_lbne * inp_sL + iz_lbne * inp_sD + iy_lbne * inp_sH + ix_lbne * inp_sW] * lbne;
          }
          if (within_bounds_4d(iq_lbsw, iz_lbsw, iy_lbsw, ix_lbsw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_lbsw * inp_sL + iz_lbsw * inp_sD + iy_lbsw * inp_sH + ix_lbsw * inp_sW] * lbsw;
          }
          if (within_bounds_4d(iq_lbse, iz_lbse, iy_lbse, ix_lbse, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_lbse * inp_sL + iz_lbse * inp_sD + iy_lbse * inp_sH + ix_lbse * inp_sW] * lbse;
          }
          if (within_bounds_4d(iq_rtnw, iz_rtnw, iy_rtnw, ix_rtnw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rtnw * inp_sL + iz_rtnw * inp_sD + iy_rtnw * inp_sH + ix_rtnw * inp_sW] * rtnw;
          }
          if (within_bounds_4d(iq_rtne, iz_rtne, iy_rtne, ix_rtne, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rtne * inp_sL + iz_rtne * inp_sD + iy_rtne * inp_sH + ix_rtne * inp_sW] * rtne;
          }
          if (within_bounds_4d(iq_rtsw, iz_rtsw, iy_rtsw, ix_rtsw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rtsw * inp_sL + iz_rtsw * inp_sD + iy_rtsw * inp_sH + ix_rtsw * inp_sW] * rtsw;
          }
          if (within_bounds_4d(iq_rtse, iz_rtse, iy_rtse, ix_rtse, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rtse * inp_sL + iz_rtse * inp_sD + iy_rtse * inp_sH + ix_rtse * inp_sW] * rtse;
          }
          if (within_bounds_4d(iq_rbnw, iz_rbnw, iy_rbnw, ix_rbnw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rbnw * inp_sL + iz_rbnw * inp_sD + iy_rbnw * inp_sH + ix_rbnw * inp_sW] * rbnw;
          }
          if (within_bounds_4d(iq_rbne, iz_rbne, iy_rbne, ix_rbne, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rbne * inp_sL + iz_rbne * inp_sD + iy_rbne * inp_sH + ix_rbne * inp_sW] * rbne;
          }
          if (within_bounds_4d(iq_rbsw, iz_rbsw, iy_rbsw, ix_rbsw, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rbsw * inp_sL + iz_rbsw * inp_sD + iy_rbsw * inp_sH + ix_rbsw * inp_sW] * rbsw;
          }
          if (within_bounds_4d(iq_rbse, iz_rbse, iy_rbse, ix_rbse, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW += inp_ptr_NC[iq_rbse * inp_sL + iz_rbse * inp_sD + iy_rbse * inp_sH + ix_rbse * inp_sW] * rbse;
          }
        }
      }
      else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(::round(ix));
        index_t iy_nearest = static_cast<index_t>(::round(iy));
        index_t iz_nearest = static_cast<index_t>(::round(iz));
        index_t iq_nearest = static_cast<index_t>(::round(iq));

        // assign nearest neighor pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCLDHW = output.data + n * out_sN + l * out_sL + d * out_sD + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCLDHW += out_sC) {
          if (within_bounds_4d(iq_nearest, iz_nearest, iy_nearest, ix_nearest, inp_L, inp_D, inp_H, inp_W)) {
            *out_ptr_NCLDHW = inp_ptr_NC[iq_nearest * inp_sL + iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCLDHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_4d_backward_kernel(
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
    index_t C = input.sizes[1];
    index_t inp_L = input.sizes[2];
    index_t inp_D = input.sizes[3];
    index_t inp_H = input.sizes[4];
    index_t inp_W = input.sizes[5];
    index_t out_L = grid.sizes[1];
    index_t out_D = grid.sizes[2];
    index_t out_H = grid.sizes[3];
    index_t out_W = grid.sizes[4];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sL = input.strides[2];
    index_t inp_sD = input.strides[3];
    index_t inp_sH = input.strides[4];
    index_t inp_sW = input.strides[5];
    index_t grid_sN = grid.strides[0];
    index_t grid_sL = grid.strides[1];
    index_t grid_sD = grid.strides[2];
    index_t grid_sH = grid.strides[3];
    index_t grid_sW = grid.strides[4];
    index_t grid_sCoor = grid.strides[5];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sL = grad_output.strides[2];
    index_t gOut_sD = grad_output.strides[3];
    index_t gOut_sH = grad_output.strides[4];
    index_t gOut_sW = grad_output.strides[5];

    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    int64_t gInp_sL = 0;
    int64_t gInp_sD = 0;
    int64_t gInp_sH = 0;
    int64_t gInp_sW = 0;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sL = grad_input.strides[2];
      gInp_sD = grad_input.strides[3];
      gInp_sH = grad_input.strides[4];
      gInp_sW = grad_input.strides[5];
    }
    index_t gGrid_sW = grad_grid.strides[4];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t l = (index / (out_H * out_W * out_D)) % out_L;
      const index_t n = index / (out_D * out_H * out_W * out_L);
      const auto grid_offset = n * grid_sN + l * grid_sL + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z, q coordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];
      scalar_t iq = grid.data[grid_offset + 3 * grid_sCoor];

      // multipliers for gradients on ix, iy, iz and iq
      scalar_t gix_mult, giy_mult, giz_mult, giq_mult;
      ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
      iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
      iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);
      iq = grid_sampler_compute_source_index_set_grad(iq, inp_L, padding_mode, align_corners, &gil_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z, q)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        // for 6d, we add left-right
        index_t ix_ltnw = static_cast<index_t>(::floor(ix));
        index_t iy_ltnw = static_cast<index_t>(::floor(iy));
        index_t iz_ltnw = static_cast<index_t>(::floor(iz));
        index_t iq_ltnw = static_cast<index_t>(::floor(iq));

        index_t ix_ltne = ix_ltnw + 1;
        index_t iy_ltne = iy_ltnw;
        index_t iz_ltne = iz_ltnw;
        index_t iq_ltne = iq_ltnw;

        index_t ix_ltsw = ix_ltnw;
        index_t iy_ltsw = iy_ltnw + 1;
        index_t iz_ltsw = iz_ltnw;
        index_t iq_ltsw = iq_ltnw;

        index_t ix_ltse = ix_ltnw + 1;
        index_t iy_ltse = iy_ltnw + 1;
        index_t iz_ltse = iz_ltnw;
        index_t iq_ltse = iq_ltnw;

        index_t ix_lbnw = ix_ltnw;
        index_t iy_lbnw = iy_ltnw;
        index_t iz_lbnw = iz_ltnw + 1;
        index_t iq_lbnw = iq_ltnw;

        index_t ix_lbne = ix_ltnw + 1;
        index_t iy_lbne = iy_ltnw;
        index_t iz_lbne = iz_ltnw + 1;
        index_t iq_lbne = iq_ltnw;

        index_t ix_lbsw = ix_ltnw;
        index_t iy_lbsw = iy_ltnw + 1;
        index_t iz_lbsw = iz_ltnw + 1;
        index_t iq_lbsw = iq_ltnw;

        index_t ix_lbse = ix_ltnw + 1;
        index_t iy_lbse = iy_ltnw + 1;
        index_t iz_lbse = iz_ltnw + 1;
        index_t iq_lbse = iq_ltnw;

        index_t ix_rtnw = ix_ltnw;
        index_t iy_rtnw = iy_ltnw;
        index_t iz_rtnw = iz_ltnw;
        index_t iq_rtnw = iq_ltnw + 1;

        index_t ix_rtne = ix_ltnw + 1;
        index_t iy_rtne = iy_ltnw;
        index_t iz_rtne = iz_ltnw;
        index_t iq_rtne = iq_ltnw + 1;

        index_t ix_rtsw = ix_ltnw;
        index_t iy_rtsw = iy_ltnw + 1;
        index_t iz_rtsw = iz_ltnw;
        index_t iq_rtsw = iq_ltnw + 1;

        index_t ix_rtse = ix_ltnw + 1;
        index_t iy_rtse = iy_ltnw + 1;
        index_t iz_rtse = iz_ltnw;
        index_t iq_rtse = iq_ltnw + 1;

        index_t ix_rbnw = ix_ltnw;
        index_t iy_rbnw = iy_ltnw;
        index_t iz_rbnw = iz_ltnw + 1;
        index_t iq_rbnw = iq_ltnw + 1;

        index_t ix_rbne = ix_ltnw + 1;
        index_t iy_rbne = iy_ltnw;
        index_t iz_rbne = iz_ltnw + 1;
        index_t iq_rbne = iq_ltnw + 1;

        index_t ix_rbsw = ix_ltnw;
        index_t iy_rbsw = iy_ltnw + 1;
        index_t iz_rbsw = iz_ltnw + 1;
        index_t iq_rbsw = iq_ltnw + 1;

        index_t ix_rbse = ix_ltnw + 1;
        index_t iy_rbse = iy_ltnw + 1;
        index_t iz_rbse = iz_ltnw + 1;
        index_t iq_rbse = iq_ltnw + 1;

        // get surfaces to each neighbor:
        scalar_t ltnw = (ix_rbse - ix) * (iy_rbse - iy) * (iz_rbse - iz) * (iq_rbse - iq);
        scalar_t ltne = (ix - ix_rbsw) * (iy_rbsw - iy) * (iz_rbsw - iz) * (iq_rbsw - iq);
        scalar_t ltsw = (ix_rbne - ix) * (iy - iy_rbne) * (iz_rbne - iz) * (iq_rbne - iq);
        scalar_t ltse = (ix - ix_rbnw) * (iy - iy_rbnw) * (iz_rbnw - iz) * (iq_rbnw - iq);
        scalar_t lbnw = (ix_rtse - ix) * (iy_rtse - iy) * (iz - iz_rtse) * (iq_rtse - iq);
        scalar_t lbne = (ix - ix_rtsw) * (iy_rtsw - iy) * (iz - iz_rtsw) * (iq_rtsw - iq);
        scalar_t lbsw = (ix_rtne - ix) * (iy - iy_rtne) * (iz - iz_rtne) * (iq_rtne - iq);
        scalar_t lbse = (ix - ix_rtnw) * (iy - iy_rtnw) * (iz - iz_rtnw) * (iq_rtnw - iq);
        scalar_t rtnw = (ix_lbse - ix) * (iy_lbse - iy) * (iq_lbse - iz) * (iq - iq_lbse);
        scalar_t rtne = (ix - ix_lbsw) * (iy_lbsw - iy) * (iq_lbsw - iz) * (iq - iq_lbsw);
        scalar_t rtsw = (ix_lbne - ix) * (iy - iy_lbne) * (iq_lbne - iz) * (iq - iq_lbne);
        scalar_t rtse = (ix - ix_lbnw) * (iy - iy_lbnw) * (iq_lbnw - iz) * (iq - iq_lbnw);
        scalar_t rbnw = (ix_ltse - ix) * (iy_ltse - iy) * (iz - iz_ltse) * (iq - iq_ltse);
        scalar_t rbne = (ix - ix_ltsw) * (iy_ltsw - iy) * (iz - iz_ltsw) * (iq - iq_ltsw);
        scalar_t rbsw = (ix_ltne - ix) * (iy - iy_ltne) * (iz - iz_ltne) * (iq - iq_ltne);
        scalar_t rbse = (ix - ix_ltnw) * (iy - iy_ltnw) * (iz - iz_ltnw) * (iq - iq_ltnw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0),
                 giz = static_cast<scalar_t>(0), giq = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCLDHW = grad_output.data + n * gOut_sN + l * gOut_sL + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        index_t NC_offset;
        if (input_requires_grad) {
          NC_offset = n * gInp_sN;
        }
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        // calculate bilinear weighted pixel value and set output pixel
        for (index_t c = 0; c < C; ++c, gOut_ptr_NCLDHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC) {
          scalar_t gOut = *gOut_ptr_NCLDHW;
          // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
          if (input_requires_grad) {
            safe_add_4d(grad_input.data, iq_ltnw, iz_ltnw, iy_ltnw, ix_ltnw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, ltnw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_ltne, iz_ltne, iy_ltne, ix_ltne, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, ltne * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_ltsw, iz_ltsw, iy_ltsw, ix_ltsw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, ltsw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_ltse, iz_ltse, iy_ltse, ix_ltse, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, ltse * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_lbnw, iz_lbnw, iy_lbnw, ix_lbnw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, lbnw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_lbne, iz_lbne, iy_lbne, ix_lbne, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, lbne * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_lbsw, iz_lbsw, iy_lbsw, ix_lbsw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, lbsw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_lbse, iz_lbse, iy_lbse, ix_lbse, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, lbse * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rtnw, iz_rtnw, iy_rtnw, ix_rtnw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rtnw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rtne, iz_rtne, iy_rtne, ix_rtne, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rtne * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rtsw, iz_rtsw, iy_rtsw, ix_rtsw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rtsw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rtse, iz_rtse, iy_rtse, ix_rtse, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rtse * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rbnw, iz_rbnw, iy_rbnw, ix_rbnw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rbnw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rbne, iz_rbne, iy_rbne, ix_rbne, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rbne * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rbsw, iz_rbsw, iy_rbsw, ix_rbsw, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rbsw * gOut, NC_offset, grad_input_memory_span);
            safe_add_4d(grad_input.data, iq_rbse, iz_rbse, iy_rbse, ix_rbse, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, rbse * gOut, NC_offset, grad_input_memory_span);
          }
          // calculate grad_grid
          if (within_bounds_4d(iq_ltnw, iz_ltnw, iy_ltnw, ix_ltnw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t ltnw_val = inp_ptr_NC[iq_ltnw * inp_sL + iz_ltnw * inp_sD + iy_ltnw * inp_sH + ix_ltnw * inp_sW];
            gix -= ltnw_val * (iy_rbse - iy) * (iz_rbse - iz) * (iq_rbse - iq) * gOut;
            giy -= ltnw_val * (ix_rbse - ix) * (iz_rbse - iz) * (iq_rbse - iq) * gOut;
            giz -= ltnw_val * (ix_rbse - ix) * (iy_rbse - iy) * (iq_rbse - iq) * gOut;
            giq -= ltnw_val * (ix_rbse - ix) * (iy_rbse - iy) * (iz_rbse - iz) * gOut;
          }
          if (within_bounds_4d(iq_ltne, iz_ltne, iy_ltne, ix_ltne, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t ltne_val = inp_ptr_NC[iq_ltne * inp_sL + iz_ltne * inp_sD + iy_ltne * inp_sH + ix_ltne * inp_sW];
            gix += ltne_val * (iy_rbsw - iy) * (iz_rbsw - iz) * (iq_rbsw - iq) * gOut;
            giy -= ltne_val * (ix - ix_rbsw) * (iz_rbsw - iz) * (iq_rbsw - iq) * gOut;
            giz -= ltne_val * (ix - ix_rbsw) * (iy_rbsw - iy) * (iq_rbsw - iq) * gOut;
            giq -= ltne_val * (ix - ix_rbsw) * (iy_rbsw - iy) * (iz_rbsw - iz) * gOut;
          }
          if (within_bounds_4d(iq_ltsw, iz_ltsw, iy_ltsw, ix_ltsw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t ltsw_val = inp_ptr_NC[iq_ltsw * inp_sL + iz_ltsw * inp_sD + iy_ltsw * inp_sH + ix_ltsw * inp_sW];
            gix -= ltsw_val * (iy - iy_rbne) * (iz_rbne - iz) * (iq_rbne - iq) * gOut;
            giy += ltsw_val * (ix_rbne - ix) * (iz_rbne - iz) * (iq_rbne - iq) * gOut;
            giz -= ltsw_val * (ix_rbne - ix) * (iy - iy_rbne) * (iq_rbne - iq) * gOut;
            giq -= ltsw_val * (ix_rbne - ix) * (iy - iy_rbne) * (iz_rbne - iz) * gOut;
          }
          if (within_bounds_4d(iq_ltse, iz_ltse, iy_ltse, ix_ltse, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t ltse_val = inp_ptr_NC[iq_ltse * inp_sL + iz_ltse * inp_sD + iy_ltse * inp_sH + ix_ltse * inp_sW];
            gix += ltse_val * (iy - iy_rbnw) * (iz_rbnw - iz) * (iq_rbnw - iq) * gOut;
            giy += ltse_val * (ix - ix_rbnw) * (iz_rbnw - iz) * (iq_rbnw - iq) * gOut;
            giz -= ltse_val * (ix - ix_rbnw) * (iy - iy_rbnw) * (iq_rbnw - iq) * gOut;
            giq -= ltse_val * (ix - ix_rbnw) * (iy - iy_rbnw) * (iz_rbnw - iz) * gOut;
          }
          if (within_bounds_4d(iq_lbnw, iz_lbnw, iy_lbnw, ix_lbnw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t lbnw_val = inp_ptr_NC[iq_lbnw * inp_sL + iz_lbnw * inp_sD + iy_lbnw * inp_sH + ix_lbnw * inp_sW];
            gix -= lbnw_val * (iy_rtse - iy) * (iz - iz_rtse) * (iq_rtse - iq) * gOut;
            giy -= lbnw_val * (ix_rtse - ix) * (iz - iz_rtse) * (iq_rtse - iq) * gOut;
            giz += lbnw_val * (ix_rtse - ix) * (iy_rtse - iy) * (iq_rtse - iq) * gOut;
            giq -= lbnw_val * (ix_rtse - ix) * (iy_rtse - iy) * (iz - iz_rtse) * gOut;
          }
          if (within_bounds_4d(iq_lbne, iz_lbne, iy_lbne, ix_lbne, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t lbne_val = inp_ptr_NC[iq_lbne * inp_sL + iz_lbne * inp_sD + iy_lbne * inp_sH + ix_lbne * inp_sW];
            gix += lbne_val * (iy_rtsw - iy) * (iz - iz_rtsw) * (iq_rtsw - iq) * gOut;
            giy -= lbne_val * (ix - ix_rtsw) * (iz - iz_rtsw) * (iq_rtsw - iq) * gOut;
            giz += lbne_val * (ix - ix_rtsw) * (iy_rtsw - iy) * (iq_rtsw - iq) * gOut;
            giq -= lbne_val * (ix - ix_rtsw) * (iy_rtsw - iy) * (iz - iz_rtsw) * gOut;
          }
          if (within_bounds_4d(iq_lbsw, iz_lbsw, iy_lbsw, ix_lbsw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t lbsw_val = inp_ptr_NC[iq_lbsw * inp_sL + iz_lbsw * inp_sD + iy_lbsw * inp_sH + ix_lbsw * inp_sW];
            gix -= lbsw_val * (iy - iy_rtne) * (iz - iz_rtne) * (iq_rtne - iq) * gOut;
            giy += lbsw_val * (ix_rtne - ix) * (iz - iz_rtne) * (iq_rtne - iq) * gOut;
            giz += lbsw_val * (ix_rtne - ix) * (iy - iy_rtne) * (iq_rtne - iq) * gOut;
            giq -= lbsw_val * (ix_rtne - ix) * (iy - iy_rtne) * (iz - iz_rtne) * gOut;
          }
          if (within_bounds_4d(iq_lbse, iz_lbse, iy_lbse, ix_lbse, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t lbse_val = inp_ptr_NC[iq_lbse * inp_sL + iz_lbse * inp_sD + iy_lbse * inp_sH + ix_lbse * inp_sW];
            gix += lbse_val * (iy - iy_rtnw) * (iz - iz_rtnw) * (iq_rtnw - iq) * gOut;
            giy += lbse_val * (ix - ix_rtnw) * (iz - iz_rtnw) * (iq_rtnw - iq) * gOut;
            giz += lbse_val * (ix - ix_rtnw) * (iy - iy_rtnw) * (iq_rtnw - iq) * gOut;
            giq -= lbse_val * (ix - ix_rtnw) * (iy - iy_rtnw) * (iz - iz_rtnw) * gOut;
          }
          if (within_bounds_4d(iq_rtnw, iz_rtnw, iy_rtnw, ix_rtnw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rtnw_val = inp_ptr_NC[iq_rtnw * inp_sL + iz_rtnw * inp_sD + iy_rtnw * inp_sH + ix_rtnw * inp_sW];
            gix -= rtnw_val * (iy_lbse - iy) * (iz_lbse - iz) * (iq_lbse - iq) * gOut;
            giy -= rtnw_val * (ix_lbse - ix) * (iz_lbse - iz) * (iq_lbse - iq) * gOut;
            giz -= rtnw_val * (ix_lbse - ix) * (iy_lbse - iy) * (iq_lbse - iq) * gOut;
            giq += rtnw_val * (ix_lbse - ix) * (iy_lbse - iy) * (iz_lbse - iz) * gOut;
          }
          if (within_bounds_4d(iq_rtne, iz_rtne, iy_rtne, ix_rtne, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rtne_val = inp_ptr_NC[iq_rtne * inp_sL + iz_rtne * inp_sD + iy_rtne * inp_sH + ix_rtne * inp_sW];
            gix += rtne_val * (iy_lbsw - iy) * (iz_lbsw - iz) * (iq_lbsw - iq) * gOut;
            giy -= rtne_val * (ix - ix_lbsw) * (iz_lbsw - iz) * (iq_lbsw - iq) * gOut;
            giz -= rtne_val * (ix - ix_lbsw) * (iy_lbsw - iy) * (iq_lbsw - iq) * gOut;
            giq += rtne_val * (ix - ix_lbsw) * (iy_lbsw - iy) * (iz_lbsw - iz) * gOut;
          }
          if (within_bounds_4d(iq_rtsw, iz_rtsw, iy_rtsw, ix_rtsw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rtsw_val = inp_ptr_NC[iq_rtsw * inp_sL + iz_rtsw * inp_sD + iy_rtsw * inp_sH + ix_rtsw * inp_sW];
            gix -= rtsw_val * (iy - iy_lbne) * (iz_lbne - iz) * (iq_lbne - iq) * gOut;
            giy += rtsw_val * (ix_lbne - ix) * (iz_lbne - iz) * (iq_lbne - iq) * gOut;
            giz -= rtsw_val * (ix_lbne - ix) * (iy - iy_lbne) * (iq_lbne - iq) * gOut;
            giq += rtsw_val * (ix_lbne - ix) * (iy - iy_lbne) * (iz_lbne - iz) * gOut;
          }
          if (within_bounds_4d(iq_rtse, iz_rtse, iy_rtse, ix_rtse, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rtse_val = inp_ptr_NC[iq_rtse * inp_sL + iz_rtse * inp_sD + iy_rtse * inp_sH + ix_rtse * inp_sW];
            gix += rtse_val * (iy - iy_lbnw) * (iz_lbnw - iz) * (iq_lbnw - iq) * gOut;
            giy += rtse_val * (ix - ix_lbnw) * (iz_lbnw - iz) * (iq_lbnw - iq) * gOut;
            giz -= rtse_val * (ix - ix_lbnw) * (iy - iy_lbnw) * (iq_lbnw - iq) * gOut;
            giq += rtse_val * (ix - ix_lbnw) * (iy - iy_lbnw) * (iz_lbnw - iz) * gOut;
          }
          if (within_bounds_4d(iq_rbnw, iz_rbnw, iy_rbnw, ix_rbnw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rbnw_val = inp_ptr_NC[iq_rbnw * inp_sL + iz_rbnw * inp_sD + iy_rbnw * inp_sH + ix_rbnw * inp_sW];
            gix -= rbnw_val * (iy_ltse - iy) * (iz - iz_ltse) * (iq_ltse - iq) * gOut;
            giy -= rbnw_val * (ix_ltse - ix) * (iz - iz_ltse) * (iq_ltse - iq) * gOut;
            giz += rbnw_val * (ix_ltse - ix) * (iy_ltse - iy) * (iq_ltse - iq) * gOut;
            giq += rbnw_val * (ix_ltse - ix) * (iy_ltse - iy) * (iz - iz_ltse) * gOut;
          }
          if (within_bounds_4d(iq_rbne, iz_rbne, iy_rbne, ix_rbne, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rbne_val = inp_ptr_NC[iq_rbne * inp_sL + iz_rbne * inp_sD + iy_rbne * inp_sH + ix_rbne * inp_sW];
            gix += rbne_val * (iy_ltsw - iy) * (iz - iz_ltsw) * (iq_ltsw - iq) * gOut;
            giy -= rbne_val * (ix - ix_ltsw) * (iz - iz_ltsw) * (iq_ltsw - iq) * gOut;
            giz += rbne_val * (ix - ix_ltsw) * (iy_ltsw - iy) * (iq_ltsw - iq) * gOut;
            giq += rbne_val * (ix - ix_ltsw) * (iy_ltsw - iy) * (iz - iz_ltsw) * gOut;
          }
          if (within_bounds_4d(iq_rbsw, iz_rbsw, iy_rbsw, ix_rbsw, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rbsw_val = inp_ptr_NC[iq_rbsw * inp_sL + iz_rbsw * inp_sD + iy_rbsw * inp_sH + ix_rbsw * inp_sW];
            gix -= rbsw_val * (iy - iy_ltne) * (iz - iz_ltne) * (iq_ltne - iq) * gOut;
            giy += rbsw_val * (ix_ltne - ix) * (iz - iz_ltne) * (iq_ltne - iq) * gOut;
            giz += rbsw_val * (ix_ltne - ix) * (iy - iy_ltne) * (iq_ltne - iq) * gOut;
            giq += rbsw_val * (ix_ltne - ix) * (iy - iy_ltne) * (iz - iz_ltne) * gOut;
          }
          if (within_bounds_4d(iq_rbse, iz_rbse, iy_rbse, ix_rbse, inp_L, inp_D, inp_H, inp_W)) {
            scalar_t rbse_val = inp_ptr_NC[iq_rbse * inp_sL + iz_rbse * inp_sD + iy_rbse * inp_sH + ix_rbse * inp_sW];
            gix += rbse_val * (iy - iy_ltnw) * (iz - iz_ltnw) * (iq_ltnw - iq) * gOut;
            giy += rbse_val * (ix - ix_ltnw) * (iz - iz_ltnw) * (iq_ltnw - iq) * gOut;
            giz += rbse_val * (ix - ix_ltnw) * (iy - iy_ltnw) * (iq_ltnw - iq) * gOut;
            giq += rbse_val * (ix - ix_ltnw) * (iy - iy_ltnw) * (iz - iz_ltnw) * gOut;
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NLDHW
        //   2. directly assign to gGrid_ptr_NLDHW[0], gGrid_ptr_NLDHW[1], gGrid_ptr_NLDHW[2], gGrid_ptr_NLDHW[3]
        scalar_t *gGrid_ptr_NLDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NLDHW[0] = gix_mult * gix;
        gGrid_ptr_NLDHW[1] = giy_mult * giy;
        gGrid_ptr_NLDHW[2] = giz_mult * giz;
        gGrid_ptr_NLDHW[3] = giq_mult * giq;
      }
      else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        if (input_requires_grad) {
          auto ix_nearest = static_cast<index_t>(::round(ix));
          auto iy_nearest = static_cast<index_t>(::round(iy));
          auto iz_nearest = static_cast<index_t>(::round(iz));
          auto iq_nearest = static_cast<index_t>(::round(iq));

          // assign nearest neighor pixel value to output pixel
          scalar_t *gOut_ptr_NCLDHW = grad_output.data + n * gOut_sN + l * gOut_sL + d * gOut_sD + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, gOut_ptr_NCLDHW += gOut_sC, NC_offset += gInp_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_4d(grad_input.data, iq_nearest, iz_nearest, iy_nearest, ix_nearest, gInp_sL, gInp_sD, gInp_sH, gInp_sW,
                        inp_L, inp_D, inp_H, inp_W, *gOut_ptr_NCLDHW, NC_offset, grad_input_memory_span);
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NLDHW
        //   2. directly assign to gGrid_ptr_NLDHW[0], gGrid_ptr_NLDHW[1], gGrid_ptr_NLDHW[2], gGrid_ptr_NLDHW[3]
        scalar_t *gGrid_ptr_NLDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NLDHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NLDHW[1] = static_cast<scalar_t>(0);
        gGrid_ptr_NLDHW[2] = static_cast<scalar_t>(0);
        gGrid_ptr_NLDHW[3] = static_cast<scalar_t>(0);
      }
    }
  }
}

void launch_grid_sampler_4d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_4d(input, grid, interpolation_mode);

  auto N = input.size(0);
  auto L = grid.size(1);
  auto D = grid.size(2);
  auto H = grid.size(3);
  auto W = grid.size(4);
  int64_t count = N * L * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_4d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_4d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
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
          <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
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

void launch_grid_sampler_4d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase& grad_output, const TensorBase& input,
    const TensorBase& grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool,2> output_mask)
{
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_4d(input, grid, interpolation_mode);

  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("grid_sampler_4d_backward_cuda");
  auto N = input.size(0);
  auto L = grid.size(1);
  auto D = grid.size(2);
  auto H = grid.size(3);
  auto W = grid.size(4);
  int64_t count = N * L * D * H * W;
  auto input_requires_grad = output_mask[0];
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_4d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        grid_sampler_4d_backward_kernel<scalar_t>
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
        grid_sampler_4d_backward_kernel<scalar_t>
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

Tensor grid_sampler_4d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3], grid_size[4]},
      input.options());
  launch_grid_sampler_4d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}


std::tuple<Tensor, Tensor>
grid_sampler_4d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool,2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_4d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

class GridSample4d : public Function<GridSample4d> {
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
        {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3], grid_size[4]},
        input.options());
      launch_grid_sampler_4d_forward_kernel(
        output, input, grid, interpolation_mode, padding_mode, align_corners);
      ctx->save_for_backward({input, grid});
      return output;
    }

    static tensor_list backward(AutogradContext *ctx,
                                tensor_list grad_outputs
    ) {
      const auto saved = ctx->get_saved_variables();
      const Tensor input = saved[0];
      const Tensor grid = saved[1];

      auto input_requires_grad = ctx->needs_input_grad[0];
      Tensor grad_input = ([&]() {
        if (input_requires_grad) {
          return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        } else {
          return Tensor();
        }
      })();
      auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      launch_grid_sampler_4d_backward_kernel(
          grad_input, grad_grid, grad_output, input,
          grid, interpolation_mode, padding_mode, align_corners, output_mask);

      return {grad_input, grad_grid, Tensor(), Tensor(), Tensor()};
    }
};

Tensor grid_sample_4d(Tensor &input,
               Tensor &grid,
               int64_t interpolation_mode,
               int64_t padding_mode,
               bool align_corners)
{
  return GridSample4d::apply(input, grid, interpolation_mode, padding_mode, align_corners);
}

static auto registry = torch::RegisterOperators()
                        .op("plenoxels::grid_sample_4d", &grid_sample_4d);
