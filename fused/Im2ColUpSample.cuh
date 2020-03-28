#pragma once

#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <c10/macros/Macros.h>
#include "../cuda/UpSample.cuh"
#include "../cuda/KernelUtils.cuh"
#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


#include <ATen/native/im2col_shape_check.h>
#include <ATen/AccumulateType.h>

namespace at {
namespace native {

using namespace at::cuda::detail;

#define CUDA_KERNEL_LOOP_C(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * 512 + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=512 * 10000, i=_i_n_d_e_x)
// Kernel for fast unfold+copy
// (borrowed from Caffe:
// https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
// CUDA_NUM_THREADS = 1024

template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_kernel(
    const int64_t n,
    const dt* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int64_t w_out = index % width_col;

    index /= width_col;

    int64_t h_out = index % height_col;
    int64_t channel_in = index / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width)
            ? data_im[i * dilation_height * width + j * dilation_width]
            : ScalarConvert<int, dt>::to(0);
        data_col += height_col * width_col;
      }
    }
  }
}

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(
    const int ns,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int height2 = odata.size(2);
  const int width2 = odata.size(3);

  if (index < ns) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][h1][w1];
          odata[n][c][h2][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = h0lambda *
                (w0lambda * idata[n][c][h1][w1] +
                 w1lambda * idata[n][c][h1][w1 + w1p]) +
            h1lambda *
                (w0lambda * idata[n][c][h1 + h1p][w1] +
                 w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
        odata[n][c][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

#include "im2col_kernel_upsample_bilinear2d_out_frame_.inc"

} // namespace native
} // namespace at
