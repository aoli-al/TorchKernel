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

#include "im2col.inc3"

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}
#include "upsample.inc1"

#include "im2col_kernel_upsample_bilinear2d_out_frame_.inc"

} // namespace native
} // namespace at
