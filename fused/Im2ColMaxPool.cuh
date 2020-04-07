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

namespace at {
namespace native {

using namespace at::cuda::detail;

static const int BACKWARD_THREADS = 256;
// kernels borrowed from Caffe

#include "maxpool.inc"
#include "im2col.inc"

#include "im2col_kernel_MaxPoolForward_.inc"

} // namespace native
} // namespace at
