#pragma once

#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <c10/macros/Macros.h>

#include <ATen/native/im2col_shape_check.h>
#include <ATen/AccumulateType.h>
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"

namespace at {
namespace native {

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct SumOp {
  __device__ SumOp(const PTA& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PTA& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template<typename T>
struct InvStd {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / device_sqrt(var + epsilon);
    }
    return invstd;
  }
};

template<typename T>
struct Var {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};

template <template<typename T> class VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {
    static int shared_n[160] __attribute__((shared));
    int plane = blockIdx.x;
    int N = input.size(0) * input.size(2);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    stat_accscalar_t *shared_avg_var = (stat_accscalar_t *)&shared_n[WARP_SIZE];
    stat_accscalar_t avg = 0;
    stat_accscalar_t var_n = 0;
    int n = 0;
    for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
        for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
            stat_accscalar_t v = input[batch][plane][x];
            stat_accscalar_t d1 = v - avg;
            n++;
            avg += d1 / n;
            var_n += d1 * (v - avg);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
        stat_accscalar_t factor = 1. / fmaxf(1., n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        shared_n[tid / WARP_SIZE] = n;
        shared_avg_var[tid / WARP_SIZE * 2] = avg;
        shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
        avg = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
        var_n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
        int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
        stat_accscalar_t factor = 1. / fmaxf(1., n + o_n);
        var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
        avg = (n * avg + o_n * o_avg) * factor;
        n += o_n;
    }
    if (tid == 0) {
        if (save_mean.data() != __null) {
            save_mean[plane] = avg;
        }
        if (save_transformed_var.data() != __null) {
            save_transformed_var[plane] = VarTransform<stat_accscalar_t>({})(var_n / N, epsilon);
        }
        if (running_mean.data() != __null) {
            running_mean[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_mean[plane] + momentum * avg);
        }
        if (running_var.data() != __null) {
            stat_accscalar_t unbiasedVar = var_n / (N - 1);
            running_var[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_var[plane] + momentum * unbiasedVar);
        }
    }
}


template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

using namespace at::cuda::detail;

#include "im2col.inc2"

#include "im2col_kernel_batch_norm_collect_statistics_kernel_.inc"

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor> im2col_batch_norm_stream(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_batch_norm_, double epsilon) {

  Tensor output = at::empty_like(input_);
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

  im2col_shape_check(
    input_,
    Tensor(),
    kernel_height,
    kernel_width,
    dilation_height,
    dilation_width,
    pad_height,
    pad_width,
    stride_height,
    stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  // Launch kernel
  using accscalar_t_batch_norm = at::acc_type<scalar_t_batch_norm, true>;
  int64_t n_input = input_batch_norm_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_batch_norm_reshaped = input_batch_norm_.reshape({input_batch_norm_.size(0), input_batch_norm_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_batch_norm_reshaped.size(0);
  auto features = input_batch_norm_reshaped.size(2);
  auto input_batch_norm = input_batch_norm_reshaped.packed_accessor<scalar_t_batch_norm, 3, RestrictPtrTraits, index_t_batch_norm>();
  auto input_batch_norm_options = input_batch_norm_.options();
  dummy_mean_ = at::empty({0}, input_batch_norm_options);
  dummy_var_ = at::empty({0}, input_batch_norm_options);
  // promote only mean_/invstd_ precision
  if (input_batch_norm_.scalar_type() == at::ScalarType::Half) {
    input_batch_norm_options = input_batch_norm_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_batch_norm_options);
  invstd_ = at::empty({n_input}, input_batch_norm_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto stream1 = at::cuda::getStreamFromPool(true);
  auto stream2 = at::cuda::getStreamFromPool(true);

  dim3 blocks(input_batch_norm.size(1));
  int tf = getNumThreads(input_batch_norm.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    input_n = input.select(0, 0);
    output_n = output.select(0, 0);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    printf("nk: %ld\n", num_kernels);
    const int num_of_threads = 1024;
    const int num_of_blocks = (num_kernels + num_of_threads - 1) / num_of_threads;


    cudaProfilerStart();
    im2col_kernel<scalar_t><<<10000, 1024, 0, stream2>>>(
        num_kernels,
        input_n.data<scalar_t>(),
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output_height,
        output_width,
        output_n.data<scalar_t>());


    AT_CUDA_CHECK(cudaGetLastError());
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  batch_norm_collect_statistics_kernel<InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm> <<<blocks, threads, 0, stream1>>>
    (input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  cudaProfilerStop();
  cudaDeviceSynchronize();
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output);
}

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor> im2col_batch_norm_fused(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_batch_norm_, double epsilon) {

  Tensor output = at::empty_like(input_);
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

  im2col_shape_check(
    input_,
    Tensor(),
    kernel_height,
    kernel_width,
    dilation_height,
    dilation_width,
    pad_height,
    pad_width,
    stride_height,
    stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  // Launch kernel
  using accscalar_t_batch_norm = at::acc_type<scalar_t_batch_norm, true>;
  int64_t n_input = input_batch_norm_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_batch_norm_reshaped = input_batch_norm_.reshape({input_batch_norm_.size(0), input_batch_norm_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_batch_norm_reshaped.size(0);
  auto features = input_batch_norm_reshaped.size(2);
  auto input_batch_norm = input_batch_norm_reshaped.packed_accessor<scalar_t_batch_norm, 3, RestrictPtrTraits, index_t_batch_norm>();
  auto input_batch_norm_options = input_batch_norm_.options();
  dummy_mean_ = at::empty({0}, input_batch_norm_options);
  dummy_var_ = at::empty({0}, input_batch_norm_options);
  // promote only mean_/invstd_ precision
  if (input_batch_norm_.scalar_type() == at::ScalarType::Half) {
    input_batch_norm_options = input_batch_norm_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_batch_norm_options);
  invstd_ = at::empty({n_input}, input_batch_norm_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();

  dim3 blocks(input_batch_norm.size(1));
  int tf = getNumThreads(input_batch_norm.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  Tensor input_n;
  Tensor output_n;

  input_n = input.select(0, 0);
  output_n = output.select(0, 0);
  int64_t num_kernels = n_input_plane * output_height * output_width;
  printf("nk: %ld\n", num_kernels);
  const int num_of_threads = 512;
  const int num_of_blocks = 10000; 

  printf("kh: %d, kw: %d\n", kernel_height, kernel_width);
  printf("nb: %ld\n", num_of_blocks);

  cudaProfilerStart();
      cudaDeviceSynchronize();
      cudaDeviceSynchronize();
    #define CALL(i, type,thread) im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_##type##_idx_##i<scalar_t_batch_norm, InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm>\
    <<<num_of_blocks, thread, 0, at::cuda::getCurrentCUDAStream()>>>(\
      num_kernels,\
      input_n.data<scalar_t_batch_norm>(),\
      input_height,\
      input_width,\
      kernel_height,\
      kernel_width,\
      pad_height,\
      pad_width,\
      stride_height,\
      stride_width,\
      dilation_height,\
      dilation_width,\
      output_height,\
      output_width,\
      output_n.data<scalar_t_batch_norm>(),\
      input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);\
      cudaDeviceSynchronize()
    CALL(0, vfuse, 512);
    CALL(0, vfuse_lb, 512);
    CALL(0, hfuse, 1024);
    CALL(0, hfuse_lb, 1024);
    CALL(1, hfuse, 1024);
    CALL(1, hfuse_lb, 1024);
    CALL(2, hfuse, 1024);
    CALL(2, hfuse_lb, 1024);
    CALL(3, hfuse, 1024);
    CALL(3, hfuse_lb, 1024);
    CALL(4, hfuse, 1024);
    CALL(4, hfuse_lb, 1024);
    CALL(5, hfuse, 1024);
    CALL(5, hfuse_lb, 1024);
    CALL(6, hfuse, 1024);
    CALL(6, hfuse_lb, 1024);
  cudaProfilerStop();

  AT_CUDA_CHECK(cudaGetLastError());
  if (!batched_input) {
    output.resize_({n_output_plane, output_length});
  }
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output);
}
} // namespace native
} // namespace at
