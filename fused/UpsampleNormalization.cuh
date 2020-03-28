#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"
#include <cuda_profiler_api.h>

namespace at { namespace native {

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

  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  stat_accscalar_t* shared_avg_var = (stat_accscalar_t*) &shared_n[WARP_SIZE];

  // first the reductions each thread does separately
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

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean.data() != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var.data() != NULL) {
      save_transformed_var[plane] = VarTransform<stat_accscalar_t>{}(var_n / N, epsilon);
    }
    if (running_mean.data() != NULL) {
      running_mean[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_mean[plane] + momentum * avg);
    }
    if (running_var.data() != NULL) {
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
    const int n,
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

  if (index < n) {
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
    for (int n = 0; n < batchsize-1; n++) {
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
    for (int n = batchsize-1; n < batchsize; n++) {
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

#include "upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_.inc"

template<typename scalar_t_bn, typename index_t_bn>
std::tuple<Tensor, Tensor> upsample_batchnorm_stm(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
  const Tensor& input_bn_, double epsilon) {
  Tensor output = at::empty_like(input);
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_bilinear2d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({input.size(0), input.size(1), output_height, output_width});

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

  printf("%d %d\n", num_kernels, num_threads);
  cudaStream_t stream = at::cuda::getStreamFromPool(true);

  using accscalar_t_bn = at::acc_type<scalar_t_bn, true>;
  int64_t n_input_bn = input_bn_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_bn_reshaped = input_bn_.reshape({input_bn_.size(0), input_bn_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_bn_reshaped.size(0);
  auto features = input_bn_reshaped.size(2);
  auto input_bn = input_bn_reshaped.packed_accessor<scalar_t_bn, 3, RestrictPtrTraits, index_t_bn>();
  auto input_bn_options = input_bn_.options();
  dummy_mean_ = at::empty({0}, input_bn_options);
  dummy_var_ = at::empty({0}, input_bn_options);
  // promote only mean_/invstd_ precision
  if (input_bn_.scalar_type() == at::ScalarType::Half) {
    input_bn_options = input_bn_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input_bn}, input_bn_options);
  invstd_ = at::empty({n_input_bn}, input_bn_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto stream1 = at::cuda::getStreamFromPool();

  dim3 blocks(input_bn.size(1));
  int tf = getNumThreads(input_bn.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
        printf("%d\n", num_blocks);
        cudaProfilerStart();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
            <<<num_blocks,
               num_threads,
               0,
               stream>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
        batch_norm_collect_statistics_kernel<InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn> <<<blocks, threads, 0, stream1>>>
          (input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("time: %f\n", milliseconds);
        cudaProfilerStop();
      });

  AT_CUDA_CHECK(cudaGetLastError());
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, mean_);
}

template<typename scalar_t_bn, typename index_t_bn>
std::tuple<Tensor, Tensor> upsample_batchnorm_fused(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
  const Tensor& input_bn_, double epsilon) {
  Tensor output = at::empty_like(input);
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_bilinear2d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({input.size(0), input.size(1), output_height, output_width});

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 512);

  printf("%d %d\n", num_kernels, num_threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using accscalar_t_bn = at::acc_type<scalar_t_bn, true>;
  int64_t n_input_bn = input_bn_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_bn_reshaped = input_bn_.reshape({input_bn_.size(0), input_bn_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_bn_reshaped.size(0);
  auto features = input_bn_reshaped.size(2);
  auto input_bn = input_bn_reshaped.packed_accessor<scalar_t_bn, 3, RestrictPtrTraits, index_t_bn>();
  auto input_bn_options = input_bn_.options();
  dummy_mean_ = at::empty({0}, input_bn_options);
  dummy_var_ = at::empty({0}, input_bn_options);
  // promote only mean_/invstd_ precision
  if (input_bn_.scalar_type() == at::ScalarType::Half) {
    input_bn_options = input_bn_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input_bn}, input_bn_options);
  invstd_ = at::empty({n_input_bn}, input_bn_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto stream1 = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input_bn.size(1));
  int tf = getNumThreads(input_bn.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
        printf("%d\n", num_blocks);
        printf("%d %d\n", idata.size(0), idata.size(1));
        cudaProfilerStart();

        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);
        cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 1024, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 1024, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_bar_sync_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 1024, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_bar_sync_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 1024, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 512, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
        cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 512, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
        cudaDeviceSynchronize();
        cudaProfilerStop();
      });

  AT_CUDA_CHECK(cudaGetLastError());
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, mean_);
}


} } // namespace at::native
