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


template <template<typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5,
    const stat_accscalar_t3 epsilon6,
    const stat_accscalar_t3 momentum7,
    PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8,
    PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9,
    PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10,
    PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) {
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx.x + threadIdx.y * blockDim.x;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx.y; batch < input5.size(0); batch += blockDim.y) {
        for (int x = threadIdx.x; x < input5.size(2); x += blockDim.x) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    __syncthreads();
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    __syncthreads();
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim.x * blockDim.y / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim.x * blockDim.y / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
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
#include "upsample.inc3"
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

  dim3 blocks(10000);
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
        #define CALL(i,type,thread) upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_##type##_idx_##i<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>\
            <<<blocks, thread, 0, stream1>>>(\
              num_kernels, rheight, rwidth, align_corners, idata, odata,\
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);\
              cudaDeviceSynchronize()
      CALL(0, vfuse,512);
      CALL(0, vfuse_lb,512);
      CALL(0, hfuse,1024);
      CALL(0, hfuse_lb,1024);
      CALL(1, hfuse,1024);
      CALL(1, hfuse_lb,1024);
      CALL(2, hfuse,1024);
      CALL(2, hfuse_lb,1024);
      CALL(3, hfuse,1024);
      CALL(3, hfuse_lb,1024);
      CALL(4, hfuse,1024);
      CALL(4, hfuse_lb,1024);
      CALL(5, hfuse,1024);
      CALL(5, hfuse_lb,1024);
      CALL(6, hfuse,1024);
      CALL(6, hfuse_lb,1024);
        cudaProfilerStop();
      });

  AT_CUDA_CHECK(cudaGetLastError());
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, mean_);
}


} } // namespace at::native
