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
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

#include <ATen/native/im2col_shape_check.h>
#include <ATen/AccumulateType.h>
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"

namespace at {
namespace native {

using namespace at::cuda::detail;

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

static const int BACKWARD_THREADS = 256;
// kernels borrowed from Caffe
template <typename scalar_t, typename accscalar_t>
__global__ void MaxPoolForward(const int nthreads, const scalar_t* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    accscalar_t maxval = at::numeric_limits<accscalar_t>::lower_bound(); // -Infinity
    int maxidx = hstart * width + wstart;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        scalar_t val = bottom_data[h * width + w];
        if ((ScalarConvert<scalar_t, accscalar_t>::to(val) > maxval) || THCNumerics<scalar_t>::isnan(val)) {
          maxidx = h * width + w;
          maxval = ScalarConvert<scalar_t, accscalar_t>::to(val);
        }
      }
    }
    top_data[index] = ScalarConvert<scalar_t, accscalar_t>::to(maxval);
    top_mask[index] = maxidx;
  }
}

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
// #if defined(__HIP_PLATFORM_HCC__)
// #else
// constexpr int MAX_BLOCK_SIZE = 512;
// #endif
constexpr int MAX_BLOCK_SIZE = 256;

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

using namespace at::cuda::detail;

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

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_stream(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size_maxpool,
    IntArrayRef stride_maxpool,
    IntArrayRef padding_maxpool,
    IntArrayRef dilation_maxpool,
    bool ceil_mode,
    const Tensor& input_batch_norm_, double epsilon) {

  Tensor output_maxpool = at::empty({0}, input_maxpool_.options());
  Tensor indices_maxpool = at::empty({0}, input_maxpool_.options().dtype(kLong));
  TensorArg output_maxpool_arg{ output_maxpool, "output_maxpool", 1 };
  TensorArg indices_maxpool_arg{ indices_maxpool, "indices_maxpool", 2 };
  TensorArg input_maxpool_arg{ input_maxpool_ , "input_maxpool_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_maxpool_out_cuda",
                  {output_maxpool_arg, indices_maxpool_arg, input_maxpool_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size_maxpool.size() == 1 || kernel_size_maxpool.size() == 2,
    "max_pool2d: kernel_size_maxpool must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size_maxpool[0]);
  const int kW = kernel_size_maxpool.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size_maxpool[1]);

  // NB: stride_maxpool default is not expressible as an integer constant, so we accept
  // empty stride_maxpool for this case
  TORCH_CHECK(stride_maxpool.size() == 0 || stride_maxpool.size() == 1 || stride_maxpool.size() == 2,
    "max_pool2d: stride_maxpool must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride_maxpool.empty() ? kH : safe_downcast<int, int64_t>(stride_maxpool[0]);
  const int dW = stride_maxpool.empty() ? kW :
                 stride_maxpool.size() == 1 ? dH : safe_downcast<int, int64_t>(stride_maxpool[1]);

  TORCH_CHECK(padding_maxpool.size() == 1 || padding_maxpool.size() == 2,
    "max_pool2d: padding_maxpool must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding_maxpool[0]);
  const int padW = padding_maxpool.size() == 1 ? padH : safe_downcast<int, int64_t>(padding_maxpool[1]);

  TORCH_CHECK(dilation_maxpool.size() == 1 || dilation_maxpool.size() == 2,
    "max_pool2d: dilation_maxpool must be either a single int, or a tuple of two ints");
  const int dilation_maxpoolH = safe_downcast<int, int64_t>(dilation_maxpool[0]);
  const int dilation_maxpoolW = dilation_maxpool.size() == 1 ? dilation_maxpoolH : safe_downcast<int, int64_t>(dilation_maxpool[1]);

  TORCH_CHECK((input_maxpool_.ndimension() == 3 || input_maxpool_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input_maxpool");

  const int64_t nbatch = input_maxpool_.ndimension() == 4 ? input_maxpool_.size(-4) : 1;
  const int64_t nInputPlane = input_maxpool_.size(-3);
  const int64_t input_maxpoolHeight = input_maxpool_.size(-2);
  const int64_t input_maxpoolWidth = input_maxpool_.size(-1);

  const int64_t output_maxpoolWidth = pooling_output_shape<int64_t>(input_maxpoolWidth, kW, padW, dW, dilation_maxpoolW, ceil_mode);
  const int64_t output_maxpoolHeight = pooling_output_shape<int64_t>(input_maxpoolHeight, kH, padH, dH, dilation_maxpoolH, ceil_mode);
  printf("%ld, %ld\n", output_maxpoolWidth, output_maxpoolHeight);

  pool2d_shape_check(
    input_maxpool_,
    kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW,
    nInputPlane,
    input_maxpoolHeight, input_maxpoolWidth,
    output_maxpoolHeight, output_maxpoolWidth);

  Tensor input_maxpool = input_maxpool_.contiguous();

  output_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  indices_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});

  const int count = safe_downcast<int, int64_t>(output_maxpool.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

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
    const int num_of_threads = 512;
    const int num_of_blocks = (num_kernels + num_of_threads - 1) / num_of_threads;
    cudaProfilerStart();
    im2col_kernel<scalar_t><<<num_of_blocks, num_of_threads, 0, stream2>>>(
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


      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_maxpool_data = output_maxpool.data<scalar_t>();
      scalar_t *input_maxpool_data = input_maxpool.data<scalar_t>();
      int64_t *indices_maxpool_data = indices_maxpool.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads);
      printf("%d %d %d\n", count, blocks, num_threads);
      MaxPoolForward<scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          count, input_maxpool_data,
          nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
          kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW, output_maxpool_data, indices_maxpool_data);
    batch_norm_collect_statistics_kernel<InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm> <<<blocks, threads, 0, stream1>>>
      (input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
    cudaDeviceSynchronize();
    cudaProfilerStop();
    AT_CUDA_CHECK(cudaGetLastError());
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  THCudaCheck(cudaGetLastError());
  if(input_maxpool.ndimension() == 3) {
    output_maxpool.resize_({nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  }
  return std::make_tuple(output, mean_, output_maxpool);
}

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_stream2(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size_maxpool,
    IntArrayRef stride_maxpool,
    IntArrayRef padding_maxpool,
    IntArrayRef dilation_maxpool,
    bool ceil_mode,
    const Tensor& input_batch_norm_, double epsilon) {

  Tensor output_maxpool = at::empty({0}, input_maxpool_.options());
  Tensor indices_maxpool = at::empty({0}, input_maxpool_.options().dtype(kLong));
  TensorArg output_maxpool_arg{ output_maxpool, "output_maxpool", 1 };
  TensorArg indices_maxpool_arg{ indices_maxpool, "indices_maxpool", 2 };
  TensorArg input_maxpool_arg{ input_maxpool_ , "input_maxpool_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_maxpool_out_cuda",
                  {output_maxpool_arg, indices_maxpool_arg, input_maxpool_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size_maxpool.size() == 1 || kernel_size_maxpool.size() == 2,
    "max_pool2d: kernel_size_maxpool must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size_maxpool[0]);
  const int kW = kernel_size_maxpool.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size_maxpool[1]);

  // NB: stride_maxpool default is not expressible as an integer constant, so we accept
  // empty stride_maxpool for this case
  TORCH_CHECK(stride_maxpool.size() == 0 || stride_maxpool.size() == 1 || stride_maxpool.size() == 2,
    "max_pool2d: stride_maxpool must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride_maxpool.empty() ? kH : safe_downcast<int, int64_t>(stride_maxpool[0]);
  const int dW = stride_maxpool.empty() ? kW :
                 stride_maxpool.size() == 1 ? dH : safe_downcast<int, int64_t>(stride_maxpool[1]);

  TORCH_CHECK(padding_maxpool.size() == 1 || padding_maxpool.size() == 2,
    "max_pool2d: padding_maxpool must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding_maxpool[0]);
  const int padW = padding_maxpool.size() == 1 ? padH : safe_downcast<int, int64_t>(padding_maxpool[1]);

  TORCH_CHECK(dilation_maxpool.size() == 1 || dilation_maxpool.size() == 2,
    "max_pool2d: dilation_maxpool must be either a single int, or a tuple of two ints");
  const int dilation_maxpoolH = safe_downcast<int, int64_t>(dilation_maxpool[0]);
  const int dilation_maxpoolW = dilation_maxpool.size() == 1 ? dilation_maxpoolH : safe_downcast<int, int64_t>(dilation_maxpool[1]);

  TORCH_CHECK((input_maxpool_.ndimension() == 3 || input_maxpool_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input_maxpool");

  const int64_t nbatch = input_maxpool_.ndimension() == 4 ? input_maxpool_.size(-4) : 1;
  const int64_t nInputPlane = input_maxpool_.size(-3);
  const int64_t input_maxpoolHeight = input_maxpool_.size(-2);
  const int64_t input_maxpoolWidth = input_maxpool_.size(-1);

  const int64_t output_maxpoolWidth = pooling_output_shape<int64_t>(input_maxpoolWidth, kW, padW, dW, dilation_maxpoolW, ceil_mode);
  const int64_t output_maxpoolHeight = pooling_output_shape<int64_t>(input_maxpoolHeight, kH, padH, dH, dilation_maxpoolH, ceil_mode);
  printf("%ld, %ld\n", output_maxpoolWidth, output_maxpoolHeight);

  pool2d_shape_check(
    input_maxpool_,
    kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW,
    nInputPlane,
    input_maxpoolHeight, input_maxpoolWidth,
    output_maxpoolHeight, output_maxpoolWidth);

  Tensor input_maxpool = input_maxpool_.contiguous();

  output_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  indices_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});

  const int count = safe_downcast<int, int64_t>(output_maxpool.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

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
    im2col_kernel<scalar_t><<<num_of_blocks, num_of_threads, 0, stream2>>>(
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


      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_maxpool_data = output_maxpool.data<scalar_t>();
      scalar_t *input_maxpool_data = input_maxpool.data<scalar_t>();
      int64_t *indices_maxpool_data = indices_maxpool.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads);
      printf("%d %d %d\n", count, blocks, num_threads);
      MaxPoolForward<scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          count, input_maxpool_data,
          nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
          kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW, output_maxpool_data, indices_maxpool_data);
    batch_norm_collect_statistics_kernel<InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm> <<<blocks, threads, 0, stream1>>>
      (input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
    cudaDeviceSynchronize();
    cudaProfilerStop();
    AT_CUDA_CHECK(cudaGetLastError());
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  THCudaCheck(cudaGetLastError());
  if(input_maxpool.ndimension() == 3) {
    output_maxpool.resize_({nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  }
  return std::make_tuple(output, mean_, output_maxpool);
}

} // namespace native
} // namespace at
