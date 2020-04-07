#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

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
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <c10/macros/Macros.h>
#include <ATen/native/im2col_shape_check.h>

#include <cuda_profiler_api.h>
namespace at {
namespace native {


using namespace at::cuda;
using namespace at::cuda::detail;

#include "im2col.inc"

#define THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM 100
#define THRESH_NUMBER_BINS_FOR_GLOBAL_MEM 1000
#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
       i += gridDim.x * blockDim.x)

/*
  Memory types used for the 3 histogram implementations.
  See `CUDA_tensor_histogram` below.
 */
enum class CUDAHistogramMemoryType { SHARED, MULTI_BLOCK, GLOBAL };
namespace {
  template<typename input_t, typename IndexType>
  __device__ static IndexType getBin(input_t bVal, input_t minvalue, input_t maxvalue, int nbins) {
    IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
    // (only applicable for histc)
    // while each bin is inclusive at the lower end and exclusive at the higher, i.e. [start, end)
    // the last bin is inclusive at both, i.e. [start, end], in order to include maxvalue if exists
    // therefore when bin == nbins, adjust bin to the last bin
    if (bin == nbins) bin -= 1;
    return bin;
  }
}

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename output_t,
    typename input_t,
    typename IndexType,
    int ADims,
    int PDims,
    int BDims,
    CUDAHistogramMemoryType MemoryType = CUDAHistogramMemoryType::MULTI_BLOCK,
    typename Op>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void kernelHistogram1D(
    TensorInfo<output_t, IndexType> a, /* output */
    TensorInfo<output_t, IndexType> p, /* partial output */
    TensorInfo<input_t, IndexType> b, /* input */
    int nbins,
    input_t minvalue,
    input_t maxvalue,
    IndexType totalElements,
    Op getOp) {
  extern __shared__ unsigned char my_smem[];
  output_t* smem = nullptr;

    ////////////////////////// Shared memory //////////////////////////
    // atomically add to block specific shared memory
    // then atomically add to the global output tensor
    smem = reinterpret_cast<output_t*>(my_smem);
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      smem[i] = 0;
    }
    __syncthreads();
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          IndexToOffset<input_t, IndexType, BDims>::get(linearIndex, b);
      const input_t bVal = b.data[bOffset];
      if (bVal >= minvalue && bVal <= maxvalue) {
        // Use value at `b` as an offset of `smem`
        const IndexType bin = getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
        atomicAdd(&smem[bin], getOp(linearIndex));
      }
    }
    __syncthreads();
    // NOTE: atomically update output bin count.
    //   Atomic update is imp since __syncthread() will only synchronize threads
    //   in a given block, not across blocks.
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      const IndexType aOffset =
          IndexToOffset<output_t, IndexType, ADims>::get(i, a);
      atomicAdd(&a.data[aOffset], smem[i]);
    }

}

#include "im2col_kernel_kernelHistogram1D_.inc"

inline int64_t getFreeGlobalMemory() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  AT_ASSERTM(
      cudaGetLastError() == cudaSuccess,
      "CUDA_tensor_histogram failed to get free global memory");
  return static_cast<int64_t>(free_mem);
}

template <typename input_hist_t>
std::tuple<Tensor, Tensor> _histc_cuda_template_fused(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  printf("2\n");
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output_hist = native::zeros({nbins}, device(DeviceType::CUDA).dtype(self_hist.scalar_type()));
  input_hist_t minvalue = min;
  input_hist_t maxvalue = max;
  if (min == max) {
    minvalue = *self_hist.min().cpu().data<input_hist_t>();
    maxvalue = *self_hist.max().cpu().data<input_hist_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  printf("3\n");
  {
  checkBackend("CUDA_tensor_histogram", {output_hist, self_hist}, Backend::CUDA);
  auto totalElements = self_hist.numel();

  const dim3 block = getApplyBlock();
  dim3 grid;
  int64_t curDevice = current_device();

  grid.x = 10000;

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(input_hist_t) + 8; // 8 guard bytes
  auto maxGlobalMem = getFreeGlobalMemory();
  auto multiBlockMem = nbins * grid.x * sizeof(input_hist_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
    printf("6\n");
  if (nbins < THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM &&
      sharedMem < maxSharedMem) {
    printf("shared\n");
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (
      nbins < THRESH_NUMBER_BINS_FOR_GLOBAL_MEM &&
      multiBlockMem < (maxGlobalMem / 2)) {
    // check against half of free mem to be extra safe
    // due to cached allocator, we may anyway have slightly more free mem
    printf("mb\n");
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  }

  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = getTensorInfo<input_hist_t, IndexType>(output_hist);
  auto bInfo = getTensorInfo<input_hist_t, IndexType>(self_hist);
  TensorInfo<input_hist_t, IndexType> pInfo(nullptr, 0, {}, {});
  Tensor partial_output_hist;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    partial_output_hist = native::zeros({grid.x, nbins}, output_hist.options());
    pInfo = getTensorInfo<input_hist_t, IndexType>(partial_output_hist);
  }

  printf("7\n");
  printf("10\n");
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    int64_t elt = 0;
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };

  cudaProfilerStart();
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_bar_sync_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_bar_sync_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_1
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_1
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_lb_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 512, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 512, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);

    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_bar_sync_imba_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_bar_sync_imba_idx_0
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        output_n.data<scalar_t>(),
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    // im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_imba_idx_1
    // <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    // <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
    //     num_kernels,
    //     input_n.data<scalar_t>(),
    //     input_height,
    //     input_width,
    //     kernel_height,
    //     kernel_width,
    //     pad_height,
    //     pad_width,
    //     stride_height,
    //     stride_width,
    //     dilation_height,
    //     dilation_width,
    //     output_height,
    //     output_width,
    //     output_n.data<scalar_t>(),
    //     aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    // im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_imba_idx_1
    // <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    // <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
    //     num_kernels,
    //     input_n.data<scalar_t>(),
    //     input_height,
    //     input_width,
    //     kernel_height,
    //     kernel_width,
    //     pad_height,
    //     pad_width,
    //     stride_height,
    //     stride_width,
    //     dilation_height,
    //     dilation_width,
    //     output_height,
    //     output_width,
    //     output_n.data<scalar_t>(),
    //     aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    // im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_imba_idx_0
    // <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    // <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
    //     num_kernels,
    //     input_n.data<scalar_t>(),
    //     input_height,
    //     input_width,
    //     kernel_height,
    //     kernel_width,
    //     pad_height,
    //     pad_width,
    //     stride_height,
    //     stride_width,
    //     dilation_height,
    //     dilation_width,
    //     output_height,
    //     output_width,
    //     output_n.data<scalar_t>(),
    //     aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    // im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_imba_idx_0
    // <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    // <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
    //     num_kernels,
    //     input_n.data<scalar_t>(),
    //     input_height,
    //     input_width,
    //     kernel_height,
    //     kernel_width,
    //     pad_height,
    //     pad_width,
    //     stride_height,
    //     stride_width,
    //     dilation_height,
    //     dilation_width,
    //     output_height,
    //     output_width,
    //     output_n.data<scalar_t>(),
    //     aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
  cudaProfilerStop();
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  cudaDeviceSynchronize();
  return std::make_tuple(output_hist, output);
}
}

template <typename input_hist_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  printf("2\n");
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output_hist = native::zeros({nbins}, device(DeviceType::CUDA).dtype(self_hist.scalar_type()));
  input_hist_t minvalue = min;
  input_hist_t maxvalue = max;
  if (min == max) {
    minvalue = *self_hist.min().cpu().data<input_hist_t>();
    maxvalue = *self_hist.max().cpu().data<input_hist_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  printf("3\n");
  {
  checkBackend("CUDA_tensor_histogram", {output_hist, self_hist}, Backend::CUDA);
  auto totalElements = self_hist.numel();

  const dim3 block = getApplyBlock();
  dim3 grid;
  int64_t curDevice = current_device();

  grid.x = 10000;

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(input_hist_t) + 8; // 8 guard bytes
  auto maxGlobalMem = getFreeGlobalMemory();
  auto multiBlockMem = nbins * grid.x * sizeof(input_hist_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
    printf("6\n");
  if (nbins < THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM &&
      sharedMem < maxSharedMem) {
    printf("shared\n");
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (
      nbins < THRESH_NUMBER_BINS_FOR_GLOBAL_MEM &&
      multiBlockMem < (maxGlobalMem / 2)) {
    // check against half of free mem to be extra safe
    // due to cached allocator, we may anyway have slightly more free mem
    printf("mb\n");
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  }

  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = getTensorInfo<input_hist_t, IndexType>(output_hist);
  auto bInfo = getTensorInfo<input_hist_t, IndexType>(self_hist);
  TensorInfo<input_hist_t, IndexType> pInfo(nullptr, 0, {}, {});
  Tensor partial_output_hist;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    partial_output_hist = native::zeros({grid.x, nbins}, output_hist.options());
    pInfo = getTensorInfo<input_hist_t, IndexType>(partial_output_hist);
  }

  printf("7\n");
  printf("10\n");
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    int64_t elt = 0;
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    cudaProfilerStart();
    im2col_kernel<<<10000, 512, 0, at::cuda::getStreamFromPool(true)>>>(
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

    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
        <<<grid,
          block,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);        \
    cudaProfilerStop();
    cudaDeviceSynchronize();
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  return std::make_tuple(output_hist, output);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> _histc_cuda2(
  const Tensor& input_im2col_,
  IntArrayRef kernel_size_im2col,
  IntArrayRef dilation_im2col,
  IntArrayRef pad_im2colding_im2col,
  IntArrayRef stride_im2col,
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
    input_im2col_,
    kernel_size_im2col,
    dilation_im2col,
    pad_im2colding_im2col,
    stride_im2col
  );
  });
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template_fused<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
    input_im2col_,
    kernel_size_im2col,
    dilation_im2col,
    pad_im2colding_im2col,
    stride_im2col
  );
  });
}

} // namespace native
} // namespace at
