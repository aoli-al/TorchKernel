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
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>
#include <c10/macros/Macros.h>

#include <c10/macros/Macros.h>
#include <ATen/native/im2col_shape_check.h>

#include <cuda_profiler_api.h>
namespace at {
namespace native {


#include "maxpool.inc"

static const int BACKWARD_THREADS = 256;


using namespace at::cuda;
using namespace at::cuda::detail;

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

inline int64_t getFreeGlobalMemory() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  AT_ASSERTM(
      cudaGetLastError() == cudaSuccess,
      "CUDA_tensor_histogram failed to get free global memory");
  return static_cast<int64_t>(free_mem);
}

#include "kernelHistogram1D_MaxPoolForward_.inc"

template <typename input_hist_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
  {
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

  Tensor output = at::empty({0}, input_.options());
  Tensor indices = at::empty({0}, input_.options().dtype(kLong));

  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {output_arg, indices_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  printf("%ld, %ld\n", outputWidth, outputHeight);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  Tensor input = input_.contiguous();

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  const int count = safe_downcast<int, int64_t>(output.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_data = output.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      int64_t *indices_data = indices.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads);
      printf("%d %d %d\n", count, blocks, num_threads);
      cudaProfilerStart();
            cudaDeviceSynchronize();
      MaxPoolForward<scalar_t, scalar_t>
        <<<10000, 256, 0, at::cuda::getStreamFromPool(true)>>>(
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
        <<<grid,
          block,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_1
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_1
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_bar_sync_idx_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_bar_sync_idx_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_lb_idx_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          512,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_idx_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          512,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    // kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_imba_idx_0
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       block.x + num_threads,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    //         cudaDeviceSynchronize();
    // kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_imba_idx_0
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       block.x + num_threads,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    //         cudaDeviceSynchronize();
    // kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_imba_idx_1
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       block.x + num_threads,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    //         cudaDeviceSynchronize();
    // kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_imba_idx_1
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       block.x + num_threads,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    //         cudaDeviceSynchronize();
    // kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_bar_sync_imba_idx_0
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       block.x + num_threads,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    //         cudaDeviceSynchronize();
    // kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_bar_sync_imba_idx_0
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       block.x + num_threads,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    //         cudaDeviceSynchronize();
      cudaProfilerStop();
    });


  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, output);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> _histc_maxpool(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,

           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode
  ) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
           input_,
           kernel_size,
           stride,
           padding,
           dilation,
           ceil_mode
  );
  });
}

} // namespace native
} // namespace at
