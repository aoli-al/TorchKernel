#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace at {
namespace native {
namespace {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

#define CUDA_MAX_THREADS 1024 // this is safe, in reality 256 is our limit

#define BLOCK_STRIDE 2 // increasing block_stride to lower # of blocks launched

static __device__ inline int p_start(int size, int pad, int kernel, int dilation, int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1)) ? 0 : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static __device__ inline int p_end(int size, int pad, int pooled_size, int stride) {
  return min((size + pad) / stride + 1, pooled_size);
}

static const int BLOCK_THREADS = 256;

template <typename scalar_t, typename accscalar_t>
#if defined (__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_2(BLOCK_THREADS, 4)
#else
C10_LAUNCH_BOUNDS_2(BLOCK_THREADS, 8)
#endif
__global__ void max_pool_backward_nchw(const int nthreads, const scalar_t* top_diff,
    const int64_t* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    scalar_t* bottom_diff) {
  CUDA_KERNEL_LOOP(index, height*width) {
    int h = index / width;
    int w = index - h * width;
    int phstart = p_start(h, pad_h, kernel_h, dilation_h, stride_h);
    int phend = p_end(h, pad_h, pooled_height, stride_h);
    int pwstart = p_start(w, pad_w, kernel_w, dilation_w, stride_w);
    int pwend = p_end(w, pad_w, pooled_width, stride_w);
    for (int n = blockIdx.y; n < num; n += gridDim.y) {
      for (int c = blockIdx.z; c < channels; c+= gridDim.z) {
        accscalar_t gradient = accscalar_t(0);
        int offset = (n * channels + c) * pooled_height * pooled_width;
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            if (top_mask[ph * pooled_width + pw + offset] == h * width + w) {
              gradient += ScalarConvert<scalar_t, accscalar_t>::to(top_diff[ph * pooled_width + pw + offset]);
            }
          }
        }
        bottom_diff[(n*channels+c)*height*width+index] = ScalarConvert<accscalar_t, scalar_t>::to(gradient);
      }
    }
  }
}


void max_pool2d_with_indices_backward_out_cuda_template(
           Tensor& gradInput,
           const Tensor& gradOutput_,
           const Tensor& input_,
           const Tensor& indices,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

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

  const auto memory_format = input_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else {
    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  }
  const Tensor input = input_.contiguous(memory_format);

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t in_stride_n = input.ndimension() == 4 ? input.stride(-4) : 0;
  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
    input_,
    gradOutput_,
    indices,
    nbatch,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth,
    /*cuda=*/ true);

  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t out_stride_c = gradOutput.stride(-3);
  const int64_t out_stride_h = gradOutput.stride(-2);
  const int64_t out_stride_w = gradOutput.stride(-1);

  gradInput.resize_as_(input);
  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  int64_t count = input.numel();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "max_pool2d_with_indices_out_cuda_frame", [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        switch (memory_format) {
          case MemoryFormat::Contiguous: {
            int imgcount = inputWidth * inputHeight;
            dim3 grid;
            const int blocks = (imgcount + BLOCK_THREADS - 1) / BLOCK_THREADS;
            grid.x = blocks;
            grid.y = nbatch;
            uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
            if (maxGridY < grid.y) grid.y = maxGridY;
            grid.z = nInputPlane;
            uint64_t maxGridZ = at::cuda::getCurrentDeviceProperties()->maxGridSize[2];
            if (maxGridZ < grid.z) grid.z = maxGridZ;

            max_pool_backward_nchw<scalar_t, accscalar_t>
            <<<grid, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                    gradOutput_data,
                    indices_data,
                    nbatch,
                    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                    gradInput_data);
            break;
          }
          default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      });
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& max_pool2d_with_indices_backward_out_cuda(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  max_pool2d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

Tensor max_pool2d_with_indices_backward_cuda(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_pool2d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

} // at::native
} // at
