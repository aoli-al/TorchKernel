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
#include <ATen/native/im2col_shape_check.h>

namespace at {
namespace native {

using namespace at::cuda::detail;

#include "col2im.inc2"
#include "maxpool_backward.inc"
#include "col2im_kernel_max_pool_backward_nchw_.inc"

Tensor col2im_batch_norm_backward(
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride, 
    const Tensor& gradOutput_,
    const Tensor& input_m_,
    const Tensor& indices,
    IntArrayRef kernel_size_m,
    IntArrayRef stride_m,
    IntArrayRef padding_m,
    IntArrayRef dilation_m,
    bool ceil_mode
  ) {

  auto gradInput = at::zeros_like(input_m_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_m_arg{ input_m_, "input_m_", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_m_arg, indices_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size_m.size() == 1 || kernel_size_m.size() == 2,
    "max_pool2d: kernel_size_m must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size_m[0]);
  const int kW = kernel_size_m.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size_m[1]);

  // NB: stride_m default is not expressible as an integer constant, so we accept
  // empty stride_m for this case
  TORCH_CHECK(stride_m.size() == 0 || stride_m.size() == 1 || stride_m.size() == 2,
    "max_pool2d: stride_m must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride_m.empty() ? kH : safe_downcast<int, int64_t>(stride_m[0]);
  const int dW = stride_m.empty() ? kW :
                 stride_m.size() == 1 ? dH : safe_downcast<int, int64_t>(stride_m[1]);

  TORCH_CHECK(padding_m.size() == 1 || padding_m.size() == 2,
    "max_pool2d: padding_m must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding_m[0]);
  const int padW = padding_m.size() == 1 ? padH : safe_downcast<int, int64_t>(padding_m[1]);

  TORCH_CHECK(dilation_m.size() == 1 || dilation_m.size() == 2,
    "max_pool2d: dilation_m must be either a single int, or a tuple of two ints");
  const int dilation_mH = safe_downcast<int, int64_t>(dilation_m[0]);
  const int dilation_mW = dilation_m.size() == 1 ? dilation_mH : safe_downcast<int, int64_t>(dilation_m[1]);

  const auto memory_format = input_m_.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input_m_.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input_m with channels_last layout");
  } else {
    TORCH_CHECK((input_m_.ndimension() == 3 || input_m_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input_m");
  }
  const Tensor input_m = input_m_.contiguous(memory_format);

  const int64_t nbatch = input_m.ndimension() == 4 ? input_m.size(-4) : 1;
  const int64_t nInputPlane = input_m.size(-3);
  const int64_t input_mHeight = input_m.size(-2);
  const int64_t input_mWidth = input_m.size(-1);

  const int64_t in_stride_m_n = input_m.ndimension() == 4 ? input_m.stride(-4) : 0;
  const int64_t in_stride_m_c = input_m.stride(-3);
  const int64_t in_stride_m_h = input_m.stride(-2);
  const int64_t in_stride_m_w = input_m.stride(-1);

  const int64_t output_mHeight = pooling_output_shape<int64_t>(input_mHeight, kH, padH, dH, dilation_mH, ceil_mode);
  const int64_t output_mWidth = pooling_output_shape<int64_t>(input_mWidth, kW, padW, dW, dilation_mW, ceil_mode);

  max_pool2d_backward_shape_check(
    input_m_,
    gradOutput_,
    indices,
    nbatch,
    kH, kW, dH, dW, padH, padW, dilation_mH, dilation_mW,
    nInputPlane,
    input_mHeight, input_mWidth,
    output_mHeight, output_mWidth,
    /*cuda=*/ true);

  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t out_stride_m_c = gradOutput.stride(-3);
  const int64_t out_stride_m_h = gradOutput.stride(-2);
  const int64_t out_stride_m_w = gradOutput.stride(-1);

  gradInput.resize_as_(input_m);
  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  int64_t count = input_m.numel();

  //================//
  Tensor output = at::empty_like(input_);
  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU("col2im_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  col2im_shape_check(
      input_,
      Tensor(),
      output_height,
      output_width,
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
  if (input.dim() == 2) {
    // Force batch
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});
  output.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "col2im_out_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;

    Tensor input_n;
    Tensor output_n;

    int64_t height_col = (output_height + 2 * pad_height -
                          (dilation_height * (kernel_height - 1) + 1)) /
            stride_height +
        1;
    int64_t width_col = (output_width + 2 * pad_width -
                         (dilation_width * (kernel_width - 1) + 1)) /
            stride_width +
        1;

    int64_t elt = 0;
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);
    int64_t num_kernels = n_output_plane * output_height * output_width;

    scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
    scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
    int64_t *indices_data = indices.data_ptr<int64_t>();
    int imgcount = input_mWidth * input_mHeight;
    dim3 grid;
    const int blocks = (imgcount + BLOCK_THREADS - 1) / BLOCK_THREADS;
    grid.x = blocks;
    grid.y = nbatch;
    uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    if (maxGridY < grid.y) grid.y = maxGridY;
    grid.z = nInputPlane;
    uint64_t maxGridZ = at::cuda::getCurrentDeviceProperties()->maxGridSize[2];
    if (maxGridZ < grid.z) grid.z = maxGridZ;

    col2im_kernel<scalar_t, accscalar_t>
    <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>
    (
      num_kernels,
      input_n.data<scalar_t>(),
      output_height, 
      output_width,
      n_output_plane,
      kernel_height,
      kernel_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      height_col,
      width_col,
      output_n.data<scalar_t>()
    );

    max_pool_backward_nchw<scalar_t, accscalar_t>
    <<<512, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        count,
            gradOutput_data,
            indices_data,
            nbatch,
            nInputPlane, input_mHeight, input_mWidth, output_mHeight, output_mWidth,
            kH, kW, dH, dW, padH, padW, dilation_mH, dilation_mW,
            gradInput_data);
    
    #define CALL(i, type, thread) col2im_kernel_max_pool_backward_nchw_fused_kernel_##type##_idx_##i<scalar_t, accscalar_t, scalar_t, accscalar_t>\
    <<<512, thread, 0, at::cuda::getCurrentCUDAStream()>>>(\
      num_kernels,\
      input_n.data<scalar_t>(),\
      output_height,\
      output_width,\
      n_output_plane,\
      kernel_height,\
      kernel_width,\
      pad_height,\
      pad_width,\
      stride_height,\
      stride_width,\
      dilation_height,\
      dilation_width,\
      height_col,\
      width_col,\
      output_n.data<scalar_t>(),\
      count,\
      gradOutput_data,\
      indices_data,\
      nbatch,\
      nInputPlane, input_mHeight, input_mWidth, output_mHeight, output_mWidth,\
      kH, kW, dH, dW, padH, padW, dilation_mH, dilation_mW,\
      gradInput_data);\
      cudaDeviceSynchronize()
    CALL(0, vfuse, 512);
    CALL(0, vfuse_lb, 512);
    CALL(0, hfuse, 768);
    CALL(0, hfuse_lb, 768);
    CALL(1, hfuse, 768);
    CALL(1, hfuse_lb, 768);
    CALL(2, hfuse, 768);
    CALL(2, hfuse_lb, 768);
    CALL(3, hfuse, 768);
    CALL(3, hfuse_lb, 768);
    CALL(4, hfuse, 768);
    CALL(4, hfuse_lb, 768);
    if (!batched_input) {
      output.resize_({n_output_plane, output_height, output_width});
    }
  });
  return output;
}

} // namespace native
} // namespace at
