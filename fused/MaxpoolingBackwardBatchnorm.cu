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
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>

namespace at {
namespace native {
namespace {

#include "maxpool_backward.inc"
#include "batchnorm_backward.inc"
#include "max_pool_backward_nchw_batch_norm_backward_kernel_.inc"

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}


template<typename sscalar_t, typename b_index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& b_grad_out, const Tensor& b_input, const Tensor& weight_,
                                                                     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
                                                                     bool train, double epsilon, std::array<bool,3> grad_b_inputmask) {


}


template<typename sscalar_t, typename b_index_t>
std::tuple<Tensor, Tensor, Tensor> 
max_pool2d_with_indices_backward_out_cuda_template(
           Tensor& gradInput,
           const Tensor& gradOutput_,
           const Tensor& input_,
           const Tensor& indices,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode,
            const Tensor& b_grad_out, const Tensor& b_input_b, const Tensor& weight_,
            const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
            bool train, double epsilon, std::array<bool,3> grad_b_input_bmask) {
  using accsscalar_t = at::acc_type<sscalar_t, true>;
  Tensor grad_b_input_b;
  Tensor grad_b_input_breshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto b_input_breshaped = b_input_b.reshape({b_input_b.size(0), b_input_b.size(1), -1});
  auto grad_output_b_reshaped = b_grad_out.reshape(b_input_breshaped.sizes());

  if (grad_b_input_bmask[0]) {
    grad_b_input_b = at::empty_like(b_input_b);
    grad_b_input_breshaped = grad_b_input_b.view(b_input_breshaped.sizes());
  }
  if (grad_b_input_bmask[1]) {
    grad_weight_ = at::empty_like(weight_);
  }
  if (grad_b_input_bmask[2]) {
    grad_bias_ = at::empty_like(weight_);
  }

  printf("1\n");
  auto input_b = b_input_breshaped.packed_accessor<sscalar_t, 3, DefaultPtrTraits, b_index_t>();
  printf("a\n");
  auto grad_output_b = grad_output_b_reshaped.packed_accessor<sscalar_t, 3, DefaultPtrTraits, b_index_t>();
  printf("b\n");
  auto grad_input_b = packed_accessor_or_dummy<sscalar_t, 3, DefaultPtrTraits, b_index_t>(grad_b_input_breshaped);
  printf("c\n");
  auto weight = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(weight_);
  printf("d\n");
  auto grad_weight = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(grad_weight_);
  printf("e\n");
  auto grad_bias = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(grad_bias_);
  printf("f\n");
  auto running_mean = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(running_mean_);
  printf("g\n");
  auto running_var = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(running_var_);
  printf("h\n");
  auto save_mean = packed_accessor_or_dummy<accsscalar_t, 1, DefaultPtrTraits, b_index_t>(save_mean_);
  printf("i\n");
  auto save_invstd = packed_accessor_or_dummy<accsscalar_t, 1, DefaultPtrTraits, b_index_t>(save_invstd_);
  printf("2\n");

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks_b(input_b.size(1));
  int tf = getNumThreads(input_b.size(2));
  dim3 threads_bs(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput_, "gradOutput_", 2 };
  TensorArg input_arg{ input_, "input_", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  printf("3\n");
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

  printf("4\n");
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

  printf("5\n");
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
        printf("function called\n");\
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

        printf("function called\n");\
        max_pool_backward_nchw<scalar_t, accscalar_t>
        <<<grid, BLOCK_THREADS, 0, at::cuda::getStreamFromPool(true)>>>(
            count,
                gradOutput_data,
                indices_data,
                nbatch,
                nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                gradInput_data);
        max_pool_backward_nchw<scalar_t, accscalar_t>
        <<<blocks_b, BLOCK_THREADS, 0, at::cuda::getStreamFromPool(true)>>>(
            count,
                gradOutput_data,
                indices_data,
                nbatch,
                nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
                kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                gradInput_data);
        batch_norm_backward_kernel<sscalar_t,  accsscalar_t, b_index_t> 
        <<<blocks_b, threads_bs, 0, at::cuda::getStreamFromPool(true)>>>
          (input_b, grad_output_b, grad_input_b, grad_weight, grad_bias, weight, running_mean, running_var,
          save_mean, save_invstd, train, epsilon);
          cudaDeviceSynchronize();
        printf("grid %d\n", grid.x);
        #define CALL(i, type, thread) max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_##type##_idx_##i<scalar_t, accscalar_t,sscalar_t,  accsscalar_t, b_index_t>\
          <<<blocks_b, thread, 0, stream>>>(\
            count,\
            gradOutput_data,\
            indices_data,\
            nbatch,\
            nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,\
            kH, kW, dH, dW, padH, padW, dilationH, dilationW,\
            gradInput_data,\
            input_b, grad_output_b, grad_input_b, grad_weight, grad_bias, weight, running_mean, running_var,\
            save_mean, save_invstd, train, epsilon);\
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
      });
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_b_input_b, grad_weight_, grad_bias_);
}


} // namespace

Tensor max_pool2d_with_indices_backward_cuda(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices,
  const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var,
  const Tensor& save_mean, const Tensor& save_invstd, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_backward_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        max_pool2d_with_indices_backward_out_cuda_template<scalar_t, int32_t>(
          gradInput,
          gradOutput_,
          input,
          indices,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          grad_out, self, weight, running_mean, running_var, 
          save_mean, save_invstd, train, epsilon, grad_input_mask);
        return gradInput;
        // return batch_norm_backward_cuda_template<scalar_t, int32_t>(

      } else {
        auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
        max_pool2d_with_indices_backward_out_cuda_template<scalar_t, int64_t>(
          gradInput,
          gradOutput_,
          input,
          indices,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          grad_out, self, weight, running_mean, running_var, 
          save_mean, save_invstd, train, epsilon, grad_input_mask);
        return gradInput;

      }
    });
}

} // at::native
} // at
