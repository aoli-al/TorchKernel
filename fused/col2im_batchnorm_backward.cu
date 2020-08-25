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
#include <ATen/AccumulateType.h>

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
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/native/im2col_shape_check.h>


namespace at {
namespace native {

using namespace at::cuda::detail;

#include "col2im.inc1"
#include "batchnorm_backward.inc2"

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}



template<typename sscalar_t, typename b_index_t>
Tensor col2im_batch_norm_backward(
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride, 
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

  auto input_b = b_input_breshaped.packed_accessor<sscalar_t, 3, DefaultPtrTraits, b_index_t>();
  auto grad_output_b = grad_output_b_reshaped.packed_accessor<sscalar_t, 3, DefaultPtrTraits, b_index_t>();
  auto grad_input_b = packed_accessor_or_dummy<sscalar_t, 3, DefaultPtrTraits, b_index_t>(grad_b_input_breshaped);
  auto weight = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(weight_);
  auto grad_weight = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(grad_weight_);
  auto grad_bias = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(grad_bias_);
  auto running_mean = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(running_mean_);
  auto running_var = packed_accessor_or_dummy<sscalar_t, 1, DefaultPtrTraits, b_index_t>(running_var_);
  auto save_mean = packed_accessor_or_dummy<accsscalar_t, 1, DefaultPtrTraits, b_index_t>(save_mean_);
  auto save_invstd = packed_accessor_or_dummy<accsscalar_t, 1, DefaultPtrTraits, b_index_t>(save_invstd_);

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks_b(input_b.size(1));
  int tf = getNumThreads(input_b.size(2));
  dim3 threads_bs(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));




  // ----------------- //
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

    for (int64_t elt = 0; elt < batch_size; elt++) {
      input_n = input.select(0, elt);
      output_n = output.select(0, elt);
      int64_t num_kernels = n_output_plane * output_height * output_width;

      batch_norm_backward_kernel<sscalar_t,  accsscalar_t, b_index_t> <<<blocks_b, threads_bs, 0, stream>>>
        (input_b, grad_output_b, grad_input_b, grad_weight, grad_bias, weight, running_mean, running_var,
        save_mean, save_invstd, train, epsilon);
      THCudaCheck(cudaGetLastError());
      col2im_kernel<scalar_t, accscalar_t>
      <<<blocks_b, 512, 0, at::cuda::getCurrentCUDAStream()>>>
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
    }
    if (!batched_input) {
      output.resize_({n_output_plane, output_height, output_width});
    }
  });
  return output;
}

Tensor col2im_batch_norm_backward_out(
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride, 
    const Tensor& b_grad_out, const Tensor& b_input_b, const Tensor& weight_,
    const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
    bool train, double epsilon, std::array<bool,3> grad_b_input_bmask) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(b_input_b.scalar_type(), "batch_norm_backward_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(b_input_b)) {
        return col2im_batch_norm_backward<scalar_t, int32_t>(
            input_,
            output_size,
            kernel_size,
            dilation,
            padding,
            stride, 
            b_grad_out, b_input_b, weight_,
            running_mean_, running_var_, save_mean_, save_invstd_,
            train, epsilon, grad_b_input_bmask);
      } else {
        return col2im_batch_norm_backward<scalar_t, int64_t>(
            input_,
            output_size,
            kernel_size,
            dilation,
            padding,
            stride, 
            b_grad_out, b_input_b, weight_,
            running_mean_, running_var_, save_mean_, save_invstd_,
            train, epsilon, grad_b_input_bmask);
      }
    });
}


} // namespace native
} // namespace at
