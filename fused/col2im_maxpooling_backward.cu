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

namespace at {
namespace native {

using namespace at::cuda::detail;

#include "col2im.inc1"


template <typename dt, typename accT>
void col2im(
    cudaStream_t stream,
    const dt* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t patch_height,
    const int64_t patch_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    dt* data_im) {
  int64_t num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // CUDA_NUM_THREADS = 1024
  col2im_kernel<dt, accT>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
          num_kernels,
          data_col,
          height,
          width,
          channels,
          patch_height,
          patch_width,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          output_height,
          output_width,
          data_im);
  AT_CUDA_CHECK(cudaGetLastError());
}



void col2im_batch_norm_backward(
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
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
    if (!batched_input) {
      output.resize_({n_output_plane, output_height, output_width});
    }
  });
  return output;
}

template<typename sscalar_t, typename b_index_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& b_grad_out, const Tensor& b_input_b, const Tensor& weight_,
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

  batch_norm_backward_kernel<sscalar_t,  accsscalar_t, b_index_t> <<<blocks_b, threads_bs, 0, stream>>>
    (input_b, grad_output_b, grad_input_b, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_invstd, train, epsilon);
  THCudaCheck(cudaGetLastError());

  return std::make_tuple(grad_b_input_b, grad_weight_, grad_bias_);
}


} // namespace native
} // namespace at
