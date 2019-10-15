#include <torch/extension.h>

#include <cuda.h>
#include <vector>

using at::IntArrayRef;
using at::TensorList;
using torch::Tensor;

namespace at
{
namespace native
{
Tensor im2col_cuda(
    const Tensor &input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride);
std::tuple<Tensor, Tensor, Tensor> max_pool_upsample_stream(
    const Tensor &input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor &input_upsample,
    const IntArrayRef output_size,
    bool align_corners);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> im2col_batchnorm_cuda(
    const Tensor &input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor &input_batch_norm);
} // namespace native
} // namespace at

const auto defaultOptions = torch::TensorOptions({at::kCUDA}).dtype(at::kFloat).requires_grad(true);

std::tuple<Tensor, Tensor, Tensor> call_max_pool_upsample_fused()
{
  auto input_max_pool = torch::randn({4, 4, 3210, 5010}, defaultOptions);
  auto input_upsample = torch::randn({12, 12, 256, 100}, defaultOptions);
  return at::native::max_pool_upsample_stream(input_max_pool, {20, 20}, {10, 10}, 0, 1, false,
                                              input_upsample, {2000, 3840}, true);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> im2col(const Tensor &t, IntArrayRef kernel_size)
{

  auto im2col_input = torch::randn({1, 1, 2750, 2048}, defaultOptions);
  // auto r = at::native::im2col_cuda(im2col_input, {251, 1}, {1, 1}, {0, 0}, {1, 1});
  // return std::make_tuple(r, r, r, r, r, r);
  auto batch_norm_input = torch::randn({10000, 10000}, defaultOptions);
  return at::native::im2col_batchnorm_cuda(im2col_input, {251, 1}, {1, 1}, {0, 0}, {1, 1},
                                           batch_norm_input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("call_max_pool_upsample_fused", &call_max_pool_upsample_fused, "LLTM forward (CUDA)");
  m.def("im2col", &im2col, "LLTM forward (CUDA)");
}
