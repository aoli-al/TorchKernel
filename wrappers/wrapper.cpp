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

std::tuple<Tensor, Tensor> max_hist_norm(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
  Tensor& input_,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode
);
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
  );
std::tuple<Tensor, Tensor> _histc_upsample(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
  );
std::tuple<Tensor, Tensor> _histc_cuda2(
  const Tensor& input_im2col_,
  IntArrayRef kernel_size_im2col,
  IntArrayRef dilation_im2col,
  IntArrayRef pad_im2colding_im2col,
  IntArrayRef stride_im2col,
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max);
std::tuple<Tensor, Tensor> hist_norm(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
  Tensor& input_);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_stream(
    const Tensor& input,
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
    const Tensor& input_batch_norm) ;
std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_upsample(
    const Tensor &input_im2col_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor &input,
    IntArrayRef output_size,
    bool align_corners);
// std::tuple<Tensor, Tensor, Tensor> softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float, const Tensor& self, double epsilon);
// Tensor im2col_cuda(
//     const Tensor &input,
//     IntArrayRef kernel_size,
//     IntArrayRef dilation,
//     IntArrayRef padding,
//     IntArrayRef stride);
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
std::tuple<Tensor, Tensor> im2col_batchnorm_cuda(
    const Tensor &input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor &input_batch_norm);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> max_pool2d_batch_norm(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& input_batch_norm_,
    double epsilon);
std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_maxpool(
  const Tensor& input_im2col_,
  IntArrayRef kernel_size_im2col,
  IntArrayRef dilation_im2col,
  IntArrayRef pad_im2colding_im2col,
  IntArrayRef stride_im2col,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode);
std::tuple<Tensor, Tensor, Tensor, Tensor> upsample_batchnorm(
  const Tensor& input,
  IntArrayRef output_size,
  bool align_corners,
  const Tensor& input_bn_, double epsilon);

} // namespace native
} // namespace at

// const auto defaultOptions = torch::TensorOptions({at::kCUDA}).dtype(at::kFloat).requires_grad(true);
// static auto batch_norm_input = torch::randn({128, 10000, 576}, defaultOptions);
// static auto input_max_pool = torch::randn({10, 10, 2560, 1000}, defaultOptions);
// static auto hist_input = torch::randn({122400000}, defaultOptions);
// static auto im2col_input = torch::randn({1, 10, 2512, 2048}, defaultOptions);
// static auto input_upsample = torch::randn({16, 16, 256, 100}, defaultOptions);



std::tuple<Tensor, Tensor, Tensor> call_max_pool_upsample_fused(Tensor input_max_pool, Tensor input_upsample)
{
  return at::native::max_pool_upsample_stream(input_max_pool, {5, 5}, {10, 10}, 2, 1, false,
                                              input_upsample, {2000, 2560}, true);
}

std::tuple<Tensor, Tensor> im2col_batchnorm(Tensor im2col_input, Tensor batch_norm_input)
{
  return at::native::im2col_batchnorm_cuda(im2col_input, {13, 1}, {1, 1}, {0, 0}, {1, 1},
                                           batch_norm_input);
}

// std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> 
// im2col_maxpool_batchnorm(Tensor im2col_input, Tensor input_max_pool, Tensor batch_norm_input)
// {
//   return at::native::im2col_maxpool_batch_norm_stream(im2col_input, {13, 1}, {1, 1}, {0, 0}, {1, 1},
//                                           input_max_pool, {5, 5}, {10, 10}, 2, 1, false,
//                                            batch_norm_input);
// }

Tensor histc(Tensor im2col_input, Tensor hist_input)
{
  at::native::_histc_cuda2(im2col_input,{13, 1}, {1, 1}, {0, 0}, {1, 1},
   hist_input, 20, c10::Scalar(0.f), c10::Scalar(0.f));
  return torch::randn({100, 100});
}


std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> 
max_pool_batch_norm(Tensor input_max_pool, Tensor batch_norm_input) {
  return at::native::max_pool2d_batch_norm(input_max_pool, {3, 3}, {1, 1}, 0, 1, false,
    batch_norm_input, 0.1);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> 
im2col_upsample(Tensor im2col_input, Tensor input_upsample) {
  return at::native::im2col_upsample(im2col_input, {13, 1}, {1, 1}, {0, 0}, {1, 1},
                                     input_upsample, {2000, 2560}, true);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> 
im2col_maxpool(Tensor im2col_input, Tensor input_max_pool) {
  return at::native::im2col_maxpool(
    im2col_input, {13, 1}, {1, 1}, {0, 0}, {1, 1},
    input_max_pool, {5, 5}, {10, 10}, 2, 1, false
  );
}

std::tuple<Tensor, Tensor, Tensor, Tensor> 
upsample_batchnorm(Tensor input_upsample, Tensor batch_norm_input) {
  return at::native::upsample_batchnorm(input_upsample, {2000, 2560}, true,
                                        batch_norm_input, 0.2);
}
Tensor hist_norm(Tensor hist_input, Tensor batch_norm_input) {
  at::native::hist_norm(hist_input, 20, 0.f, 0.f, batch_norm_input);
 return torch::randn({100, 100});
}

Tensor histc_maxpool(Tensor hist_input, Tensor input_max_pool)
{
  at::native::_histc_maxpool(hist_input, 20, 0.f, 0.f,
    input_max_pool, {5, 5}, {10, 10}, 2, 1, false
  );
 return torch::randn({100, 100});
}

Tensor histc_upsample(Tensor hist_input, Tensor input_upsample)
{
  at::native::_histc_upsample(
    hist_input, 20, 0.f, 0.f,
                              input_upsample, {2000, 2560}, true);
 return torch::randn({100, 100});
}

// Tensor max_hist_norm( Tensor
// ) {
  // at::native::max_hist_norm(
    // hist_input, 20, 0.f, 0.f,
    // batch_norm_input,
    // input_max_pool, {5, 5}, {10, 10}, 2, 1, false
    // );
 // return torch::randn({100, 100});
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("call_max_pool_upsample_fused", &call_max_pool_upsample_fused, "LLTM forward (CUDA)");
  m.def("im2col_batchnorm", &im2col_batchnorm, "LLTM forward (CUDA)");
  m.def("max_pool_batch_norm", &max_pool_batch_norm, "LLTM forward (CUDA)");
  m.def("im2col_upsample", &im2col_upsample, "LLTM forward (CUDA)");
  m.def("im2col_maxpool", &im2col_maxpool, "LLTM forward (CUDA)");
  m.def("upsample_batchnorm", &upsample_batchnorm, "LLTM forward (CUDA)");
  m.def("histc", &histc, "LLTM forward (CUDA)");
  m.def("hist_norm", &hist_norm, "LLTM forward (CUDA)");
  m.def("histc_maxpool", &histc_maxpool, "LLTM forward (CUDA)");
  m.def("histc_upsample", &histc_upsample, "LLTM forward (CUDA)");
}
