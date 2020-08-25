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

    int idx = index / width_col;

    int64_t h_out = idx % height_col;
    int64_t channel_in = idx / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    // data_col += (channel_out * height_col + h_out) * width_col + w_out;
    // data_im += (channel_in * height + h_in) * width + w_in;
    dt* block_col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const dt* block_im = data_im + (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *block_col = (h >= 0 && w >= 0 && h < height && w < width)
            ? block_im[i * dilation_height * width + j * dilation_width]
            : ScalarConvert<int, dt>::to(0);
        block_col += height_col * width_col;
      }
    }
  }
}
