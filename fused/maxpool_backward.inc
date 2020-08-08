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

