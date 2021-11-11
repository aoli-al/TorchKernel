#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(
    const int ns,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
    const int batchsize = idata.size(0);
    const int channels = idata.size(1);
    const int height1 = idata.size(2);
    const int width1 = idata.size(3);
    const int height2 = odata.size(2);
    const int width2 = odata.size(3);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (ns); index += blockDim.x * gridDim.x) {
        const int w2 = index % width2;
        const int h2 = index / width2;
        if (height1 == height2 && width1 == width2) {
            const int h1 = h2;
            const int w1 = w2;
            for (int n = 0; n < batchsize; n++) {
                for (int c = 0; c < channels; ++c) {
                    const scalar_t val = idata[n][c][h1][w1];
                    odata[n][c][h2][w2] = val;
                }
            }
            return;
        }
        const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(rheight, h2, align_corners, false);
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const accscalar_t h1lambda = h1r - h1;
        const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
        const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(rwidth, w2, align_corners, false);
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const accscalar_t w1lambda = w1r - w1;
        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
        for (int n = 0; n < batchsize; n++) {
            for (int c = 0; c < channels; ++c) {
                const accscalar_t val = h0lambda * (w0lambda * idata[n][c][h1][w1] + w1lambda * idata[n][c][h1][w1 + w1p]) + h1lambda * (w0lambda * idata[n][c][h1 + h1p][w1] + w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
                odata[n][c][h2][w2] = static_cast<scalar_t>(val);
            }
        }
    }
}


