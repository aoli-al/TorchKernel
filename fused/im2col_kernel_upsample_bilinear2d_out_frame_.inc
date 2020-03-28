template <typename dt0, typename scalar_t24, typename accscalar_t25>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns26, const accscalar_t25 rheight27, const accscalar_t25 rwidth28, const bool align_corners29, const PackedTensorAccessor<scalar_t24, 4> idata30, PackedTensorAccessor<scalar_t24, 4> odata31)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 512;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16;
        w_out16 = index % width_col14;
        index /= width_col14;
        int64_t h_out17;
        h_out17 = index % height_col13;
        int64_t channel_in18;
        channel_in18 = index / height_col13;
        int64_t channel_out19;
        channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
        int64_t h_in20;
        h_in20 = h_out17 * stride_height9 - pad_height7;
        int64_t w_in21;
        w_in21 = w_out16 * stride_width10 - pad_width8;
        data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
        data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h22;
                h22 = h_in20 + i * dilation_height11;
                int64_t w23;
                w23 = w_in21 + j * dilation_width12;
                * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                data_col15 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 512;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index32;
    index32 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    int batchsize33;
    batchsize33 = idata30.size(0);
    int channels34;
    channels34 = idata30.size(1);
    int height135;
    height135 = idata30.size(2);
    int width136;
    width136 = idata30.size(3);
    int height237;
    height237 = odata31.size(2);
    int width238;
    width238 = odata31.size(3);
    if (index32 < ns26) {
        int w239;
        w239 = index32 % width238;
        int h240;
        h240 = index32 / width238;
        if (height135 == height237 && width136 == width238) {
            int h151;
            h151 = h240;
            int w152;
            w152 = w239;
            for (int n = 0; n < batchsize33; n++) {
                for (int c = 0; c < channels34; ++c) {
                    scalar_t24 val53;
                    val53 = idata30[n][c][h151][w152];
                    odata31[n][c][h240][w239] = val53;
                }
            }
            return;
        }
        accscalar_t25 h1r41;
        h1r41 = area_pixel_compute_source_index<accscalar_t25>(rheight27, h240, align_corners29, false);
        int h142;
        h142 = h1r41;
        int h1p43;
        h1p43 = (h142 < height135 - 1) ? 1 : 0;
        accscalar_t25 h1lambda44;
        h1lambda44 = h1r41 - h142;
        accscalar_t25 h0lambda45;
        h0lambda45 = static_cast<accscalar_t25>(1) - h1lambda44;
        accscalar_t25 w1r46;
        w1r46 = area_pixel_compute_source_index<accscalar_t25>(rwidth28, w239, align_corners29, false);
        int w147;
        w147 = w1r46;
        int w1p48;
        w1p48 = (w147 < width136 - 1) ? 1 : 0;
        accscalar_t25 w1lambda49;
        w1lambda49 = w1r46 - w147;
        accscalar_t25 w0lambda50;
        w0lambda50 = static_cast<accscalar_t25>(1) - w1lambda49;
        for (int n = 0; n < batchsize33; n++) {
            for (int c = 0; c < channels34; ++c) {
                accscalar_t25 val54;
                val54 = h0lambda45 * (w0lambda50 * idata30[n][c][h142][w147] + w1lambda49 * idata30[n][c][h142][w147 + w1p48]) + h1lambda44 * (w0lambda50 * idata30[n][c][h142 + h1p43][w147] + w1lambda49 * idata30[n][c][h142 + h1p43][w147 + w1p48]);
                odata31[n][c][h240][w239] = static_cast<scalar_t24>(val54);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t24, typename accscalar_t25>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_vfuse_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns26, const accscalar_t25 rheight27, const accscalar_t25 rwidth28, const bool align_corners29, const PackedTensorAccessor<scalar_t24, 4> idata30, PackedTensorAccessor<scalar_t24, 4> odata31)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 512;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16;
        w_out16 = index % width_col14;
        index /= width_col14;
        int64_t h_out17;
        h_out17 = index % height_col13;
        int64_t channel_in18;
        channel_in18 = index / height_col13;
        int64_t channel_out19;
        channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
        int64_t h_in20;
        h_in20 = h_out17 * stride_height9 - pad_height7;
        int64_t w_in21;
        w_in21 = w_out16 * stride_width10 - pad_width8;
        data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
        data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h22;
                h22 = h_in20 + i * dilation_height11;
                int64_t w23;
                w23 = w_in21 + j * dilation_width12;
                * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                data_col15 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 512;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index32;
    index32 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    int batchsize33;
    batchsize33 = idata30.size(0);
    int channels34;
    channels34 = idata30.size(1);
    int height135;
    height135 = idata30.size(2);
    int width136;
    width136 = idata30.size(3);
    int height237;
    height237 = odata31.size(2);
    int width238;
    width238 = odata31.size(3);
    if (index32 < ns26) {
        int w239;
        w239 = index32 % width238;
        int h240;
        h240 = index32 / width238;
        if (height135 == height237 && width136 == width238) {
            int h151;
            h151 = h240;
            int w152;
            w152 = w239;
            for (int n = 0; n < batchsize33; n++) {
                for (int c = 0; c < channels34; ++c) {
                    scalar_t24 val53;
                    val53 = idata30[n][c][h151][w152];
                    odata31[n][c][h240][w239] = val53;
                }
            }
            return;
        }
        accscalar_t25 h1r41;
        h1r41 = area_pixel_compute_source_index<accscalar_t25>(rheight27, h240, align_corners29, false);
        int h142;
        h142 = h1r41;
        int h1p43;
        h1p43 = (h142 < height135 - 1) ? 1 : 0;
        accscalar_t25 h1lambda44;
        h1lambda44 = h1r41 - h142;
        accscalar_t25 h0lambda45;
        h0lambda45 = static_cast<accscalar_t25>(1) - h1lambda44;
        accscalar_t25 w1r46;
        w1r46 = area_pixel_compute_source_index<accscalar_t25>(rwidth28, w239, align_corners29, false);
        int w147;
        w147 = w1r46;
        int w1p48;
        w1p48 = (w147 < width136 - 1) ? 1 : 0;
        accscalar_t25 w1lambda49;
        w1lambda49 = w1r46 - w147;
        accscalar_t25 w0lambda50;
        w0lambda50 = static_cast<accscalar_t25>(1) - w1lambda49;
        for (int n = 0; n < batchsize33; n++) {
            for (int c = 0; c < channels34; ++c) {
                accscalar_t25 val54;
                val54 = h0lambda45 * (w0lambda50 * idata30[n][c][h142][w147] + w1lambda49 * idata30[n][c][h142][w147 + w1p48]) + h1lambda44 * (w0lambda50 * idata30[n][c][h142 + h1p43][w147] + w1lambda49 * idata30[n][c][h142 + h1p43][w147 + w1p48]);
                odata31[n][c][h240][w239] = static_cast<scalar_t24>(val54);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t24, typename accscalar_t25>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns26, const accscalar_t25 rheight27, const accscalar_t25 rwidth28, const bool align_corners29, const PackedTensorAccessor<scalar_t24, 4> idata30, PackedTensorAccessor<scalar_t24, 4> odata31)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_4;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_4:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_5;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index32;
index32 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize33;
batchsize33 = idata30.size(0);
int channels34;
channels34 = idata30.size(1);
int height135;
height135 = idata30.size(2);
int width136;
width136 = idata30.size(3);
int height237;
height237 = odata31.size(2);
int width238;
width238 = odata31.size(3);
if (index32 < ns26) {
    int w239;
    w239 = index32 % width238;
    int h240;
    h240 = index32 / width238;
    if (height135 == height237 && width136 == width238) {
        int h151;
        h151 = h240;
        int w152;
        w152 = w239;
        for (int n = 0; n < batchsize33; n++) {
            for (int c = 0; c < channels34; ++c) {
                scalar_t24 val53;
                val53 = idata30[n][c][h151][w152];
                odata31[n][c][h240][w239] = val53;
            }
        }
        return;
    }
    accscalar_t25 h1r41;
    h1r41 = area_pixel_compute_source_index<accscalar_t25>(rheight27, h240, align_corners29, false);
    int h142;
    h142 = h1r41;
    int h1p43;
    h1p43 = (h142 < height135 - 1) ? 1 : 0;
    accscalar_t25 h1lambda44;
    h1lambda44 = h1r41 - h142;
    accscalar_t25 h0lambda45;
    h0lambda45 = static_cast<accscalar_t25>(1) - h1lambda44;
    accscalar_t25 w1r46;
    w1r46 = area_pixel_compute_source_index<accscalar_t25>(rwidth28, w239, align_corners29, false);
    int w147;
    w147 = w1r46;
    int w1p48;
    w1p48 = (w147 < width136 - 1) ? 1 : 0;
    accscalar_t25 w1lambda49;
    w1lambda49 = w1r46 - w147;
    accscalar_t25 w0lambda50;
    w0lambda50 = static_cast<accscalar_t25>(1) - w1lambda49;
    for (int n = 0; n < batchsize33; n++) {
        for (int c = 0; c < channels34; ++c) {
            accscalar_t25 val54;
            val54 = h0lambda45 * (w0lambda50 * idata30[n][c][h142][w147] + w1lambda49 * idata30[n][c][h142][w147 + w1p48]) + h1lambda44 * (w0lambda50 * idata30[n][c][h142 + h1p43][w147] + w1lambda49 * idata30[n][c][h142 + h1p43][w147 + w1p48]);
            odata31[n][c][h240][w239] = static_cast<scalar_t24>(val54);
        }
    }
}
label_5:;
}
template <typename dt0, typename scalar_t24, typename accscalar_t25>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns26, const accscalar_t25 rheight27, const accscalar_t25 rwidth28, const bool align_corners29, const PackedTensorAccessor<scalar_t24, 4> idata30, PackedTensorAccessor<scalar_t24, 4> odata31)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_6;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_6:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_7;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index32;
index32 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize33;
batchsize33 = idata30.size(0);
int channels34;
channels34 = idata30.size(1);
int height135;
height135 = idata30.size(2);
int width136;
width136 = idata30.size(3);
int height237;
height237 = odata31.size(2);
int width238;
width238 = odata31.size(3);
if (index32 < ns26) {
    int w239;
    w239 = index32 % width238;
    int h240;
    h240 = index32 / width238;
    if (height135 == height237 && width136 == width238) {
        int h151;
        h151 = h240;
        int w152;
        w152 = w239;
        for (int n = 0; n < batchsize33; n++) {
            for (int c = 0; c < channels34; ++c) {
                scalar_t24 val53;
                val53 = idata30[n][c][h151][w152];
                odata31[n][c][h240][w239] = val53;
            }
        }
        return;
    }
    accscalar_t25 h1r41;
    h1r41 = area_pixel_compute_source_index<accscalar_t25>(rheight27, h240, align_corners29, false);
    int h142;
    h142 = h1r41;
    int h1p43;
    h1p43 = (h142 < height135 - 1) ? 1 : 0;
    accscalar_t25 h1lambda44;
    h1lambda44 = h1r41 - h142;
    accscalar_t25 h0lambda45;
    h0lambda45 = static_cast<accscalar_t25>(1) - h1lambda44;
    accscalar_t25 w1r46;
    w1r46 = area_pixel_compute_source_index<accscalar_t25>(rwidth28, w239, align_corners29, false);
    int w147;
    w147 = w1r46;
    int w1p48;
    w1p48 = (w147 < width136 - 1) ? 1 : 0;
    accscalar_t25 w1lambda49;
    w1lambda49 = w1r46 - w147;
    accscalar_t25 w0lambda50;
    w0lambda50 = static_cast<accscalar_t25>(1) - w1lambda49;
    for (int n = 0; n < batchsize33; n++) {
        for (int c = 0; c < channels34; ++c) {
            accscalar_t25 val54;
            val54 = h0lambda45 * (w0lambda50 * idata30[n][c][h142][w147] + w1lambda49 * idata30[n][c][h142][w147 + w1p48]) + h1lambda44 * (w0lambda50 * idata30[n][c][h142 + h1p43][w147] + w1lambda49 * idata30[n][c][h142 + h1p43][w147 + w1p48]);
            odata31[n][c][h240][w239] = static_cast<scalar_t24>(val54);
        }
    }
}
label_7:;
}
