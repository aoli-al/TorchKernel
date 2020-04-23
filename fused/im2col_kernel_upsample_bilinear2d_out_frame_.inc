template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_vfuse_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_3(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_4(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_5(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_6(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_3(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_4(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_5(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
template <typename dt0, typename scalar_t27, typename accscalar_t28>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_6(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns29, const accscalar_t28 rheight30, const accscalar_t28 rwidth31, const bool align_corners32, const PackedTensorAccessor<scalar_t27, 4> idata33, PackedTensorAccessor<scalar_t27, 4> odata34)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    int index35 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    const int batchsize36 = idata33.size(0);
    const int channels37 = idata33.size(1);
    const int height138 = idata33.size(2);
    const int width139 = idata33.size(3);
    const int height240 = odata34.size(2);
    const int width241 = odata34.size(3);
    if (index35 < ns29) {
        const int w242 = index35 % width241;
        const int h243 = index35 / width241;
        if (height138 == height240 && width139 == width241) {
            const int h154 = h243;
            const int w155 = w242;
            for (int n = 0; n < batchsize36; n++) {
                for (int c = 0; c < channels37; ++c) {
                    const scalar_t27 val56 = idata33[n][c][h154][w155];
                    odata34[n][c][h243][w242] = val56;
                }
            }
            return;
        }
        const accscalar_t28 h1r44 = area_pixel_compute_source_index<accscalar_t28>(rheight30, h243, align_corners32, false);
        const int h145 = h1r44;
        const int h1p46 = (h145 < height138 - 1) ? 1 : 0;
        const accscalar_t28 h1lambda47 = h1r44 - h145;
        const accscalar_t28 h0lambda48 = static_cast<accscalar_t28>(1) - h1lambda47;
        const accscalar_t28 w1r49 = area_pixel_compute_source_index<accscalar_t28>(rwidth31, w242, align_corners32, false);
        const int w150 = w1r49;
        const int w1p51 = (w150 < width139 - 1) ? 1 : 0;
        const accscalar_t28 w1lambda52 = w1r49 - w150;
        const accscalar_t28 w0lambda53 = static_cast<accscalar_t28>(1) - w1lambda52;
        for (int n = 0; n < batchsize36; n++) {
            for (int c = 0; c < channels37; ++c) {
                const accscalar_t28 val57 = h0lambda48 * (w0lambda53 * idata33[n][c][h145][w150] + w1lambda52 * idata33[n][c][h145][w150 + w1p51]) + h1lambda47 * (w0lambda53 * idata33[n][c][h145 + h1p46][w150] + w1lambda52 * idata33[n][c][h145 + h1p46][w150 + w1p51]);
                odata34[n][c][h243][w242] = static_cast<scalar_t27>(val57);
            }
        }
    }
}
}
