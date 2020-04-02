template <typename dt31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_MaxPoolForward_fused_kernel_vfuse_lb_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out47;
        w_out47 = index % width_col45;
        index /= width_col45;
        int64_t h_out48;
        h_out48 = index % height_col44;
        int64_t channel_in49;
        channel_in49 = index / height_col44;
        int64_t channel_out50;
        channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
        int64_t h_in51;
        h_in51 = h_out48 * stride_height40 - pad_height38;
        int64_t w_in52;
        w_in52 = w_out47 * stride_width41 - pad_width39;
        data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
        data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
        for (int64_t i = 0; i < kernel_height36; ++i) {
            for (int64_t j = 0; j < kernel_width37; ++j) {
                int64_t h53;
                h53 = h_in51 + i * dilation_height42;
                int64_t w54;
                w54 = w_in52 + j * dilation_width43;
                * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
                data_col46 += height_col44 * width_col45;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 256;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20;
        pw20 = index % pooled_width9;
        int ph21;
        ph21 = (index / pooled_width9) % pooled_height8;
        int c22;
        c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23;
        n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24;
        hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25;
        wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26;
        hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27;
        wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28;
        maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29;
        maxidx29 = hstart24 * width7 + wstart25;
        bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val30;
                val30 = bottom_data3[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                    maxidx29 = h * width7 + w;
                    maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
        top_mask19[index] = maxidx29;
    }
}
}
template <typename dt31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_MaxPoolForward_fused_kernel_vfuse_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out47;
        w_out47 = index % width_col45;
        index /= width_col45;
        int64_t h_out48;
        h_out48 = index % height_col44;
        int64_t channel_in49;
        channel_in49 = index / height_col44;
        int64_t channel_out50;
        channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
        int64_t h_in51;
        h_in51 = h_out48 * stride_height40 - pad_height38;
        int64_t w_in52;
        w_in52 = w_out47 * stride_width41 - pad_width39;
        data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
        data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
        for (int64_t i = 0; i < kernel_height36; ++i) {
            for (int64_t j = 0; j < kernel_width37; ++j) {
                int64_t h53;
                h53 = h_in51 + i * dilation_height42;
                int64_t w54;
                w54 = w_in52 + j * dilation_width43;
                * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
                data_col46 += height_col44 * width_col45;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 256;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20;
        pw20 = index % pooled_width9;
        int ph21;
        ph21 = (index / pooled_width9) % pooled_height8;
        int c22;
        c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23;
        n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24;
        hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25;
        wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26;
        hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27;
        wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28;
        maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29;
        maxidx29 = hstart24 * width7 + wstart25;
        bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val30;
                val30 = bottom_data3[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                    maxidx29 = h * width7 + w;
                    maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
        top_mask19[index] = maxidx29;
    }
}
}
template <typename dt31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_4;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out47;
    w_out47 = index % width_col45;
    index /= width_col45;
    int64_t h_out48;
    h_out48 = index % height_col44;
    int64_t channel_in49;
    channel_in49 = index / height_col44;
    int64_t channel_out50;
    channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
    int64_t h_in51;
    h_in51 = h_out48 * stride_height40 - pad_height38;
    int64_t w_in52;
    w_in52 = w_out47 * stride_width41 - pad_width39;
    data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
    data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
    for (int64_t i = 0; i < kernel_height36; ++i) {
        for (int64_t j = 0; j < kernel_width37; ++j) {
            int64_t h53;
            h53 = h_in51 + i * dilation_height42;
            int64_t w54;
            w54 = w_in52 + j * dilation_width43;
            * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
            data_col46 += height_col44 * width_col45;
        }
    }
}
label_4:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_5;
unsigned int blockDim_x_0;
blockDim_x_0 = 256;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
    int pw20;
    pw20 = index % pooled_width9;
    int ph21;
    ph21 = (index / pooled_width9) % pooled_height8;
    int c22;
    c22 = (index / pooled_width9 / pooled_height8) % channels5;
    int n23;
    n23 = index / pooled_width9 / pooled_height8 / channels5;
    int hstart24;
    hstart24 = ph21 * stride_h12 - pad_h14;
    int wstart25;
    wstart25 = pw20 * stride_w13 - pad_w15;
    int hend26;
    hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
    int wend27;
    wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
    while (hstart24 < 0)
        hstart24 += dilation_h16;
    while (wstart25 < 0)
        wstart25 += dilation_w17;
    accscalar_t1 maxval28;
    maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
    int maxidx29;
    maxidx29 = hstart24 * width7 + wstart25;
    bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val30;
            val30 = bottom_data3[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_5:;
}
template <typename dt31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_6;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out47;
    w_out47 = index % width_col45;
    index /= width_col45;
    int64_t h_out48;
    h_out48 = index % height_col44;
    int64_t channel_in49;
    channel_in49 = index / height_col44;
    int64_t channel_out50;
    channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
    int64_t h_in51;
    h_in51 = h_out48 * stride_height40 - pad_height38;
    int64_t w_in52;
    w_in52 = w_out47 * stride_width41 - pad_width39;
    data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
    data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
    for (int64_t i = 0; i < kernel_height36; ++i) {
        for (int64_t j = 0; j < kernel_width37; ++j) {
            int64_t h53;
            h53 = h_in51 + i * dilation_height42;
            int64_t w54;
            w54 = w_in52 + j * dilation_width43;
            * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
            data_col46 += height_col44 * width_col45;
        }
    }
}
label_6:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_7;
unsigned int blockDim_x_0;
blockDim_x_0 = 256;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
    int pw20;
    pw20 = index % pooled_width9;
    int ph21;
    ph21 = (index / pooled_width9) % pooled_height8;
    int c22;
    c22 = (index / pooled_width9 / pooled_height8) % channels5;
    int n23;
    n23 = index / pooled_width9 / pooled_height8 / channels5;
    int hstart24;
    hstart24 = ph21 * stride_h12 - pad_h14;
    int wstart25;
    wstart25 = pw20 * stride_w13 - pad_w15;
    int hend26;
    hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
    int wend27;
    wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
    while (hstart24 < 0)
        hstart24 += dilation_h16;
    while (wstart25 < 0)
        wstart25 += dilation_w17;
    accscalar_t1 maxval28;
    maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
    int maxidx29;
    maxidx29 = hstart24 * width7 + wstart25;
    bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val30;
            val30 = bottom_data3[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_7:;
}
template <typename dt31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_imba_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 288)) goto label_8;
unsigned int blockDim_x_1;
blockDim_x_1 = 288;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 288;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 288 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 288;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out47;
    w_out47 = index % width_col45;
    index /= width_col45;
    int64_t h_out48;
    h_out48 = index % height_col44;
    int64_t channel_in49;
    channel_in49 = index / height_col44;
    int64_t channel_out50;
    channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
    int64_t h_in51;
    h_in51 = h_out48 * stride_height40 - pad_height38;
    int64_t w_in52;
    w_in52 = w_out47 * stride_width41 - pad_width39;
    data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
    data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
    for (int64_t i = 0; i < kernel_height36; ++i) {
        for (int64_t j = 0; j < kernel_width37; ++j) {
            int64_t h53;
            h53 = h_in51 + i * dilation_height42;
            int64_t w54;
            w54 = w_in52 + j * dilation_width43;
            * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
            data_col46 += height_col44 * width_col45;
        }
    }
}
label_8:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=288 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_9;
unsigned int blockDim_x_0;
blockDim_x_0 = 480;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 288) % 480;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 288) / 480 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 288) / 480;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
    int pw20;
    pw20 = index % pooled_width9;
    int ph21;
    ph21 = (index / pooled_width9) % pooled_height8;
    int c22;
    c22 = (index / pooled_width9 / pooled_height8) % channels5;
    int n23;
    n23 = index / pooled_width9 / pooled_height8 / channels5;
    int hstart24;
    hstart24 = ph21 * stride_h12 - pad_h14;
    int wstart25;
    wstart25 = pw20 * stride_w13 - pad_w15;
    int hend26;
    hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
    int wend27;
    wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
    while (hstart24 < 0)
        hstart24 += dilation_h16;
    while (wstart25 < 0)
        wstart25 += dilation_w17;
    accscalar_t1 maxval28;
    maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
    int maxidx29;
    maxidx29 = hstart24 * width7 + wstart25;
    bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val30;
            val30 = bottom_data3[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_9:;
}
template <typename dt31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_imba_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 288)) goto label_10;
unsigned int blockDim_x_1;
blockDim_x_1 = 288;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 288;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 288 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 288;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out47;
    w_out47 = index % width_col45;
    index /= width_col45;
    int64_t h_out48;
    h_out48 = index % height_col44;
    int64_t channel_in49;
    channel_in49 = index / height_col44;
    int64_t channel_out50;
    channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
    int64_t h_in51;
    h_in51 = h_out48 * stride_height40 - pad_height38;
    int64_t w_in52;
    w_in52 = w_out47 * stride_width41 - pad_width39;
    data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
    data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
    for (int64_t i = 0; i < kernel_height36; ++i) {
        for (int64_t j = 0; j < kernel_width37; ++j) {
            int64_t h53;
            h53 = h_in51 + i * dilation_height42;
            int64_t w54;
            w54 = w_in52 + j * dilation_width43;
            * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
            data_col46 += height_col44 * width_col45;
        }
    }
}
label_10:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=288 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_11;
unsigned int blockDim_x_0;
blockDim_x_0 = 480;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 288) % 480;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 288) / 480 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 288) / 480;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
    int pw20;
    pw20 = index % pooled_width9;
    int ph21;
    ph21 = (index / pooled_width9) % pooled_height8;
    int c22;
    c22 = (index / pooled_width9 / pooled_height8) % channels5;
    int n23;
    n23 = index / pooled_width9 / pooled_height8 / channels5;
    int hstart24;
    hstart24 = ph21 * stride_h12 - pad_h14;
    int wstart25;
    wstart25 = pw20 * stride_w13 - pad_w15;
    int hend26;
    hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
    int wend27;
    wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
    while (hstart24 < 0)
        hstart24 += dilation_h16;
    while (wstart25 < 0)
        wstart25 += dilation_w17;
    accscalar_t1 maxval28;
    maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
    int maxidx29;
    maxidx29 = hstart24 * width7 + wstart25;
    bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val30;
            val30 = bottom_data3[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_11:;
}
