template <typename dt32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_MaxPoolForward_fused_kernel_vfuse_lb_idx_0(const int64_t n33, const dt32 *data_im34, const int64_t height35, const int64_t width36, const int64_t kernel_height37, const int64_t kernel_width38, const int64_t pad_height39, const int64_t pad_width40, const int64_t stride_height41, const int64_t stride_width42, const int64_t dilation_height43, const int64_t dilation_width44, const int64_t height_col45, const int64_t width_col46, dt32 *data_col47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n33); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out48;
        w_out48 = index % width_col46;
        int idx49;
        idx49 = index / width_col46;
        int64_t h_out50;
        h_out50 = idx49 % height_col45;
        int64_t channel_in51;
        channel_in51 = idx49 / height_col45;
        int64_t channel_out52;
        channel_out52 = channel_in51 * kernel_height37 * kernel_width38;
        int64_t h_in53;
        h_in53 = h_out50 * stride_height41 - pad_height39;
        int64_t w_in54;
        w_in54 = w_out48 * stride_width42 - pad_width40;
        dt32 *block_col55;
        block_col55 = data_col47 + (channel_out52 * height_col45 + h_out50) * width_col46 + w_out48;
        const dt32 *block_im56;
        block_im56 = data_im34 + (channel_in51 * height35 + h_in53) * width36 + w_in54;
        for (int64_t i = 0; i < kernel_height37; ++i) {
            for (int64_t j = 0; j < kernel_width38; ++j) {
                int64_t h57;
                h57 = h_in53 + i * dilation_height43;
                int64_t w58;
                w58 = w_in54 + j * dilation_width44;
                * block_col55 = (h57 >= 0 && w58 >= 0 && h57 < height35 && w58 < width36) ? block_im56[i * dilation_height43 * width36 + j * dilation_width44] : ScalarConvert<int, dt32>::to(0);
                block_col55 += height_col45 * width_col46;
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
        const scalar_t0 *btm_data30;
        btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31;
                val31 = btm_data30[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val31) > maxval28) || THCNumerics<scalar_t0>::isnan(val31)) {
                    maxidx29 = h * width7 + w;
                    maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val31);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
        top_mask19[index] = maxidx29;
    }
}
}
template <typename dt32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_MaxPoolForward_fused_kernel_vfuse_idx_0(const int64_t n33, const dt32 *data_im34, const int64_t height35, const int64_t width36, const int64_t kernel_height37, const int64_t kernel_width38, const int64_t pad_height39, const int64_t pad_width40, const int64_t stride_height41, const int64_t stride_width42, const int64_t dilation_height43, const int64_t dilation_width44, const int64_t height_col45, const int64_t width_col46, dt32 *data_col47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n33); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out48;
        w_out48 = index % width_col46;
        int idx49;
        idx49 = index / width_col46;
        int64_t h_out50;
        h_out50 = idx49 % height_col45;
        int64_t channel_in51;
        channel_in51 = idx49 / height_col45;
        int64_t channel_out52;
        channel_out52 = channel_in51 * kernel_height37 * kernel_width38;
        int64_t h_in53;
        h_in53 = h_out50 * stride_height41 - pad_height39;
        int64_t w_in54;
        w_in54 = w_out48 * stride_width42 - pad_width40;
        dt32 *block_col55;
        block_col55 = data_col47 + (channel_out52 * height_col45 + h_out50) * width_col46 + w_out48;
        const dt32 *block_im56;
        block_im56 = data_im34 + (channel_in51 * height35 + h_in53) * width36 + w_in54;
        for (int64_t i = 0; i < kernel_height37; ++i) {
            for (int64_t j = 0; j < kernel_width38; ++j) {
                int64_t h57;
                h57 = h_in53 + i * dilation_height43;
                int64_t w58;
                w58 = w_in54 + j * dilation_width44;
                * block_col55 = (h57 >= 0 && w58 >= 0 && h57 < height35 && w58 < width36) ? block_im56[i * dilation_height43 * width36 + j * dilation_width44] : ScalarConvert<int, dt32>::to(0);
                block_col55 += height_col45 * width_col46;
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
        const scalar_t0 *btm_data30;
        btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31;
                val31 = btm_data30[h * width7 + w];
                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val31) > maxval28) || THCNumerics<scalar_t0>::isnan(val31)) {
                    maxidx29 = h * width7 + w;
                    maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val31);
                }
            }
        }
        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
        top_mask19[index] = maxidx29;
    }
}
}
template <typename dt32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_idx_0(const int64_t n33, const dt32 *data_im34, const int64_t height35, const int64_t width36, const int64_t kernel_height37, const int64_t kernel_width38, const int64_t pad_height39, const int64_t pad_width40, const int64_t stride_height41, const int64_t stride_width42, const int64_t dilation_height43, const int64_t dilation_width44, const int64_t height_col45, const int64_t width_col46, dt32 *data_col47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n33); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out48;
    w_out48 = index % width_col46;
    int idx49;
    idx49 = index / width_col46;
    int64_t h_out50;
    h_out50 = idx49 % height_col45;
    int64_t channel_in51;
    channel_in51 = idx49 / height_col45;
    int64_t channel_out52;
    channel_out52 = channel_in51 * kernel_height37 * kernel_width38;
    int64_t h_in53;
    h_in53 = h_out50 * stride_height41 - pad_height39;
    int64_t w_in54;
    w_in54 = w_out48 * stride_width42 - pad_width40;
    dt32 *block_col55;
    block_col55 = data_col47 + (channel_out52 * height_col45 + h_out50) * width_col46 + w_out48;
    const dt32 *block_im56;
    block_im56 = data_im34 + (channel_in51 * height35 + h_in53) * width36 + w_in54;
    for (int64_t i = 0; i < kernel_height37; ++i) {
        for (int64_t j = 0; j < kernel_width38; ++j) {
            int64_t h57;
            h57 = h_in53 + i * dilation_height43;
            int64_t w58;
            w58 = w_in54 + j * dilation_width44;
            * block_col55 = (h57 >= 0 && w58 >= 0 && h57 < height35 && w58 < width36) ? block_im56[i * dilation_height43 * width36 + j * dilation_width44] : ScalarConvert<int, dt32>::to(0);
            block_col55 += height_col45 * width_col46;
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
    const scalar_t0 *btm_data30;
    btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val31;
            val31 = btm_data30[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val31) > maxval28) || THCNumerics<scalar_t0>::isnan(val31)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val31);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_5:;
}
template <typename dt32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_idx_0(const int64_t n33, const dt32 *data_im34, const int64_t height35, const int64_t width36, const int64_t kernel_height37, const int64_t kernel_width38, const int64_t pad_height39, const int64_t pad_width40, const int64_t stride_height41, const int64_t stride_width42, const int64_t dilation_height43, const int64_t dilation_width44, const int64_t height_col45, const int64_t width_col46, dt32 *data_col47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n33); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out48;
    w_out48 = index % width_col46;
    int idx49;
    idx49 = index / width_col46;
    int64_t h_out50;
    h_out50 = idx49 % height_col45;
    int64_t channel_in51;
    channel_in51 = idx49 / height_col45;
    int64_t channel_out52;
    channel_out52 = channel_in51 * kernel_height37 * kernel_width38;
    int64_t h_in53;
    h_in53 = h_out50 * stride_height41 - pad_height39;
    int64_t w_in54;
    w_in54 = w_out48 * stride_width42 - pad_width40;
    dt32 *block_col55;
    block_col55 = data_col47 + (channel_out52 * height_col45 + h_out50) * width_col46 + w_out48;
    const dt32 *block_im56;
    block_im56 = data_im34 + (channel_in51 * height35 + h_in53) * width36 + w_in54;
    for (int64_t i = 0; i < kernel_height37; ++i) {
        for (int64_t j = 0; j < kernel_width38; ++j) {
            int64_t h57;
            h57 = h_in53 + i * dilation_height43;
            int64_t w58;
            w58 = w_in54 + j * dilation_width44;
            * block_col55 = (h57 >= 0 && w58 >= 0 && h57 < height35 && w58 < width36) ? block_im56[i * dilation_height43 * width36 + j * dilation_width44] : ScalarConvert<int, dt32>::to(0);
            block_col55 += height_col45 * width_col46;
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
    const scalar_t0 *btm_data30;
    btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val31;
            val31 = btm_data30[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val31) > maxval28) || THCNumerics<scalar_t0>::isnan(val31)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val31);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_7:;
}
template <typename dt32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_imba_idx_0(const int64_t n33, const dt32 *data_im34, const int64_t height35, const int64_t width36, const int64_t kernel_height37, const int64_t kernel_width38, const int64_t pad_height39, const int64_t pad_width40, const int64_t stride_height41, const int64_t stride_width42, const int64_t dilation_height43, const int64_t dilation_width44, const int64_t height_col45, const int64_t width_col46, dt32 *data_col47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_8;
unsigned int blockDim_x_1;
blockDim_x_1 = 160;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 160;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n33); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out48;
    w_out48 = index % width_col46;
    int idx49;
    idx49 = index / width_col46;
    int64_t h_out50;
    h_out50 = idx49 % height_col45;
    int64_t channel_in51;
    channel_in51 = idx49 / height_col45;
    int64_t channel_out52;
    channel_out52 = channel_in51 * kernel_height37 * kernel_width38;
    int64_t h_in53;
    h_in53 = h_out50 * stride_height41 - pad_height39;
    int64_t w_in54;
    w_in54 = w_out48 * stride_width42 - pad_width40;
    dt32 *block_col55;
    block_col55 = data_col47 + (channel_out52 * height_col45 + h_out50) * width_col46 + w_out48;
    const dt32 *block_im56;
    block_im56 = data_im34 + (channel_in51 * height35 + h_in53) * width36 + w_in54;
    for (int64_t i = 0; i < kernel_height37; ++i) {
        for (int64_t j = 0; j < kernel_width38; ++j) {
            int64_t h57;
            h57 = h_in53 + i * dilation_height43;
            int64_t w58;
            w58 = w_in54 + j * dilation_width44;
            * block_col55 = (h57 >= 0 && w58 >= 0 && h57 < height35 && w58 < width36) ? block_im56[i * dilation_height43 * width36 + j * dilation_width44] : ScalarConvert<int, dt32>::to(0);
            block_col55 += height_col45 * width_col46;
        }
    }
}
label_8:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_9;
unsigned int blockDim_x_0;
blockDim_x_0 = 608;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) % 608;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 608 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 608;
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
    const scalar_t0 *btm_data30;
    btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val31;
            val31 = btm_data30[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val31) > maxval28) || THCNumerics<scalar_t0>::isnan(val31)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val31);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_9:;
}
template <typename dt32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void im2col_kernel_MaxPoolForward_fused_kernel_hfuse_lb_imba_idx_0(const int64_t n33, const dt32 *data_im34, const int64_t height35, const int64_t width36, const int64_t kernel_height37, const int64_t kernel_width38, const int64_t pad_height39, const int64_t pad_width40, const int64_t stride_height41, const int64_t stride_width42, const int64_t dilation_height43, const int64_t dilation_width44, const int64_t height_col45, const int64_t width_col46, dt32 *data_col47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_10;
unsigned int blockDim_x_1;
blockDim_x_1 = 160;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 160;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 160;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n33); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out48;
    w_out48 = index % width_col46;
    int idx49;
    idx49 = index / width_col46;
    int64_t h_out50;
    h_out50 = idx49 % height_col45;
    int64_t channel_in51;
    channel_in51 = idx49 / height_col45;
    int64_t channel_out52;
    channel_out52 = channel_in51 * kernel_height37 * kernel_width38;
    int64_t h_in53;
    h_in53 = h_out50 * stride_height41 - pad_height39;
    int64_t w_in54;
    w_in54 = w_out48 * stride_width42 - pad_width40;
    dt32 *block_col55;
    block_col55 = data_col47 + (channel_out52 * height_col45 + h_out50) * width_col46 + w_out48;
    const dt32 *block_im56;
    block_im56 = data_im34 + (channel_in51 * height35 + h_in53) * width36 + w_in54;
    for (int64_t i = 0; i < kernel_height37; ++i) {
        for (int64_t j = 0; j < kernel_width38; ++j) {
            int64_t h57;
            h57 = h_in53 + i * dilation_height43;
            int64_t w58;
            w58 = w_in54 + j * dilation_width44;
            * block_col55 = (h57 >= 0 && w58 >= 0 && h57 < height35 && w58 < width36) ? block_im56[i * dilation_height43 * width36 + j * dilation_width44] : ScalarConvert<int, dt32>::to(0);
            block_col55 += height_col45 * width_col46;
        }
    }
}
label_10:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_11;
unsigned int blockDim_x_0;
blockDim_x_0 = 608;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) % 608;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 608 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 160) / 608;
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
    const scalar_t0 *btm_data30;
    btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val31;
            val31 = btm_data30[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val31) > maxval28) || THCNumerics<scalar_t0>::isnan(val31)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val31);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_11:;
}
