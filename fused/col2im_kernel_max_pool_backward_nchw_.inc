template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(512, 2) void col2im_kernel_max_pool_backward_nchw_fused_kernel_vfuse_lb_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 512;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=0 && threadIdx.x < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 0) / 256;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(512, 0) void col2im_kernel_max_pool_backward_nchw_fused_kernel_vfuse_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 512;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=0 && threadIdx.x < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 0) / 256;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 0) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 128;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=128 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = (threadIdx.x - 128) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 128) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 128) / 640;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 0) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_idx_1(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 256;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=256 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = (threadIdx.x - 256) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 256) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 256) / 512;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 0) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_idx_2(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 384;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=384 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = (threadIdx.x - 384) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 384) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 384) / 384;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 0) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_idx_3(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 512;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=512 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 512) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 512) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 512) / 256;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 0) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_idx_4(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 640;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=640 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = (threadIdx.x - 640) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 640) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 640) / 128;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 2) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_lb_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 128;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=128 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = (threadIdx.x - 128) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 128) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 128) / 640;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 2) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_lb_idx_1(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 256;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=256 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = (threadIdx.x - 256) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 256) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 256) / 512;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 2) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_lb_idx_2(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 384;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=384 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = (threadIdx.x - 384) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 384) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 384) / 384;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 2) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_lb_idx_3(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 512;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=512 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 512) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 512) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 512) / 256;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33>
 __global__ __launch_bounds__(768, 1) void col2im_kernel_max_pool_backward_nchw_fused_kernel_hfuse_lb_idx_4(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const int nthreads34, const scalar_t32 *top_diff35, const int64_t *top_mask36, const int num37, const int channels38, const int height39, const int width40, const int pooled_height41, const int pooled_width42, const int kernel_h43, const int kernel_w44, const int stride_h45, const int stride_w46, const int pad_h47, const int pad_w48, const int dilation_h49, const int dilation_w50, scalar_t32 *bottom_diff51)
 {
if ((threadIdx.x>=0 && threadIdx.x < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 640;
    int64_t _i_n_d_e_x18 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x18; _i_n_d_e_x18 < (n2); _i_n_d_e_x18 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x18) {
        accT1 val19 = static_cast<accT1>(0);
        const int64_t w_im20 = index % width5 + pad_width10;
        const int64_t h_im21 = (index / width5) % height4 + pad_height9;
        const int64_t c_im22 = index / (width5 * height4);
        int64_t kernel_extent_w23 = (kernel_w8 - 1) * dilation_width14 + 1;
        int64_t kernel_extent_h24 = (kernel_h7 - 1) * dilation_height13 + 1;
        const int64_t w_col_start25 = (w_im20 < kernel_extent_w23) ? 0 : (w_im20 - kernel_extent_w23) / stride_width12 + 1;
        const int64_t w_col_end26 = ::min(w_im20 / stride_width12 + 1, width_col16);
        const int64_t h_col_start27 = (h_im21 < kernel_extent_h24) ? 0 : (h_im21 - kernel_extent_h24) / stride_height11 + 1;
        const int64_t h_col_end28 = ::min(h_im21 / stride_height11 + 1, height_col15);
        for (int64_t h_col = h_col_start27; h_col < h_col_end28; h_col += 1) {
            for (int64_t w_col = w_col_start25; w_col < w_col_end26; w_col += 1) {
                int64_t h_k29 = (h_im21 - h_col * stride_height11);
                int64_t w_k30 = (w_im20 - w_col * stride_width12);
                if (h_k29 % dilation_height13 == 0 && w_k30 % dilation_width14 == 0) {
                    h_k29 /= dilation_height13;
                    w_k30 /= dilation_width14;
                    int64_t data_col_index31 = (((c_im22 * kernel_h7 + h_k29) * kernel_w8 + w_k30) * height_col15 + h_col) * width_col16 + w_col;
                    val19 += data_col3[data_col_index31];
                }
            }
        }
        data_im17[index] = static_cast<dt0>(val19);
    }
}
if ((threadIdx.x>=640 && threadIdx.x < 768)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = (threadIdx.x - 640) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = (threadIdx.x - 640) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 640) / 128;
    int64_t _i_n_d_e_x52 = blockIdx.z * blockDim_x_1 + threadIdx_x_1;
    for (int index = _i_n_d_e_x52; _i_n_d_e_x52 < (height39 * width40); _i_n_d_e_x52 += blockDim_x_1 * gridDim.z , index = _i_n_d_e_x52) {
        int h53 = index / width40;
        int w54 = index - h53 * width40;
        int phstart55 = p_start(h53, pad_h47, kernel_h43, dilation_h49, stride_h45);
        int phend56 = p_end(h53, pad_h47, pooled_height41, stride_h45);
        int pwstart57 = p_start(w54, pad_w48, kernel_w44, dilation_w50, stride_w46);
        int pwend58 = p_end(w54, pad_w48, pooled_width42, stride_w46);
        for (int n = blockIdx.y; n < num37; n += gridDim.y) {
            for (int c = blockIdx.x; c < channels38; c += gridDim.x) {
                accscalar_t33 gradient59 = accscalar_t33(0);
                int offset60 = (n * channels38 + c) * pooled_height41 * pooled_width42;
                for (int ph = phstart55; ph < phend56; ++ph) {
                    for (int pw = pwstart57; pw < pwend58; ++pw) {
                        if (top_mask36[ph * pooled_width42 + pw + offset60] == h53 * width40 + w54) {
                            gradient59 += ScalarConvert<scalar_t32, accscalar_t33>::to(top_diff35[ph * pooled_width42 + pw + offset60]);
                        }
                    }
                }
                bottom_diff51[(n * channels38 + c) * height39 * width40 + index] = ScalarConvert<accscalar_t33, scalar_t32>::to(gradient59);
            }
        }
    }
}
}
