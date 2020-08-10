template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(512, 2) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_vfuse_lb_idx_0(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
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
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(512, 0) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_vfuse_idx_0(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
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
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 0) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_idx_0(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,640;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,640;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 0) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_idx_1(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 0) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_idx_2(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,384;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,384;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 0) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_idx_3(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,256;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,256;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 0) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_idx_4(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,128;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,128;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 2) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_0(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,640;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,640;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 2) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_1(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 2) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_2(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,384;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,384;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 2) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_3(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,256;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,256;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
template <typename scalar_t0, typename accscalar_t1, typename scalar_t29, typename accscalar_t30, typename index_t31>
 __global__ __launch_bounds__(768, 2) void max_pool_backward_nchw_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_4(const int nthreads2, const scalar_t0 *top_diff3, const int64_t *top_mask4, const int num5, const int channels6, const int height7, const int width8, const int pooled_height9, const int pooled_width10, const int kernel_h11, const int kernel_w12, const int stride_h13, const int stride_w14, const int pad_h15, const int pad_w16, const int dilation_h17, const int dilation_w18, scalar_t0 *bottom_diff19, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> input32, const PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_output33, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> grad_input34, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_weight35, PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> grad_bias36, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> weight37, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_mean38, const PackedTensorAccessor<scalar_t29, 1, DefaultPtrTraits, index_t31> running_var39, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_mean40, const PackedTensorAccessor<accscalar_t30, 1, DefaultPtrTraits, index_t31> save_invstd41, bool train42, accscalar_t30 epsilon43)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    int64_t _i_n_d_e_x20 = blockIdx.x * blockDim_x_0 + threadIdx_x_0;
    for (int index = _i_n_d_e_x20; _i_n_d_e_x20 < (height7 * width8); _i_n_d_e_x20 += blockDim_x_0 * gridDim.x , index = _i_n_d_e_x20) {
        int h21 = index / width8;
        int w22 = index - h21 * width8;
        int phstart23 = p_start(h21, pad_h15, kernel_h11, dilation_h17, stride_h13);
        int phend24 = p_end(h21, pad_h15, pooled_height9, stride_h13);
        int pwstart25 = p_start(w22, pad_w16, kernel_w12, dilation_w18, stride_w14);
        int pwend26 = p_end(w22, pad_w16, pooled_width10, stride_w14);
        for (int n = blockIdx.y; n < num5; n += gridDim.y) {
            for (int c = blockIdx.z; c < channels6; c += gridDim.z) {
                accscalar_t1 gradient27 = accscalar_t1(0);
                int offset28 = (n * channels6 + c) * pooled_height9 * pooled_width10;
                for (int ph = phstart23; ph < phend24; ++ph) {
                    for (int pw = pwstart25; pw < pwend26; ++pw) {
                        if (top_mask4[ph * pooled_width10 + pw + offset28] == h21 * width8 + w22) {
                            gradient27 += ScalarConvert<scalar_t0, accscalar_t1>::to(top_diff3[ph * pooled_width10 + pw + offset28]);
                        }
                    }
                }
                bottom_diff19[(n * channels6 + c) * height7 * width8 + index] = ScalarConvert<accscalar_t1, scalar_t0>::to(gradient27);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
    index_t31 plane44 = blockIdx.x;
    index_t31 N45 = grad_output33.size(0) * grad_output33.size(2);
    accscalar_t30 mean46, invstd47;
    if (train42) {
        mean46 = save_mean40[plane44];
        invstd47 = save_invstd41[plane44];
    } else {
        mean46 = static_cast<accscalar_t30>(running_mean38[plane44]);
        invstd47 = static_cast<accscalar_t30>(1) / device_sqrt(static_cast<accscalar_t30>(running_var39[plane44]) + epsilon43);
    }
    accscalar_t30 weight_val48 = weight37.size(0) > 0 ? static_cast<accscalar_t30>(weight37[plane44]) : accscalar_t30(1);
    accscalar_t30 norm49 = accscalar_t30(1) / N45;
    GradOp<scalar_t29, accscalar_t30, PackedTensorAccessor<scalar_t29, 3, DefaultPtrTraits, index_t31> > g50(mean46, input32, grad_output33);
    struct Float2<scalar_t29, accscalar_t30> sum51 = static_cast<struct Float2<scalar_t29, accscalar_t30>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
            sum51 += g50(batch, plane44, x);
        }
    }
    sum51 = warpSum(sum51);
    static struct Float2<scalar_t29, accscalar_t30> shared52[32] __attribute__((shared));
    asm ("bar.sync 1,128;");
    ;
    int tid53 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid53 % WARP_SIZE == 0) {
        shared52[tid53 / WARP_SIZE] = sum51;
    }
    if (tid53 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid53 < WARP_SIZE) {
        shared52[tid53] = (struct Float2<scalar_t29, accscalar_t30>)0;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid53 / WARP_SIZE == 0) {
        sum51 = warpSum(shared52[tid53]);
        if (tid53 == 0) {
            shared52[0] = sum51;
        }
    }
    asm ("bar.sync 1,128;");
    ;
    at::native::Float2<scalar_t29, accscalar_t30> res54 = shared52[0];
    accscalar_t30 grad_output_sum55 = res54.v1;
    accscalar_t30 dot_p56 = res54.v2;
    accscalar_t30 grad_mean57 = grad_output_sum55 * norm49;
    accscalar_t30 proj_scale58 = dot_p56 * norm49 * invstd47 * invstd47;
    accscalar_t30 grad_scale59 = invstd47 * weight_val48;
    if (grad_input34.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output33.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output33.size(2); x += blockDim_x_1) {
                scalar_t29 go60 = grad_output33[batch][plane44][x];
                if (train42) {
                    scalar_t29 inp61 = input32[batch][plane44][x];
                    accscalar_t30 proj62 = (inp61 - mean46) * proj_scale58;
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>((go60 - proj62 - grad_mean57) * grad_scale59);
                } else {
                    grad_input34[batch][plane44][x] = static_cast<scalar_t29>(go60 * grad_scale59);
                }
            }
        }
    }
    if (grad_weight35.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight35[plane44] = static_cast<scalar_t29>(dot_p56 * invstd47);
        }
    }
    if (grad_bias36.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias36[plane44] = static_cast<scalar_t29>(grad_output_sum55);
        }
    }
}
}
