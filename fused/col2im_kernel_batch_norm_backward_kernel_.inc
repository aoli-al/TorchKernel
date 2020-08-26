template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(512, 2) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_vfuse_lb_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=0 && threadIdx.x < 512)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 0) % 256;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 0) / 256 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 0) / 512;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(512, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_vfuse_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=0 && threadIdx.x < 512)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 0) % 256;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 0) / 256 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 0) / 512;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=128 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 448;
    unsigned int threadIdx_x_1 = (threadIdx.x - 128) % 448;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 128) / 448 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 128) / 896;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,896;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,896;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_1(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=256 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = (threadIdx.x - 256) % 384;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 256) / 384 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 256) / 768;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,768;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,768;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_2(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=384 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 320;
    unsigned int threadIdx_x_1 = (threadIdx.x - 384) % 320;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 384) / 320 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 384) / 640;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,640;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,640;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_3(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=512 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 512) % 256;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 512) / 256 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 512) / 512;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_4(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=640 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 192;
    unsigned int threadIdx_x_1 = (threadIdx.x - 640) % 192;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 640) / 192 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 640) / 384;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,384;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,384;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_5(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
 {
if ((threadIdx.x>=0 && threadIdx.x < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 768;
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
if ((threadIdx.x>=768 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = (threadIdx.x - 768) % 128;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 768) / 128 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 768) / 256;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,256;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,256;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 0) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_idx_6(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
 {
if ((threadIdx.x>=0 && threadIdx.x < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 896;
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
if ((threadIdx.x>=896 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 64;
    unsigned int threadIdx_x_1 = (threadIdx.x - 896) % 64;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 896) / 64 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 896) / 128;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,128;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,128;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 2) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_0(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=128 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 448;
    unsigned int threadIdx_x_1 = (threadIdx.x - 128) % 448;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 128) / 448 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 128) / 896;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,896;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,896;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 2) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_1(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=256 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = (threadIdx.x - 256) % 384;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 256) / 384 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 256) / 768;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,768;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,768;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 2) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_2(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=384 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 320;
    unsigned int threadIdx_x_1 = (threadIdx.x - 384) % 320;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 384) / 320 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 384) / 640;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,640;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,640;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 2) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_3(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=512 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = (threadIdx.x - 512) % 256;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 512) / 256 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 512) / 512;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,512;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,512;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,512;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 1) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_4(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
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
if ((threadIdx.x>=640 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 192;
    unsigned int threadIdx_x_1 = (threadIdx.x - 640) % 192;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 640) / 192 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 640) / 384;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,384;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,384;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 1) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_5(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
 {
if ((threadIdx.x>=0 && threadIdx.x < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 768;
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
if ((threadIdx.x>=768 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = (threadIdx.x - 768) % 128;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 768) / 128 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 768) / 256;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,256;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,256;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
template <typename dt0, typename accT1, typename scalar_t32, typename accscalar_t33, typename index_t34>
 __global__ __launch_bounds__(1024, 1) void col2im_kernel_batch_norm_backward_kernel_fused_kernel_hfuse_lb_idx_6(const int64_t n2, const dt0 *data_col3, const int64_t height4, const int64_t width5, const int64_t channels6, const int64_t kernel_h7, const int64_t kernel_w8, const int64_t pad_height9, const int64_t pad_width10, const int64_t stride_height11, const int64_t stride_width12, const int64_t dilation_height13, const int64_t dilation_width14, const int64_t height_col15, const int64_t width_col16, dt0 *data_im17, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> input35, const PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_output36, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> grad_input37, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_weight38, PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> grad_bias39, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> weight40, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_mean41, const PackedTensorAccessor<scalar_t32, 1, DefaultPtrTraits, index_t34> running_var42, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_mean43, const PackedTensorAccessor<accscalar_t33, 1, DefaultPtrTraits, index_t34> save_invstd44, bool train45, accscalar_t33 epsilon46)
 {
if ((threadIdx.x>=0 && threadIdx.x < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = (threadIdx.x - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = (threadIdx.x - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = (threadIdx.x - 0) / 896;
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
if ((threadIdx.x>=896 && threadIdx.x < 1024)){
    unsigned int blockDim_x_1 = 64;
    unsigned int threadIdx_x_1 = (threadIdx.x - 896) % 64;
    unsigned int blockDim_y_1 = 2;
    unsigned int threadIdx_y_1 = (threadIdx.x - 896) / 64 % 2;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = (threadIdx.x - 896) / 128;
    index_t34 plane47 = blockIdx.x;
    index_t34 N48 = grad_output36.size(0) * grad_output36.size(2);
    accscalar_t33 mean49, invstd50;
    if (train45) {
        mean49 = save_mean43[plane47];
        invstd50 = save_invstd44[plane47];
    } else {
        mean49 = static_cast<accscalar_t33>(running_mean41[plane47]);
        invstd50 = static_cast<accscalar_t33>(1) / device_sqrt(static_cast<accscalar_t33>(running_var42[plane47]) + epsilon46);
    }
    accscalar_t33 weight_val51 = weight40.size(0) > 0 ? static_cast<accscalar_t33>(weight40[plane47]) : accscalar_t33(1);
    accscalar_t33 norm52 = accscalar_t33(1) / N48;
    GradOp<scalar_t32, accscalar_t33, PackedTensorAccessor<scalar_t32, 3, DefaultPtrTraits, index_t34> > g53(mean49, input35, grad_output36);
    struct Float2<scalar_t32, accscalar_t33> sum54 = static_cast<struct Float2<scalar_t32, accscalar_t33>>(0);
    for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
        for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
            sum54 += g53(batch, plane47, x);
        }
    }
    sum54 = warpSum(sum54);
    static struct Float2<scalar_t32, accscalar_t33> shared55[32] __attribute__((shared));
    asm ("bar.sync 1,128;");
    ;
    int tid56 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
    if (tid56 % WARP_SIZE == 0) {
        shared55[tid56 / WARP_SIZE] = sum54;
    }
    if (tid56 >= blockDim_x_1 * blockDim_y_1 / WARP_SIZE && tid56 < WARP_SIZE) {
        shared55[tid56] = (struct Float2<scalar_t32, accscalar_t33>)0;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid56 / WARP_SIZE == 0) {
        sum54 = warpSum(shared55[tid56]);
        if (tid56 == 0) {
            shared55[0] = sum54;
        }
    }
    asm ("bar.sync 1,128;");
    ;
    at::native::Float2<scalar_t32, accscalar_t33> res57 = shared55[0];
    accscalar_t33 grad_output_sum58 = res57.v1;
    accscalar_t33 dot_p59 = res57.v2;
    accscalar_t33 grad_mean60 = grad_output_sum58 * norm52;
    accscalar_t33 proj_scale61 = dot_p59 * norm52 * invstd50 * invstd50;
    accscalar_t33 grad_scale62 = invstd50 * weight_val51;
    if (grad_input37.data() != __null) {
        for (int batch = threadIdx_y_1; batch < grad_output36.size(0); batch += blockDim_y_1) {
            for (int x = threadIdx_x_1; x < grad_output36.size(2); x += blockDim_x_1) {
                scalar_t32 go63 = grad_output36[batch][plane47][x];
                if (train45) {
                    scalar_t32 inp64 = input35[batch][plane47][x];
                    accscalar_t33 proj65 = (inp64 - mean49) * proj_scale61;
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>((go63 - proj65 - grad_mean60) * grad_scale62);
                } else {
                    grad_input37[batch][plane47][x] = static_cast<scalar_t32>(go63 * grad_scale62);
                }
            }
        }
    }
    if (grad_weight38.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_weight38[plane47] = static_cast<scalar_t32>(dot_p59 * invstd50);
        }
    }
    if (grad_bias39.size(0) > 0) {
        if (threadIdx_x_1 == 0) {
            grad_bias39[plane47] = static_cast<scalar_t32>(grad_output_sum58);
        }
    }
}
}
