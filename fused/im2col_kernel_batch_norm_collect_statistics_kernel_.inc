template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out45;
        w_out45 = index % width_col43;
        int idx46;
        idx46 = index / width_col43;
        int64_t h_out47;
        h_out47 = idx46 % height_col42;
        int64_t channel_in48;
        channel_in48 = idx46 / height_col42;
        int64_t channel_out49;
        channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
        int64_t h_in50;
        h_in50 = h_out47 * stride_height38 - pad_height36;
        int64_t w_in51;
        w_in51 = w_out45 * stride_width39 - pad_width37;
        dt29 *block_col52;
        block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
        const dt29 *block_im53;
        block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h54;
                h54 = h_in50 + i * dilation_height40;
                int64_t w55;
                w55 = w_in51 + j * dilation_width41;
                * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col52 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 32;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 16;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13;
    plane13 = blockIdx.x;
    int N14;
    N14 = input5.size(0) * input5.size(2);
    int tid15;
    tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16;
    shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17;
    avg17 = 0;
    stat_accscalar_t3 var_n18;
    var_n18 = 0;
    int n19;
    n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20;
            v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121;
            d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22;
        o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23;
        o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24;
        factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    __syncthreads();
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    __syncthreads();
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25;
        o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26;
        o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27;
        factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28;
            unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out45;
        w_out45 = index % width_col43;
        int idx46;
        idx46 = index / width_col43;
        int64_t h_out47;
        h_out47 = idx46 % height_col42;
        int64_t channel_in48;
        channel_in48 = idx46 / height_col42;
        int64_t channel_out49;
        channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
        int64_t h_in50;
        h_in50 = h_out47 * stride_height38 - pad_height36;
        int64_t w_in51;
        w_in51 = w_out45 * stride_width39 - pad_width37;
        dt29 *block_col52;
        block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
        const dt29 *block_im53;
        block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
        for (int64_t i = 0; i < kernel_height34; ++i) {
            for (int64_t j = 0; j < kernel_width35; ++j) {
                int64_t h54;
                h54 = h_in50 + i * dilation_height40;
                int64_t w55;
                w55 = w_in51 + j * dilation_width41;
                * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
                block_col52 += height_col42 * width_col43;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 32;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 16;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13;
    plane13 = blockIdx.x;
    int N14;
    N14 = input5.size(0) * input5.size(2);
    int tid15;
    tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16;
    shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17;
    avg17 = 0;
    stat_accscalar_t3 var_n18;
    var_n18 = 0;
    int n19;
    n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20;
            v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121;
            d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22;
        o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23;
        o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24;
        factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    __syncthreads();
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    __syncthreads();
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25;
        o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26;
        o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27;
        factor27 = 1. / fmaxf(1., n19 + o_n26);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
        avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
        n19 += o_n26;
    }
    if (tid15 == 0) {
        if (save_mean10.data() != __null) {
            save_mean10[plane13] = avg17;
        }
        if (save_transformed_var11.data() != __null) {
            save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
        }
        if (running_mean8.data() != __null) {
            running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
        }
        if (running_var9.data() != __null) {
            stat_accscalar_t3 unbiasedVar28;
            unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_bar_sync_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_8;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_8:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_9;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
asm ("bar.sync 1,512;");
;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
asm ("bar.sync 1,512;");
;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_9:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_10;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_10:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_11;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_12;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_12:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_13:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_11;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_12;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_12:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_10;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_10:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_13:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_11;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_12;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_12:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_13:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_10;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_10:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_14;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_14:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_15;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_16;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_16:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_17:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_15;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_16;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_14;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_17:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_15;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_16;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_16:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_17:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_14;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_14:;
}
template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_bar_sync_idx_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_18;
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    int idx46;
    idx46 = index / width_col43;
    int64_t h_out47;
    h_out47 = idx46 % height_col42;
    int64_t channel_in48;
    channel_in48 = idx46 / height_col42;
    int64_t channel_out49;
    channel_out49 = channel_in48 * kernel_height34 * kernel_width35;
    int64_t h_in50;
    h_in50 = h_out47 * stride_height38 - pad_height36;
    int64_t w_in51;
    w_in51 = w_out45 * stride_width39 - pad_width37;
    dt29 *block_col52;
    block_col52 = data_col44 + (channel_out49 * height_col42 + h_out47) * width_col43 + w_out45;
    const dt29 *block_im53;
    block_im53 = data_im31 + (channel_in48 * height32 + h_in50) * width33 + w_in51;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h54;
            h54 = h_in50 + i * dilation_height40;
            int64_t w55;
            w55 = w_in51 + j * dilation_width41;
            * block_col52 = (h54 >= 0 && w55 >= 0 && h54 < height32 && w55 < width33) ? block_im53[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            block_col52 += height_col42 * width_col43;
        }
    }
}
label_18:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
asm ("bar.sync 1,512;");
;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
asm ("bar.sync 1,512;");
;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_19:;
}
