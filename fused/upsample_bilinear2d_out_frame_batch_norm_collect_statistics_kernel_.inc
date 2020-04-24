template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
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
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
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
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 56;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 56;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 56 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 48;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 48;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 48 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_3(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
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
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_4(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_5(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_6(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 56;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 56;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 56 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,896;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 48;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 48;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 48 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,768;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 40;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 40;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 40 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,640;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_3(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 32;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
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
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_4(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 24;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 24;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 24 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,384;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 1) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_5(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 16;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,256;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 1) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_6(const int ns31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    const int batchsize37 = idata35.size(0);
    const int channels38 = idata35.size(1);
    const int height139 = idata35.size(2);
    const int width140 = idata35.size(3);
    const int height241 = odata36.size(2);
    const int width242 = odata36.size(3);
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (ns31); index += blockDim_x_1 * gridDim.x) {
        const int w243 = index % width242;
        const int h244 = index / width242;
        if (height139 == height241 && width140 == width242) {
            const int h155 = h244;
            const int w156 = w243;
            for (int n = 0; n < batchsize37; n++) {
                for (int c = 0; c < channels38; ++c) {
                    const scalar_t29 val57 = idata35[n][c][h155][w156];
                    odata36[n][c][h244][w243] = val57;
                }
            }
            return;
        }
        const accscalar_t30 h1r45 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h244, align_corners34, false);
        const int h146 = h1r45;
        const int h1p47 = (h146 < height139 - 1) ? 1 : 0;
        const accscalar_t30 h1lambda48 = h1r45 - h146;
        const accscalar_t30 h0lambda49 = static_cast<accscalar_t30>(1) - h1lambda48;
        const accscalar_t30 w1r50 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w243, align_corners34, false);
        const int w151 = w1r50;
        const int w1p52 = (w151 < width140 - 1) ? 1 : 0;
        const accscalar_t30 w1lambda53 = w1r50 - w151;
        const accscalar_t30 w0lambda54 = static_cast<accscalar_t30>(1) - w1lambda53;
        for (int n = 0; n < batchsize37; n++) {
            for (int c = 0; c < channels38; ++c) {
                const accscalar_t30 val58 = h0lambda49 * (w0lambda54 * idata35[n][c][h146][w151] + w1lambda53 * idata35[n][c][h146][w151 + w1p52]) + h1lambda48 * (w0lambda54 * idata35[n][c][h146 + h1p47][w151] + w1lambda53 * idata35[n][c][h146 + h1p47][w151 + w1p52]);
                odata36[n][c][h244][w243] = static_cast<scalar_t29>(val58);
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 8;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 8;
    unsigned int blockDim_y_0 = 16;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 8 % 16;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    static int shared_n12[160] __attribute__((shared));
    int plane13 = blockIdx.x;
    int N14 = input5.size(0) * input5.size(2);
    int tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
    stat_accscalar_t3 *shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
    stat_accscalar_t3 avg17 = 0;
    stat_accscalar_t3 var_n18 = 0;
    int n19 = 0;
    for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
        for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
            stat_accscalar_t3 v20 = input5[batch][plane13][x];
            stat_accscalar_t3 d121 = v20 - avg17;
            n19++;
            avg17 += d121 / n19;
            var_n18 += d121 * (v20 - avg17);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor24 = 1. / fmaxf(1., n19 + o_n23);
        var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
        avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
        n19 += o_n23;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 % WARP_SIZE == 0) {
        shared_n12[tid15 / WARP_SIZE] = n19;
        shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
        shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
    }
    asm ("bar.sync 1,128;");
    ;
    if (tid15 < WARP_SIZE) {
        n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
        avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
        var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t3 o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
        int o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
        stat_accscalar_t3 factor27 = 1. / fmaxf(1., n19 + o_n26);
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
            stat_accscalar_t3 unbiasedVar28 = var_n18 / (N14 - 1);
            running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
        }
    }
}
}
