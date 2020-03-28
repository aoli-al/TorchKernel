template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
    int index37;
    index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    int batchsize38;
    batchsize38 = idata35.size(0);
    int channels39;
    channels39 = idata35.size(1);
    int height140;
    height140 = idata35.size(2);
    int width141;
    width141 = idata35.size(3);
    int height242;
    height242 = odata36.size(2);
    int width243;
    width243 = odata36.size(3);
    if (index37 < n31) {
        int w244;
        w244 = index37 % width243;
        int h245;
        h245 = index37 / width243;
        if (height140 == height242 && width141 == width243) {
            int h156;
            h156 = h245;
            int w157;
            w157 = w244;
            for (int n = 0; n < batchsize38; n++) {
                for (int c = 0; c < channels39; ++c) {
                    scalar_t29 val58;
                    val58 = idata35[n][c][h156][w157];
                    odata36[n][c][h245][w244] = val58;
                }
            }
            return;
        }
        accscalar_t30 h1r46;
        h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
        int h147;
        h147 = h1r46;
        int h1p48;
        h1p48 = (h147 < height140 - 1) ? 1 : 0;
        accscalar_t30 h1lambda49;
        h1lambda49 = h1r46 - h147;
        accscalar_t30 h0lambda50;
        h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
        accscalar_t30 w1r51;
        w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
        int w152;
        w152 = w1r51;
        int w1p53;
        w1p53 = (w152 < width141 - 1) ? 1 : 0;
        accscalar_t30 w1lambda54;
        w1lambda54 = w1r51 - w152;
        accscalar_t30 w0lambda55;
        w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
        for (int n = 0; n < batchsize38 - 1; n++) {
            for (int c = 0; c < channels39; ++c) {
                accscalar_t30 val59;
                val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
                odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
            }
        }
        for (int n = batchsize38 - 1; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                accscalar_t30 val60;
                val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
                odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
    int index37;
    index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    int batchsize38;
    batchsize38 = idata35.size(0);
    int channels39;
    channels39 = idata35.size(1);
    int height140;
    height140 = idata35.size(2);
    int width141;
    width141 = idata35.size(3);
    int height242;
    height242 = odata36.size(2);
    int width243;
    width243 = odata36.size(3);
    if (index37 < n31) {
        int w244;
        w244 = index37 % width243;
        int h245;
        h245 = index37 / width243;
        if (height140 == height242 && width141 == width243) {
            int h156;
            h156 = h245;
            int w157;
            w157 = w244;
            for (int n = 0; n < batchsize38; n++) {
                for (int c = 0; c < channels39; ++c) {
                    scalar_t29 val58;
                    val58 = idata35[n][c][h156][w157];
                    odata36[n][c][h245][w244] = val58;
                }
            }
            return;
        }
        accscalar_t30 h1r46;
        h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
        int h147;
        h147 = h1r46;
        int h1p48;
        h1p48 = (h147 < height140 - 1) ? 1 : 0;
        accscalar_t30 h1lambda49;
        h1lambda49 = h1r46 - h147;
        accscalar_t30 h0lambda50;
        h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
        accscalar_t30 w1r51;
        w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
        int w152;
        w152 = w1r51;
        int w1p53;
        w1p53 = (w152 < width141 - 1) ? 1 : 0;
        accscalar_t30 w1lambda54;
        w1lambda54 = w1r51 - w152;
        accscalar_t30 w0lambda55;
        w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
        for (int n = 0; n < batchsize38 - 1; n++) {
            for (int c = 0; c < channels39; ++c) {
                accscalar_t30 val59;
                val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
                odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
            }
        }
        for (int n = batchsize38 - 1; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                accscalar_t30 val60;
                val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
                odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_bar_sync_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_1(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_2(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
        }
    }
}
label_10:;
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_1(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_2(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
        }
    }
}
label_14:;
}
template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_bar_sync_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 1; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
    for (int n = batchsize38 - 1; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val60;
            val60 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val60);
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
