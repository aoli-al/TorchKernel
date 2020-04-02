template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
    extern unsigned char my_smem45[] __attribute__((shared));
    output_t29 *smem46;
    smem46 = nullptr;
    smem46 = reinterpret_cast<output_t29 *>(my_smem45);
    for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
        smem46[i] = 0;
    }
    __syncthreads();
    for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType31 bOffset47;
        bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
        input_t30 bVal48;
        bVal48 = b39.data[bOffset47];
        if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
            IndexType31 bin49;
            bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
            atomicAdd(&smem46[bin49], getOp44(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
        IndexType31 aOffset50;
        aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
        atomicAdd(&a37.data[aOffset50], smem46[i]);
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_vfuse_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
    extern unsigned char my_smem45[] __attribute__((shared));
    output_t29 *smem46;
    smem46 = nullptr;
    smem46 = reinterpret_cast<output_t29 *>(my_smem45);
    for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
        smem46[i] = 0;
    }
    __syncthreads();
    for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType31 bOffset47;
        bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
        input_t30 bVal48;
        bVal48 = b39.data[bOffset47];
        if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
            IndexType31 bin49;
            bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
            atomicAdd(&smem46[bin49], getOp44(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
        IndexType31 aOffset50;
        aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
        atomicAdd(&a37.data[aOffset50], smem46[i]);
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_bar_sync_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_12;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
asm ("bar.sync 2,512;");
;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
asm ("bar.sync 2,512;");
;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_12:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
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
label_13:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_1(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_2(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_3(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_4(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_5(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_idx_6(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_18;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_18:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_16:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_3(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_4(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_5(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_idx_6(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
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
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_24;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_24:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_25;
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
label_25:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_20;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_20:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_21;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_21:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_22;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_22:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_bar_sync_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_26;
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
asm ("bar.sync 2,512;");
;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
asm ("bar.sync 2,512;");
;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_26:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_27;
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
label_27:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_bar_sync_imba_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_28;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
asm ("bar.sync 2,192;");
;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
asm ("bar.sync 2,192;");
;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_28:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_29;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
asm ("bar.sync 1,832;");
;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
asm ("bar.sync 1,832;");
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
label_29:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_1(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_2(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_3(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_4(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_5(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_imba_idx_6(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_33;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_33:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_34;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_34:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_35;
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
label_35:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_30;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_30:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_31;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_31:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_32;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_32:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_1(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_2(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_3(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_4(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_5(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_imba_idx_6(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_39;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
label_39:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_40;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_40:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_41;
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
label_41:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_36;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
label_36:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_37;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_37:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_38;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_38:;
}
template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_hfuse_lb_bar_sync_imba_idx_0(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 192)) goto label_42;
unsigned int blockDim_x_1;
blockDim_x_1 = 192;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 192;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 192;
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
}
asm ("bar.sync 2,192;");
;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
asm ("bar.sync 2,192;");
;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_42:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=192 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_43;
unsigned int blockDim_x_0;
blockDim_x_0 = 52;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) % 52;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 52 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 192) / 832;
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
asm ("bar.sync 1,832;");
;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
asm ("bar.sync 1,832;");
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
label_43:;
}
