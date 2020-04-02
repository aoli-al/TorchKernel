template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_lb_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48;
    smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    __syncthreads();
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType33 bOffset49;
        bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        input_t32 bVal50;
        bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            IndexType33 bin51;
            bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        IndexType33 aOffset52;
        aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
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
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48;
    smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    __syncthreads();
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType33 bOffset49;
        bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        input_t32 bVal50;
        bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            IndexType33 bin51;
            bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        IndexType33 aOffset52;
        aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
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
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_bar_sync_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
asm ("bar.sync 1,512;");
;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
asm ("bar.sync 1,512;");
;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_8:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_9;
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
label_9:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_10:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_11;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_12;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_12:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_13;
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
label_13:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_10:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_11;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_11:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_13;
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
label_13:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_12;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_12:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_10:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_13;
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
label_13:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_11;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_12;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_12:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_17;
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
label_17:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_15:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_16:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_14:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_17;
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
label_17:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_15;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_16;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_16:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_bar_sync_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
asm ("bar.sync 1,512;");
;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
asm ("bar.sync 1,512;");
;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_18:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_19;
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
label_19:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_bar_sync_imba_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_20;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
asm ("bar.sync 1,160;");
;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
asm ("bar.sync 1,160;");
;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_20:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_21;
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
label_21:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_imba_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_22;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_22:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_23;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_24;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_24:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_25;
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
label_25:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_imba_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_22;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_22:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_23;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_23:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_25;
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
label_25:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_24;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_24:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_imba_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_22;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_25;
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
label_25:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_23;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_23:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_24;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_24:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_imba_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_26;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_26:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_27;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_27:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_28;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_28:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_29;
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
label_29:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_imba_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_26;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_26:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_27;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_27:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_29;
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
label_29:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_28;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_28:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_imba_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_26;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_26:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_29;
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
label_29:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_27;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_27:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_28;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_28:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_bar_sync_imba_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 160)) goto label_30;
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
asm ("bar.sync 1,160;");
;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
asm ("bar.sync 1,160;");
;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_30:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=160 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_31;
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
label_31:;
}
