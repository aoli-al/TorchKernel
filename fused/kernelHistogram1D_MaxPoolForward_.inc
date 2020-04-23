template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_vfuse_idx_0(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_0(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_1(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_2(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_3(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_idx_4(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_3(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
template <typename output_t32, typename input_t33, typename IndexType34, int ADims35, int PDims36, int BDims37, at::native::CUDAHistogramMemoryType MemoryType38 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op39, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void kernelHistogram1D_MaxPoolForward_fused_kernel_hfuse_lb_idx_4(TensorInfo<output_t32, IndexType34> a40, TensorInfo<output_t32, IndexType34> p41, TensorInfo<input_t33, IndexType34> b42, int nbins43, input_t33 minvalue44, input_t33 maxvalue45, IndexType34 totalElements46, Op39 getOp47, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem48[] __attribute__((shared));
    output_t32 *smem49 = nullptr;
    smem49 = reinterpret_cast<output_t32 *>(my_smem48);
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        smem49[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType34 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements46; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType34 bOffset50 = IndexToOffset<input_t33, IndexType34, BDims37>::get(linearIndex, b42);
        const input_t33 bVal51 = b42.data[bOffset50];
        if (bVal51 >= minvalue44 && bVal51 <= maxvalue45) {
            const IndexType34 bin52 = getBin<input_t33, IndexType34>(bVal51, minvalue44, maxvalue45, nbins43);
            atomicAdd(&smem49[bin52], getOp47(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType34 i = threadIdx_x_1; i < a40.sizes[0]; i += blockDim_x_1) {
        const IndexType34 aOffset53 = IndexToOffset<output_t32, IndexType34, ADims35>::get(i, a40);
        atomicAdd(&a40.data[aOffset53], smem49[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
        int pw20 = index % pooled_width9;
        int ph21 = (index / pooled_width9) % pooled_height8;
        int c22 = (index / pooled_width9 / pooled_height8) % channels5;
        int n23 = index / pooled_width9 / pooled_height8 / channels5;
        int hstart24 = ph21 * stride_h12 - pad_h14;
        int wstart25 = pw20 * stride_w13 - pad_w15;
        int hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
        int wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
        while (hstart24 < 0)
            hstart24 += dilation_h16;
        while (wstart25 < 0)
            wstart25 += dilation_w17;
        accscalar_t1 maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
        int maxidx29 = hstart24 * width7 + wstart25;
        const scalar_t0 *btm_data30 = bottom_data3 + (n23 * channels5 + c22) * height6 * width7;
        for (int h = hstart24; h < hend26; h += dilation_h16) {
            for (int w = wstart25; w < wend27; w += dilation_w17) {
                scalar_t0 val31 = btm_data30[h * width7 + w];
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
