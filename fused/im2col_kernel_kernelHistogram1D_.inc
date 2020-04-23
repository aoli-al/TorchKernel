template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_lb_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
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
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
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
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_3(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_4(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_5(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_idx_6(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_3(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_4(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_5(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
template <typename dt0, typename output_t27, typename input_t28, typename IndexType29, int ADims30, int PDims31, int BDims32, at::native::CUDAHistogramMemoryType MemoryType33 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op34>
 __global__ __launch_bounds__(1024, 1) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_idx_6(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t27, IndexType29> a35, TensorInfo<output_t27, IndexType29> p36, TensorInfo<input_t28, IndexType29> b37, int nbins38, input_t28 minvalue39, input_t28 maxvalue40, IndexType29 totalElements41, Op34 getOp42)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16 = index % width_col14;
        int idx17 = index / width_col14;
        int64_t h_out18 = idx17 % height_col13;
        int64_t channel_in19 = idx17 / height_col13;
        int64_t channel_out20 = channel_in19 * kernel_height5 * kernel_width6;
        int64_t h_in21 = h_out18 * stride_height9 - pad_height7;
        int64_t w_in22 = w_out16 * stride_width10 - pad_width8;
        dt0 *block_col23 = data_col15 + (channel_out20 * height_col13 + h_out18) * width_col14 + w_out16;
        const dt0 *block_im24 = data_im2 + (channel_in19 * height3 + h_in21) * width4 + w_in22;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h25 = h_in21 + i * dilation_height11;
                int64_t w26 = w_in22 + j * dilation_width12;
                * block_col23 = (h25 >= 0 && w26 >= 0 && h25 < height3 && w26 < width4) ? block_im24[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                block_col23 += height_col13 * width_col14;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    extern unsigned char my_smem43[] __attribute__((shared));
    output_t27 *smem44 = nullptr;
    smem44 = reinterpret_cast<output_t27 *>(my_smem43);
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        smem44[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType29 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements41; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType29 bOffset45 = IndexToOffset<input_t28, IndexType29, BDims32>::get(linearIndex, b37);
        const input_t28 bVal46 = b37.data[bOffset45];
        if (bVal46 >= minvalue39 && bVal46 <= maxvalue40) {
            const IndexType29 bin47 = getBin<input_t28, IndexType29>(bVal46, minvalue39, maxvalue40, nbins38);
            atomicAdd(&smem44[bin47], getOp42(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType29 i = threadIdx_x_1; i < a35.sizes[0]; i += blockDim_x_1) {
        const IndexType29 aOffset48 = IndexToOffset<output_t27, IndexType29, ADims30>::get(i, a35);
        atomicAdd(&a35.data[aOffset48], smem44[i]);
    }
}
}
