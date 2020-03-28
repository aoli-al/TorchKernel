template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(512, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_lb_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 512;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16;
        w_out16 = index % width_col14;
        index /= width_col14;
        int64_t h_out17;
        h_out17 = index % height_col13;
        int64_t channel_in18;
        channel_in18 = index / height_col13;
        int64_t channel_out19;
        channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
        int64_t h_in20;
        h_in20 = h_out17 * stride_height9 - pad_height7;
        int64_t w_in21;
        w_in21 = w_out16 * stride_width10 - pad_width8;
        data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
        data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h22;
                h22 = h_in20 + i * dilation_height11;
                int64_t w23;
                w23 = w_in21 + j * dilation_width12;
                * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                data_col15 += height_col13 * width_col14;
            }
        }
    }
}
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
    extern unsigned char my_smem40[] __attribute__((shared));
    output_t24 *smem41;
    smem41 = nullptr;
    smem41 = reinterpret_cast<output_t24 *>(my_smem40);
    for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
        smem41[i] = 0;
    }
    __syncthreads();
    for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType26 bOffset42;
        bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
        input_t25 bVal43;
        bVal43 = b34.data[bOffset42];
        if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
            IndexType26 bin44;
            bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
            atomicAdd(&smem41[bin44], getOp39(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
        IndexType26 aOffset45;
        aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
        atomicAdd(&a32.data[aOffset45], smem41[i]);
    }
}
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(512, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_vfuse_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 512;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16;
        w_out16 = index % width_col14;
        index /= width_col14;
        int64_t h_out17;
        h_out17 = index % height_col13;
        int64_t channel_in18;
        channel_in18 = index / height_col13;
        int64_t channel_out19;
        channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
        int64_t h_in20;
        h_in20 = h_out17 * stride_height9 - pad_height7;
        int64_t w_in21;
        w_in21 = w_out16 * stride_width10 - pad_width8;
        data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
        data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h22;
                h22 = h_in20 + i * dilation_height11;
                int64_t w23;
                w23 = w_in21 + j * dilation_width12;
                * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                data_col15 += height_col13 * width_col14;
            }
        }
    }
}
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
    extern unsigned char my_smem40[] __attribute__((shared));
    output_t24 *smem41;
    smem41 = nullptr;
    smem41 = reinterpret_cast<output_t24 *>(my_smem40);
    for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
        smem41[i] = 0;
    }
    __syncthreads();
    for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType26 bOffset42;
        bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
        input_t25 bVal43;
        bVal43 = b34.data[bOffset42];
        if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
            IndexType26 bin44;
            bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
            atomicAdd(&smem41[bin44], getOp39(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
        IndexType26 aOffset45;
        aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
        atomicAdd(&a32.data[aOffset45], smem41[i]);
    }
}
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_bar_sync_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_8;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_8:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_9;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
asm ("bar.sync 1,512;");
;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
asm ("bar.sync 1,512;");
;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_9:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_10;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_10:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_11;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_12;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_12:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_13:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_11;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_12;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_12:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_10;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_10:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_13:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 0) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_11;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_11:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_12;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_12:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_13:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_10;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_10:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_14;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_14:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_15;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_16;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_16:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_17:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_1(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_15;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_16;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_16:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_14;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_14:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_17:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_2(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_15;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_15:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_16;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_16:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_17:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_14;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_14:;
}
template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
 __global__ __launch_bounds__(1024, 2) void im2col_kernel_kernelHistogram1D_fused_kernel_hfuse_lb_bar_sync_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_18;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_18:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
asm ("bar.sync 1,512;");
;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
asm ("bar.sync 1,512;");
;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_19:;
}
