template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_vfuse_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_3(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_4(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_5(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_6(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 1) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 1) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_3(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_4(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_5(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_6(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType33 bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        const input_t32 bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            const IndexType33 bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        const IndexType33 aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    int index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    const int batchsize9 = idata6.size(0);
    const int channels10 = idata6.size(1);
    const int height111 = idata6.size(2);
    const int width112 = idata6.size(3);
    const int height213 = odata7.size(2);
    const int width214 = odata7.size(3);
    if (index8 < n2) {
        const int w215 = index8 % width214;
        const int h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            const int h127 = h216;
            const int w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    const scalar_t0 val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        const accscalar_t1 h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        const int h118 = h1r17;
        const int h1p19 = (h118 < height111 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda20 = h1r17 - h118;
        const accscalar_t1 h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        const accscalar_t1 w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        const int w123 = w1r22;
        const int w1p24 = (w123 < width112 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda25 = w1r22 - w123;
        const accscalar_t1 w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                const accscalar_t1 val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
