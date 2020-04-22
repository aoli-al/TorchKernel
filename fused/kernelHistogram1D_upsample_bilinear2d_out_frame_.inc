template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
    int index8;
    index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    int batchsize9;
    batchsize9 = idata6.size(0);
    int channels10;
    channels10 = idata6.size(1);
    int height111;
    height111 = idata6.size(2);
    int width112;
    width112 = idata6.size(3);
    int height213;
    height213 = odata7.size(2);
    int width214;
    width214 = odata7.size(3);
    if (index8 < n2) {
        int w215;
        w215 = index8 % width214;
        int h216;
        h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            int h127;
            h127 = h216;
            int w128;
            w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    scalar_t0 val29;
                    val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        accscalar_t1 h1r17;
        h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        int h118;
        h118 = h1r17;
        int h1p19;
        h1p19 = (h118 < height111 - 1) ? 1 : 0;
        accscalar_t1 h1lambda20;
        h1lambda20 = h1r17 - h118;
        accscalar_t1 h0lambda21;
        h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        accscalar_t1 w1r22;
        w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        int w123;
        w123 = w1r22;
        int w1p24;
        w1p24 = (w123 < width112 - 1) ? 1 : 0;
        accscalar_t1 w1lambda25;
        w1lambda25 = w1r22 - w123;
        accscalar_t1 w0lambda26;
        w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                accscalar_t1 val30;
                val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
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
    int index8;
    index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    int batchsize9;
    batchsize9 = idata6.size(0);
    int channels10;
    channels10 = idata6.size(1);
    int height111;
    height111 = idata6.size(2);
    int width112;
    width112 = idata6.size(3);
    int height213;
    height213 = odata7.size(2);
    int width214;
    width214 = odata7.size(3);
    if (index8 < n2) {
        int w215;
        w215 = index8 % width214;
        int h216;
        h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            int h127;
            h127 = h216;
            int w128;
            w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    scalar_t0 val29;
                    val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        accscalar_t1 h1r17;
        h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        int h118;
        h118 = h1r17;
        int h1p19;
        h1p19 = (h118 < height111 - 1) ? 1 : 0;
        accscalar_t1 h1lambda20;
        h1lambda20 = h1r17 - h118;
        accscalar_t1 h0lambda21;
        h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        accscalar_t1 w1r22;
        w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        int w123;
        w123 = w1r22;
        int w1p24;
        w1p24 = (w123 < width112 - 1) ? 1 : 0;
        accscalar_t1 w1lambda25;
        w1lambda25 = w1r22 - w123;
        accscalar_t1 w0lambda26;
        w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                accscalar_t1 val30;
                val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_bar_sync_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_9;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_9:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_13:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
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
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_13;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
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
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_17:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
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
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_17;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
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
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_bar_sync_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_19;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_19:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_bar_sync_imba_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 352)) goto label_20;
unsigned int blockDim_x_1;
blockDim_x_1 = 352;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 352;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 352 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 352;
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
asm ("bar.sync 1,352;");
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
asm ("bar.sync 1,352;");
;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_20:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=352 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_21;
unsigned int blockDim_x_0;
blockDim_x_0 = 672;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 352) % 672;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 352) / 672 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 352) / 672;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_21:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_bar_sync_imba_idx_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 352)) goto label_22;
unsigned int blockDim_x_1;
blockDim_x_1 = 352;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 352;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 352 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 352;
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
asm ("bar.sync 1,352;");
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
asm ("bar.sync 1,352;");
;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_22:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=352 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_23;
unsigned int blockDim_x_0;
blockDim_x_0 = 672;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 352) % 672;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 352) / 672 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 352) / 672;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_23:;
}
