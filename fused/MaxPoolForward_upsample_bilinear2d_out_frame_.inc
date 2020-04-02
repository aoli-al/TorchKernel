template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 256;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
        int pw51;
        pw51 = index % pooled_width40;
        int ph52;
        ph52 = (index / pooled_width40) % pooled_height39;
        int c53;
        c53 = (index / pooled_width40 / pooled_height39) % channels36;
        int n54;
        n54 = index / pooled_width40 / pooled_height39 / channels36;
        int hstart55;
        hstart55 = ph52 * stride_h43 - pad_h45;
        int wstart56;
        wstart56 = pw51 * stride_w44 - pad_w46;
        int hend57;
        hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
        int wend58;
        wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
        while (hstart55 < 0)
            hstart55 += dilation_h47;
        while (wstart56 < 0)
            wstart56 += dilation_w48;
        accscalar_t32 maxval59;
        maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
        int maxidx60;
        maxidx60 = hstart55 * width38 + wstart56;
        bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
        for (int h = hstart55; h < hend57; h += dilation_h47) {
            for (int w = wstart56; w < wend58; w += dilation_w48) {
                scalar_t31 val61;
                val61 = bottom_data34[h * width38 + w];
                if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                    maxidx60 = h * width38 + w;
                    maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
                }
            }
        }
        top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
        top_mask50[index] = maxidx60;
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
template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_vfuse_idx_0(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 256;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
        int pw51;
        pw51 = index % pooled_width40;
        int ph52;
        ph52 = (index / pooled_width40) % pooled_height39;
        int c53;
        c53 = (index / pooled_width40 / pooled_height39) % channels36;
        int n54;
        n54 = index / pooled_width40 / pooled_height39 / channels36;
        int hstart55;
        hstart55 = ph52 * stride_h43 - pad_h45;
        int wstart56;
        wstart56 = pw51 * stride_w44 - pad_w46;
        int hend57;
        hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
        int wend58;
        wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
        while (hstart55 < 0)
            hstart55 += dilation_h47;
        while (wstart56 < 0)
            wstart56 += dilation_w48;
        accscalar_t32 maxval59;
        maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
        int maxidx60;
        maxidx60 = hstart55 * width38 + wstart56;
        bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
        for (int h = hstart55; h < hend57; h += dilation_h47) {
            for (int w = wstart56; w < wend58; w += dilation_w48) {
                scalar_t31 val61;
                val61 = bottom_data34[h * width38 + w];
                if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                    maxidx60 = h * width38 + w;
                    maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
                }
            }
        }
        top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
        top_mask50[index] = maxidx60;
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
template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_4;
unsigned int blockDim_x_1;
blockDim_x_1 = 256;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
    int pw51;
    pw51 = index % pooled_width40;
    int ph52;
    ph52 = (index / pooled_width40) % pooled_height39;
    int c53;
    c53 = (index / pooled_width40 / pooled_height39) % channels36;
    int n54;
    n54 = index / pooled_width40 / pooled_height39 / channels36;
    int hstart55;
    hstart55 = ph52 * stride_h43 - pad_h45;
    int wstart56;
    wstart56 = pw51 * stride_w44 - pad_w46;
    int hend57;
    hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
    int wend58;
    wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
    while (hstart55 < 0)
        hstart55 += dilation_h47;
    while (wstart56 < 0)
        wstart56 += dilation_w48;
    accscalar_t32 maxval59;
    maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
    int maxidx60;
    maxidx60 = hstart55 * width38 + wstart56;
    bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
    for (int h = hstart55; h < hend57; h += dilation_h47) {
        for (int w = wstart56; w < wend58; w += dilation_w48) {
            scalar_t31 val61;
            val61 = bottom_data34[h * width38 + w];
            if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                maxidx60 = h * width38 + w;
                maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
            }
        }
    }
    top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
    top_mask50[index] = maxidx60;
}
label_4:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_5;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
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
label_5:;
}
template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_6;
unsigned int blockDim_x_1;
blockDim_x_1 = 256;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
    int pw51;
    pw51 = index % pooled_width40;
    int ph52;
    ph52 = (index / pooled_width40) % pooled_height39;
    int c53;
    c53 = (index / pooled_width40 / pooled_height39) % channels36;
    int n54;
    n54 = index / pooled_width40 / pooled_height39 / channels36;
    int hstart55;
    hstart55 = ph52 * stride_h43 - pad_h45;
    int wstart56;
    wstart56 = pw51 * stride_w44 - pad_w46;
    int hend57;
    hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
    int wend58;
    wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
    while (hstart55 < 0)
        hstart55 += dilation_h47;
    while (wstart56 < 0)
        wstart56 += dilation_w48;
    accscalar_t32 maxval59;
    maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
    int maxidx60;
    maxidx60 = hstart55 * width38 + wstart56;
    bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
    for (int h = hstart55; h < hend57; h += dilation_h47) {
        for (int w = wstart56; w < wend58; w += dilation_w48) {
            scalar_t31 val61;
            val61 = bottom_data34[h * width38 + w];
            if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                maxidx60 = h * width38 + w;
                maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
            }
        }
    }
    top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
    top_mask50[index] = maxidx60;
}
label_6:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_7;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
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
label_7:;
}
template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_imba_idx_0(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)) goto label_8;
unsigned int blockDim_x_1;
blockDim_x_1 = 384;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
    int pw51;
    pw51 = index % pooled_width40;
    int ph52;
    ph52 = (index / pooled_width40) % pooled_height39;
    int c53;
    c53 = (index / pooled_width40 / pooled_height39) % channels36;
    int n54;
    n54 = index / pooled_width40 / pooled_height39 / channels36;
    int hstart55;
    hstart55 = ph52 * stride_h43 - pad_h45;
    int wstart56;
    wstart56 = pw51 * stride_w44 - pad_w46;
    int hend57;
    hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
    int wend58;
    wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
    while (hstart55 < 0)
        hstart55 += dilation_h47;
    while (wstart56 < 0)
        wstart56 += dilation_w48;
    accscalar_t32 maxval59;
    maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
    int maxidx60;
    maxidx60 = hstart55 * width38 + wstart56;
    bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
    for (int h = hstart55; h < hend57; h += dilation_h47) {
        for (int w = wstart56; w < wend58; w += dilation_w48) {
            scalar_t31 val61;
            val61 = bottom_data34[h * width38 + w];
            if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                maxidx60 = h * width38 + w;
                maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
            }
        }
    }
    top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
    top_mask50[index] = maxidx60;
}
label_8:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_9;
unsigned int blockDim_x_0;
blockDim_x_0 = 384;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
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
template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_imba_idx_0(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)) goto label_10;
unsigned int blockDim_x_1;
blockDim_x_1 = 384;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
    int pw51;
    pw51 = index % pooled_width40;
    int ph52;
    ph52 = (index / pooled_width40) % pooled_height39;
    int c53;
    c53 = (index / pooled_width40 / pooled_height39) % channels36;
    int n54;
    n54 = index / pooled_width40 / pooled_height39 / channels36;
    int hstart55;
    hstart55 = ph52 * stride_h43 - pad_h45;
    int wstart56;
    wstart56 = pw51 * stride_w44 - pad_w46;
    int hend57;
    hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
    int wend58;
    wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
    while (hstart55 < 0)
        hstart55 += dilation_h47;
    while (wstart56 < 0)
        wstart56 += dilation_w48;
    accscalar_t32 maxval59;
    maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
    int maxidx60;
    maxidx60 = hstart55 * width38 + wstart56;
    bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
    for (int h = hstart55; h < hend57; h += dilation_h47) {
        for (int w = wstart56; w < wend58; w += dilation_w48) {
            scalar_t31 val61;
            val61 = bottom_data34[h * width38 + w];
            if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                maxidx60 = h * width38 + w;
                maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
            }
        }
    }
    top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
    top_mask50[index] = maxidx60;
}
label_10:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_11;
unsigned int blockDim_x_0;
blockDim_x_0 = 384;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
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
label_11:;
}
