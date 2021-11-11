#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <cuda_profiler_api.h>


#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"
#include <c10/macros/Macros.h>

#define THREAD_1 256
#define THREAD_2 768
#define _stringfy(x) #x
#define STRINGFY(x) _stringfy(x)

namespace at {
namespace native {

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}


using namespace at::cuda;
using namespace at::cuda::detail;

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct SumOp {
  __device__ SumOp(const PTA& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PTA& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % WARP_SIZE == 0) {
    shared[tid / WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t, typename accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> input,
    PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> output,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> bias,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}

template<typename T>
struct InvStd {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / device_sqrt(var + epsilon);
    }
    return invstd;
  }
};

template<typename T>
struct Var {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};



template <template<typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5,
    const stat_accscalar_t3 epsilon6,
    const stat_accscalar_t3 momentum7,
    PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8,
    PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9,
    PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10,
    PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) {
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




#define THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM 100
#define THRESH_NUMBER_BINS_FOR_GLOBAL_MEM 1000
#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
       i += gridDim.x * blockDim.x)

/*
  Memory types used for the 3 histogram implementations.
  See `CUDA_tensor_histogram` below.
 */
enum class CUDAHistogramMemoryType { SHARED, MULTI_BLOCK, GLOBAL };
namespace {
  template<typename input_t, typename IndexType>
  __device__ static IndexType getBin(input_t bVal, input_t minvalue, input_t maxvalue, int nbins) {
    IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
    // (only applicable for histc)
    // while each bin is inclusive at the lower end and exclusive at the higher, i.e. [start, end)
    // the last bin is inclusive at both, i.e. [start, end], in order to include maxvalue if exists
    // therefore when bin == nbins, adjust bin to the last bin
    if (bin == nbins) bin -= 1;
    return bin;
  }
}

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename output_t29,
    typename input_t30,
    typename IndexType31,
    int ADims32,
    int PDims33,
    int BDims34,
    CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK,
    typename Op36>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void kernelHistogram1D(
    TensorInfo<output_t29, IndexType31> a37, /* output */
    TensorInfo<output_t29, IndexType31> p38, /* partial output */
    TensorInfo<input_t30, IndexType31> b39, /* input */
    int nbins40,
    input_t30 minvalue41,
    input_t30 maxvalue42,
    IndexType31 totalElements43,
    Op36 getOp44) {
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem45[] __attribute__((shared));
    output_t29 *smem46 = nullptr;
    smem46 = reinterpret_cast<output_t29 *>(my_smem45);
    for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
        smem46[i] = 0;
    }
    __syncthreads();
    for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType31 bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
        const input_t30 bVal48 = b39.data[bOffset47];
        if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
            const IndexType31 bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
            atomicAdd(& smem46[bin49], getOp44(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
        const IndexType31 aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
        atomicAdd(& a37.data[aOffset50], smem46[i]);
    }
}



#include "kernelHistogram1D_batch_norm_collect_statistics_kernel_.inc"

inline int64_t getFreeGlobalMemory() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  AT_ASSERTM(
      cudaGetLastError() == cudaSuccess,
      "CUDA_tensor_histogram failed to get free global memory");
  return static_cast<int64_t>(free_mem);
}
template <typename input_hist_t, typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_, double epsilon
    ) {
  printf("2\n");
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output_hist = native::zeros({nbins}, device(DeviceType::CUDA).dtype(self_hist.scalar_type()));
  input_hist_t minvalue = min;
  input_hist_t maxvalue = max;
  if (min == max) {
    minvalue = *self_hist.min().cpu().data<input_hist_t>();
    maxvalue = *self_hist.max().cpu().data<input_hist_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  printf("3\n");
  {
  checkBackend("CUDA_tensor_histogram", {output_hist, self_hist}, Backend::CUDA);
  auto totalElements = self_hist.numel();

  const dim3 block = getApplyBlock();
  dim3 grid;
  int64_t curDevice = current_device();

  grid.x = 10000;

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(input_hist_t) + 8; // 8 guard bytes
  auto maxGlobalMem = getFreeGlobalMemory();
  auto multiBlockMem = nbins * grid.x * sizeof(input_hist_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
    printf("6\n");
  if (nbins < THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM &&
      sharedMem < maxSharedMem) {
    printf("shared\n");
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (
      nbins < THRESH_NUMBER_BINS_FOR_GLOBAL_MEM &&
      multiBlockMem < (maxGlobalMem / 2)) {
    // check against half of free mem to be extra safe
    // due to cached allocator, we may anyway have slightly more free mem
    printf("mb\n");
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  }

  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = getTensorInfo<input_hist_t, IndexType>(output_hist);
  auto bInfo = getTensorInfo<input_hist_t, IndexType>(self_hist);
  TensorInfo<input_hist_t, IndexType> pInfo(nullptr, 0, {}, {});
  Tensor partial_output_hist;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    partial_output_hist = native::zeros({grid.x, nbins}, output_hist.options());
    pInfo = getTensorInfo<input_hist_t, IndexType>(partial_output_hist);
  }

  printf("7\n");
  printf("10\n");
  // Launch kernel
  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  dummy_mean_ = at::empty({0}, input_options);
  dummy_var_ = at::empty({0}, input_options);
  // promote only mean_/invstd_ precision
  if (input_.scalar_type() == at::ScalarType::Half) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_options);
  invstd_ = at::empty({n_input}, input_options);
  auto mean = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("input: %d %d \n", input.size(0), input.size(1));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
  cudaProfilerStart();
  batch_norm_collect_statistics_kernel<InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, getStreamFromPool(true)>>>
    (input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
      <<<grid,
        block,
        sharedMem,
        getStreamFromPool(true)>>>(
          aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
  cudaDeviceSynchronize();
  cudaProfilerStop();
  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, mean_);
}
}
template <typename input_hist_t, typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor> _histc_cuda_fused(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_, double epsilon
    ) {
  printf("2\n");
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output_hist = native::zeros({nbins}, device(DeviceType::CUDA).dtype(self_hist.scalar_type()));
  input_hist_t minvalue = min;
  input_hist_t maxvalue = max;
  if (min == max) {
    minvalue = *self_hist.min().cpu().data<input_hist_t>();
    maxvalue = *self_hist.max().cpu().data<input_hist_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  printf("3\n");
  {
  checkBackend("CUDA_tensor_histogram", {output_hist, self_hist}, Backend::CUDA);
  auto totalElements = self_hist.numel();

  const dim3 block = getApplyBlock();
  dim3 grid;
  int64_t curDevice = current_device();

  grid.x = 10000;

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(input_hist_t) + 8; // 8 guard bytes
  auto maxGlobalMem = getFreeGlobalMemory();
  auto multiBlockMem = nbins * grid.x * sizeof(input_hist_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
    printf("6\n");
  if (nbins < THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM &&
      sharedMem < maxSharedMem) {
    printf("shared\n");
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (
      nbins < THRESH_NUMBER_BINS_FOR_GLOBAL_MEM &&
      multiBlockMem < (maxGlobalMem / 2)) {
    // check against half of free mem to be extra safe
    // due to cached allocator, we may anyway have slightly more free mem
    printf("mb\n");
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  }

  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = getTensorInfo<input_hist_t, IndexType>(output_hist);
  auto bInfo = getTensorInfo<input_hist_t, IndexType>(self_hist);
  TensorInfo<input_hist_t, IndexType> pInfo(nullptr, 0, {}, {});
  Tensor partial_output_hist;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    partial_output_hist = native::zeros({grid.x, nbins}, output_hist.options());
    pInfo = getTensorInfo<input_hist_t, IndexType>(partial_output_hist);
  }

  printf("7\n");
  printf("10\n");
  // Launch kernel
  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  dummy_mean_ = at::empty({0}, input_options);
  dummy_var_ = at::empty({0}, input_options);
  // promote only mean_/invstd_ precision
  if (input_.scalar_type() == at::ScalarType::Half) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_options);
  invstd_ = at::empty({n_input}, input_options);
  auto mean = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  THCudaCheck(cudaGetLastError());

    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    cudaProfilerStart();
    #define CALL(i,type,thread) kernelHistogram1D_batch_norm_collect_statistics_kernel_fused_kernel_##type##_idx_##i<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),\
  InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<10000, thread, sharedMem, stream>>>\
         (aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,\
    input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);\
    cudaDeviceSynchronize()

      CALL(0, vfuse,512);
      CALL(0, vfuse_lb,512);
      CALL(0, hfuse,1024);
      CALL(0, hfuse_lb,1024);
      CALL(1, hfuse,1024);
      CALL(1, hfuse_lb,1024);
      CALL(2, hfuse,1024);
      CALL(2, hfuse_lb,1024);
      CALL(3, hfuse,1024);
      CALL(3, hfuse_lb,1024);
      CALL(4, hfuse,1024);
      CALL(4, hfuse_lb,1024);
      CALL(5, hfuse,1024);
      CALL(5, hfuse_lb,1024);
      CALL(6, hfuse,1024);
      CALL(6, hfuse_lb,1024);

  cudaDeviceSynchronize();
    cudaProfilerStop();
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, mean_);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> hist_norm(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
  Tensor& input_) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_fused<scalar_t, scalar_t, int32_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>()
    , input_, 0.2
  );
  });
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template<scalar_t, scalar_t, int32_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>()
    , input_, 0.2
  );
  });
}

} // namespace native
} // namespace at
