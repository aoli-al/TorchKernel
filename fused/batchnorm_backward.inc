
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


template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_backward_kernel(
    const PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t> input,
    const PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t> grad_output,
    PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t> grad_input,
    PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> grad_weight,
    PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> grad_bias,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> weight,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> running_mean,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, index_t> running_var,
    const PackedTensorAccessor<accscalar_t, 1, DefaultPtrTraits, index_t> save_mean,
    const PackedTensorAccessor<accscalar_t, 1, DefaultPtrTraits, index_t> save_invstd,
    bool train,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;
  index_t N = grad_output.size(0) * grad_output.size(2);

  accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<accscalar_t>(running_mean[plane]);
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(running_var[plane]) + epsilon);
  }

  accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
  accscalar_t norm = accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<scalar_t, accscalar_t, PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t>> g(mean, input, grad_output);


  // first the reductions each thread does separately
struct at::native::(anonymous namespace)::Float2<double, double> sum = static_cast<struct at::native::(anonymous namespace)::Float2<double, double>>(0);
for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
  for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
    sum += g(batch, plane, x);
  }
}

// first warpSum to get one value per thread to
// one value per warp
sum = warpSum(sum);

// this writes each warps  item into shared memory
// there are at most WARP_SIZE items left because
// there are at most WARP_SIZE**2 threads at the beginning
__shared__ struct at::native::(anonymous namespace)::Float2<double, double> shared[WARP_SIZE];
__syncthreads();
int tid = threadIdx.x + threadIdx.y * blockDim.x;
if (tid % WARP_SIZE == 0) {
  shared[tid / WARP_SIZE] = sum;
}
if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
  // zero out the other entries in shared
  shared[tid] = (struct at::native::(anonymous namespace)::Float2<double, double>)0;
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
at::native::Float2<double, double> res = shared[0];



  //Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t,
  //                                                                                 PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, index_t>>>(g, grad_output, plane);
  accscalar_t grad_output_sum = res.v1;
  accscalar_t dot_p = res.v2;

  accscalar_t grad_mean = grad_output_sum * norm;
  accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        scalar_t go = grad_output[batch][plane][x];
        if (train) {
          scalar_t inp = input[batch][plane][x];
          accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<scalar_t>(grad_output_sum);
    }
  }
}
