#pragma once

__device__ __forceinline__ int32_t clamp(int32_t val, int32_t lower, int32_t upper) {
    return min(upper, max(lower, val));
}
__device__ __forceinline__ float clamp(float val, float lower, float upper) {
    return fminf(upper, fmaxf(lower, val));
}
__device__ __forceinline__ double clamp(double val, double lower, double upper) {
    return fmin(upper, fmax(lower, val));
}

// c = c + a * b
__device__ __forceinline__ float myfma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ double myfma(double a, double b, double c) { return fma(a, b, c); }
__device__ __forceinline__ void myfma(float a, float b, float *c) {
    *c = fmaf(a, b, *c);
}
__device__ __forceinline__ void myfma(double a, double b, double *c) {
    *c = fma(a, b, *c);
}

__device__ __forceinline__ float myfloor(float a) { return floorf(a); }
__device__ __forceinline__ double myfloor(double a) { return floor(a); }

__device__ __forceinline__ float myabs(float a) { return fabsf(a); }
__device__ __forceinline__ double myabs(double a) { return fabs(a); }

__device__ __forceinline__ float mymax(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double mymax(double a, double b) { return fmax(a, b); }

__device__ __forceinline__ int32_t floor2int(float a) { return __float2int_rd(a); }
__device__ __forceinline__ int32_t floor2int(double a) { return __double2int_rd(a); }

__device__ __forceinline__ float myexp(float a) { return __expf(a); }
__device__ __forceinline__ double myexp(double a) { return exp(a); }

// sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
__device__ __forceinline__ float vec3_norm(const float* __restrict__ dir) {
    return norm3df(dir[0], dir[1], dir[2]);
}
__device__ __forceinline__ double vec3_norm(const double* __restrict__ dir) {
    return norm3d(dir[0], dir[1], dir[2]);
}

// 1.0 / vec3_norm(dir
__device__ __forceinline__ float vec3_rnorm(const float* __restrict__ dir) {
    return rnorm3df(dir[0], dir[1], dir[2]);
}
__device__ __forceinline__ double vec3_rnorm(const double* __restrict__ dir) {
    return rnorm3d(dir[0], dir[1], dir[2]);
}

// Linear interp - subtract and fused multiply-add: (1-w) a + w b
__device__ __inline__ float lerp(float a, float b, float w) {
    return fmaf(w, b - a, a);
}
__device__ __inline__ double lerp(double a, double b, double w) {
    return fma(w, b - a, a);
}

__host__ __device__ __inline__ int round_up(int num, int multiple) {
    return ((num + multiple - 1) / multiple) * multiple;
}

template <typename T>
__host__ __device__ constexpr T div_round_up(T val, T divisor) {
        return (val + divisor - 1) / divisor;
}


template <int pow2>
__host__ __device__ __inline__ void fast_divmod_pow2(const int n, int& __restrict__ q, int& __restrict__ r) {
    q = n >> pow2;
    r = n & ((2 << (pow2 - 1)) - 1);
}

// The code below is based on section 4 Unsigned division of paper https://gmplib.org/~tege/divcnst-pldi94.pdf
// In current ORT, fast_divmod is used for calculating the position of a element in tensor,
// so unsigned integer division from the paper is good enough for ORT. The advantage is that div is very simple,
// then GPU compiler can do loop unroll easilly when divmod is called in a loop.
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cuda/shared_inc/fast_divmod.h
struct fast_divmod {
  fast_divmod(int d = 1) {
    d_ = d == 0 ? 1 : d;
    for (l_ = 0; l_ < 32; l_++)
      if ((1U << l_) >= d_) break;

    uint64_t one = 1;
    uint64_t m = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
    M_ = static_cast<uint32_t>(m);
  }

  __device__ inline int div(int n) const {
    uint32_t t = __umulhi(M_, n);
    return (t + n) >> l_;
  }

  __device__ inline int mod(int n) const {
    return n - div(n) * d_;
  }

  __device__ inline void divmod(int n, int& q, int& r) const {
    q = div(n);
    r = n - q * d_;
  }

  uint32_t d_;  // divisor
  uint32_t M_;  // m' in the paper.
  uint32_t l_;  // l_ = ceil(log2(d_))
};

// https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
template <typename T>
__device__ __inline__ T* shared_memory_proxy()
{
    // do we need an __align__() here? I don't think so...
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

struct Half2Sum
{
    __device__ __forceinline__ __half2 operator()(const __half2 &a, const __half2 &b) const
    {
        return __hadd2(a, b);
    }
};
