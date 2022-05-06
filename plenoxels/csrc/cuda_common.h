#pragma once


#define _SOFTPLUS_M1(x) (__logf(1 + __expf((x) - 1)))
#define _SIGMOID(x) (1 / (1 + __expf(-(x))))


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements, T n_threads_linear) {
	return (uint32_t)div_round_up(n_elements, n_threads_linear);
}


__device__ __inline__ float3 diff_prod(const float3 &a, const float3 &b, const float &c) {
    // (a - b) * c
    return make_float3(
        (a.x - b.x) * c,
        (a.y - b.y) * c,
        (a.z - b.z) * c
    );
}
__device__ __inline__ void diff_prod(const float  * __restrict__ a,
                                     const float  * __restrict__ b,
                                     const float  & __restrict__ c,
                                           float3 & __restrict__ out) {
    // out += (a - b) * c
    out.x += (a[0] - b[0]) * c;
    out.y += (a[1] - b[1]) * c;
    out.z += (a[2] - b[2]) * c;
}


__device__ __inline__ float prod_diff(const float &a, const float &b, const float &c, const float &d) {
    // a * b - c * d (using kahan sum)
    float cd = __fmul_rn(c, d);  // use intrinsic to avoid compiler optimizing this out.
    float err = fmaf(-c, d, cd);
    float dop = fmaf(a, b, -cd);
    return dop + err;
}



__device__ __inline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ __inline__ float3 operator+(const float3 &a, const float &b) {
    return make_float3(a.x+b, a.y+b, a.z+b);
}
__device__ __inline__ float3 operator+(const float &a, const float3 &b) {
    return make_float3(a+b.x, a+b.y, a+b.z);
}

__device__ __inline__ float3 operator +=(float3 & __restrict__ a, const float3 & __restrict__ b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}


__device__ __inline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ __inline__ float3 operator-(const float3 &a, const float &b) {
    return make_float3(a.x-b, a.y-b, a.z-b);
}
__device__ __inline__ float3 operator-(const float &a, const float3 &b) {
    return make_float3(a-b.x, a-b.y, a-b.z);
}

__device__ __inline__ float3 operator -=(float3 &a, const float3 &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}


__device__ __inline__ float3 operator/(const float3 &a, const float3 &b) {
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}
__device__ __inline__ float3 operator/(const float3 &a, const float &b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}
__device__ __inline__ float3 operator/(const float &a, const float3 &b) {
    return make_float3(a/b.x, a/b.y, a/b.z);
}

__device__ __inline__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ __inline__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}
__device__ __inline__ float3 operator*(const float &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ __inline__ float dot(const float3 & a, const float3 & b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}



__host__ __device__ __inline__ void clamp_coord(float3 & __restrict__ q_out, float lower, float upper) {
    q_out.x = q_out.x < lower ? lower : (upper < q_out.x ? upper : q_out.x);
    q_out.y = q_out.y < lower ? lower : (upper < q_out.y ? upper : q_out.y);
    q_out.z = q_out.z < lower ? lower : (upper < q_out.z ? upper : q_out.z);
}
__host__ __device__ __inline__ void clamp_coord(float * __restrict__ q_out, float lower, float upper) {
    q_out[0] = q_out[0] < lower ? lower : (upper < q_out[0] ? upper : q_out[0]);
    q_out[1] = q_out[1] < lower ? lower : (upper < q_out[1] ? upper : q_out[1]);
    q_out[2] = q_out[2] < lower ? lower : (upper < q_out[2] ? upper : q_out[2]);
}


__device__ __inline__ void transform_coord(float* __restrict__ q,
                                           const float* __restrict__ offset,
                                           const float* __restrict__ scaling) {
    #pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + scaling[i] * q[i];
    }
}
__device__ __inline__ void transform_coord(float3 & __restrict__ q,
                                           const float* __restrict__ offset,
                                           const float* __restrict__ scaling) {
    q.x = offset[0] + scaling[0] * q.x;
    q.y = offset[1] + scaling[1] * q.y;
    q.z = offset[2] + scaling[2] * q.z;
}



template<typename T>
__host__ __device__ __inline__ static T _norm(const T * __restrict__ dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}
template<typename T>
__host__ __device__ __inline__ static T _norm(const T & __restrict__ d0, const T & __restrict__ d1, const T & __restrict__ d2) {
    return sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
}

template<typename T>
__host__ __device__ __inline__ static void _normalize(T* dir)
{
    T norm = _norm(dir);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
}

template<typename T>
__host__ __device__ __inline__ static T _dot3(const T* __restrict__ u, const T* __restrict__ v)
{
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

__device__ __inline__ float _get_delta_scale(
    const float* __restrict__ scaling,
    float3 & __restrict__ dir)
{
    dir.x *= scaling[0];
    dir.y *= scaling[1];
    dir.z *= scaling[2];
    float delta_scale = 1.f / _norm<float>(dir.x, dir.y, dir.z);
    dir.x *= delta_scale;
    dir.y *= delta_scale;
    dir.z *= delta_scale;
    return delta_scale;
}

__device__ __inline__ void _dda_unit(
        const float3& __restrict__ cen,
        const float3& __restrict__ invdir,
        float* __restrict__ tmin,
        float* __restrict__ tmax)
{
    // Intersect unit AABB
    float t1, t2;

    t1 = -cen.x * invdir.x;
    t2 = t1 + invdir.x;
    *tmin = min(t1, t2);
    *tmax = max(t1, t2);

    t1 = -cen.y * invdir.y;
    t2 = t1 + invdir.y;
    *tmin = max(*tmin, min(t1, t2));
    *tmax = min(*tmax, max(t1, t2));

    t1 = -cen.z * invdir.z;
    t2 = t1 + invdir.z;
    *tmin = max(*tmin, min(t1, t2));
    *tmax = min(*tmax, max(t1, t2));
}
