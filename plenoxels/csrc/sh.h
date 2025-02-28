#pragma once


// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __constant__ const float C4[] = {
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
};


// Calculate basis functions depending on format, for given view directions
template <int basis_dim>
__device__ void calc_sh_basis(
    const float3& __restrict__ dir,
    float* __restrict__ out)
{
    out[0] = C0;
    const float xx = dir.x * dir.x, yy = dir.y * dir.y, zz = dir.z * dir.z;
    const float xy = dir.x * dir.y, yz = dir.y * dir.z, xz = dir.x * dir.z;
    switch (basis_dim) {
        case 25:
            out[16] = C4[0] * xy * (xx - yy);
            out[17] = C4[1] * yz * (3 * xx - yy);
            out[18] = C4[2] * xy * (7 * zz - 1.f);
            out[19] = C4[3] * yz * (7 * zz - 3.f);
            out[20] = C4[4] * (zz * (35 * zz - 30) + 3);
            out[21] = C4[5] * xz * (7 * zz - 3);
            out[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
            out[23] = C4[7] * xz * (xx - 3 * yy);
            out[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
            [[fallthrough]];
        case 16:
            out[9] = C3[0] * dir.y * (3 * xx - yy);
            out[10] = C3[1] * xy * dir.z;
            out[11] = C3[2] * dir.y * (4 * zz - xx - yy);
            out[12] = C3[3] * dir.z * (2 * zz - 3 * xx - 3 * yy);
            out[13] = C3[4] * dir.x * (4 * zz - xx - yy);
            out[14] = C3[5] * dir.z * (xx - yy);
            out[15] = C3[6] * dir.x * (xx - 3 * yy);
            [[fallthrough]];
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * dir.y;
            out[2] = C1 * dir.z;
            out[3] = -C1 * dir.x;
    }
}


// Calculate basis functions depending on format, for given view directions
template <int basis_dim>
__device__ void calc_sh_basis(
    const float * const __restrict__ dir,
          float *       __restrict__ out)
{
    out[0] = C0;
    const float xx = dir[0] * dir[0], yy = dir[1] * dir[1], zz = dir[2] * dir[2];
    const float xy = dir[0] * dir[1], yz = dir[1] * dir[2], xz = dir[0] * dir[2];
    switch (basis_dim) {
        case 25:
            out[16] = C4[0] * xy * (xx - yy);
            out[17] = C4[1] * yz * (3 * xx - yy);
            out[18] = C4[2] * xy * (7 * zz - 1.f);
            out[19] = C4[3] * yz * (7 * zz - 3.f);
            out[20] = C4[4] * (zz * (35 * zz - 30) + 3);
            out[21] = C4[5] * xz * (7 * zz - 3);
            out[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
            out[23] = C4[7] * xz * (xx - 3 * yy);
            out[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
            [[fallthrough]];
        case 16:
            out[9] = C3[0] * dir[1] * (3 * xx - yy);
            out[10] = C3[1] * xy * dir[2];
            out[11] = C3[2] * dir[1] * (4 * zz - xx - yy);
            out[12] = C3[3] * dir[2] * (2 * zz - 3 * xx - 3 * yy);
            out[13] = C3[4] * dir[0] * (4 * zz - xx - yy);
            out[14] = C3[5] * dir[2] * (xx - yy);
            out[15] = C3[6] * dir[0] * (xx - 3 * yy);
            [[fallthrough]];
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * dir[1];
            out[2] = C1 * dir[2];
            out[3] = -C1 * dir[0];
    }
}
