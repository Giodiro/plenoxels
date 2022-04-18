/*
 * Copyright 2021 PlenOctree Authors
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace {
namespace device {

template <typename scalar_t>
__device__ __inline__ void clamp_coord(scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        q[i] = max(scalar_t(0.0), min(scalar_t(1.0) - 1e-6, q[i]));
    }
}

template <typename scalar_t>
__device__ __inline__ void transform_coord(scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ offset,
                                           const scalar_t* __restrict__ scaling) {
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + scaling[i] * q[i];
    }
}

template <uint32_t N>
__device__ __inline__ int64_t pack_idx(int32_t i, int32_t u, int32_t v, int32_t w)
{
    int64_t out = int64_t(i) * (N * N * N) +
            u * N * N +
            v * N +
            w;
    return out; 
}

template <uint32_t N>
__device__ __inline__ void unpack_idx(int64_t idx, int32_t *i, int32_t *u, int32_t *v, int32_t *w)
{
    *w = idx % N;
    idx = idx / N;
    *v = idx % N;
    idx = idx / N;
    *u = idx % N;
    *i = idx / N;
}

template <typename scalar_t>
__device__ __inline__ void trilinear_weights(const scalar_t* __restrict__ dist_o,
                                             scalar_t* __restrict__ weights_out)
{
    weights_out[0] = (1 - dist_o[0]) * (1 - dist_o[1]) * (1 - dist_o[2]);
    weights_out[1] = (1 - dist_o[0]) * (1 - dist_o[1]) * dist_o[2]      ;
    weights_out[2] = (1 - dist_o[0]) * dist_o[1]       * (1 - dist_o[2]);
    weights_out[3] = (1 - dist_o[0]) * dist_o[1]       * dist_o[2]      ;
    weights_out[4] = dist_o[0]       * (1 - dist_o[1]) * (1 - dist_o[2]);
    weights_out[5] = dist_o[0]       * (1 - dist_o[1]) * dist_o[2]      ;
    weights_out[6] = dist_o[0]       * dist_o[1]       * (1 - dist_o[2]);
    weights_out[7] = dist_o[0]       * dist_o[1]       * dist_o[2]      ;
}

template <typename scalar_t>
__device__ __inline__ void traverse_tree_level(
            scalar_t* __restrict__ xyz_inout,
            const scalar_t N,
            int32_t* __restrict__ u_out,
            int32_t* __restrict__ v_out,
            int32_t* __restrict__ w_out)
{
    xyz_inout[0] *= N;
    xyz_inout[1] *= N;
    xyz_inout[2] *= N;
    *u_out = floor(xyz_inout[0]);
    *v_out = floor(xyz_inout[1]);
    *w_out = floor(xyz_inout[2]);
    xyz_inout[0] -= *u_out;
    xyz_inout[1] -= *v_out;
    xyz_inout[2] -= *w_out;
}


template <typename scalar_t>
__device__ __inline__ scalar_t* query_single_from_root(
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    scalar_t* __restrict__ xyz_inout,
    scalar_t* __restrict__ cube_sz_out,
    int64_t* __restrict__ node_id_out=nullptr)
{
    const scalar_t N = child.size(1);
    clamp_coord<scalar_t>(xyz_inout);

    int32_t node_id = 0;
    int32_t u, v, w;
    *cube_sz_out = N;
    while (true) {
        xyz_inout[0] *= N;
        xyz_inout[1] *= N;
        xyz_inout[2] *= N;
        u = floor(xyz_inout[0]);
        v = floor(xyz_inout[1]);
        w = floor(xyz_inout[2]);
        xyz_inout[0] -= u;
        xyz_inout[1] -= v;
        xyz_inout[2] -= w;

        const int32_t skip = child[node_id][u][v][w];
        if (skip == 0) {
            if (node_id_out != nullptr) {
                *node_id_out = node_id * int64_t(N * N * N) +
                               u * int32_t(N * N) + v * int32_t(N) + w;
            }
            return &data[node_id][u][v][w][0];
        }
        *cube_sz_out *= N;
        node_id += skip;
    }
    return nullptr;
}


template <typename scalar_t>
__device__ __inline__ void query_node_info_from_root(
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    scalar_t* __restrict__ xyz_inout,
    scalar_t* __restrict__ cube_sz_out,
    int64_t* __restrict__ node_id_out)
{
    const scalar_t N = child.size(1);
    clamp_coord<scalar_t>(xyz_inout);

    int32_t node_id = 0;
    int32_t u, v, w;
    *cube_sz_out = N;
    while (true) {
        xyz_inout[0] *= N;
        xyz_inout[1] *= N;
        xyz_inout[2] *= N;
        u = floor(xyz_inout[0]);
        v = floor(xyz_inout[1]);
        w = floor(xyz_inout[2]);
        xyz_inout[0] -= u;
        xyz_inout[1] -= v;
        xyz_inout[2] -= w;

        const int32_t skip = child[node_id][u][v][w];
        if (skip == 0) {
            *node_id_out = node_id * int64_t(N * N * N) + u * int32_t(N * N) + v * int32_t(N) + w;
            return;
        }
        *cube_sz_out *= N;
        node_id += skip;
    }
}


template <typename scalar_t, int K>
__device__ __inline__ void query_interp_from_root(
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
    scalar_t* __restrict__ neighbor_data_buf,
    scalar_t* __restrict__ xyz_inout,
    scalar_t* __restrict__ iweights_out,
    int64_t* __restrict__ neighbor_ids_out,
    scalar_t* __restrict__ interp_out)
{
    const scalar_t N = child.size(1);
    scalar_t *neighbor_data;
    clamp_coord<scalar_t>(xyz_inout);

    int32_t node_id = 0;
    int32_t u, v, w, pu, pv, pw;
    scalar_t dist_o[3] = {0.25, 0.25, 0.25};
    scalar_t parent_sum[K] = {};

    // initialize neighbor_data_buf as zeros (since no data in root of tree).
    for (int i = 0; i < 8; ++i) {
        for (int data_idx = 0; data_idx < K; ++data_idx) {
            neighbor_data_buf[i * K + data_idx] = 0.0;
        }
    }
    //printf("input coords: %f, %f, %f\n", xyz_inout[0], xyz_inout[1], xyz_inout[2]);
    // First iteration, to initialize parent coordinates
    traverse_tree_level(xyz_inout, N, &pu, &pv, &pw);
    //printf("coords after root: %f, %f, %f\n", xyz_inout[0], xyz_inout[1], xyz_inout[2]);

    while (true) {
        traverse_tree_level(xyz_inout, N, &u, &v, &w);
        //printf("coords: %f, %f, %f\n", xyz_inout[0], xyz_inout[1], xyz_inout[2]);
        //printf("pt = %d,%d,%d - pt+1 = %d,%d,%d\n", pu, pv, pw, u, v, w);
        // i, j, k index neighbors
        #pragma unroll 2
        for (int i = 0; i < 2; ++i) {
            #pragma unroll 2
            for (int j = 0; j < 2; ++j) {
                #pragma unroll 2
                for (int k = 0; k < 2; ++k) {
                    if ((pu + u + i - 1 == 0 || pu + u + i - 1 == 1) &&
                        (pv + v + j - 1 == 0 || pv + v + j - 1 == 1) &&
                        (pw + w + k - 1 == 0 || pw + w + k - 1 == 1)) {
                        neighbor_data = &data[node_id][pu + u + i - 1][pv + v + j - 1][pw + w + k - 1][0];
                        if (neighbor_ids_out != nullptr)
                        {
                            neighbor_ids_out[(i << 2) + (j << 1) + k] = pack_idx<2>(node_id, pu + u + i - 1, pv + v + j - 1, pw + w + k - 1);
                        }
                        //printf("Found valid %d,%d,%d neighbor at node [%d][%d][%d][%d] with value [%f, %f, %f] \n", i, j, k, node_id, pu + u + i - 1, pv + v + j - 1, pw + w + k - 1, neighbor_data[0], neighbor_data[1], neighbor_data[2]);
                        for (int data_idx = 0; data_idx < K; ++data_idx) {
                            neighbor_data_buf[((i << 2) + (j << 1) + k) * K + data_idx] = parent_sum[data_idx] + neighbor_data[data_idx];
                        }
                        // TODO: It's not clear why the next two blocks work.
                        if (i == 0 && j == 0 && k == 0) {
                            dist_o[0] = (xyz_inout[0] + u) / N;
                            dist_o[0] = dist_o[0] < 0.5 ? dist_o[0] + 0.5 : dist_o[0] - 0.5;
                            dist_o[1] = (xyz_inout[1] + v) / N;
                            dist_o[1] = dist_o[1] < 0.5 ? dist_o[1] + 0.5 : dist_o[1] - 0.5;
                            dist_o[2] = (xyz_inout[2] + w) / N;
                            dist_o[2] = dist_o[2] < 0.5 ? dist_o[2] + 0.5 : dist_o[2] - 0.5;
                            //printf("Distance: %f, %f, %f\n", dist_o[0], dist_o[1], dist_o[2]);
                        } else if (i == 1 && j == 1 && k == 1) {
                            dist_o[0] = (xyz_inout[0] + u) / N;
                            dist_o[0] = dist_o[0] < 0.5 ? dist_o[0] + 0.5 : dist_o[0] - 0.5;
                            dist_o[1] = (xyz_inout[1] + v) / N;
                            dist_o[1] = dist_o[1] < 0.5 ? dist_o[1] + 0.5 : dist_o[1] - 0.5;
                            dist_o[2] = (xyz_inout[2] + w) / N;
                            dist_o[2] = dist_o[2] < 0.5 ? dist_o[2] + 0.5 : dist_o[2] - 0.5;
                            //printf("Distance: %f, %f, %f\n", dist_o[0], dist_o[1], dist_o[2]);
                        }
                    }
                }
            }
        }
        for (int data_idx = 0; data_idx < K; ++data_idx) {
            parent_sum[data_idx] += data[node_id][pu][pv][pw][data_idx];
        }
        const int32_t skip = child[node_id][pu][pv][pw];
        //printf("At node %d - child %d, %d, %d - skip %d\n", node_id, pu, pv, pw, skip);
        if (skip == 0) {
            trilinear_weights(dist_o, iweights_out);
            for (int data_idx = 0; data_idx < K; ++data_idx) {
                interp_out[data_idx] = 0;
                for (int i = 0; i < 8; ++i) {
                    interp_out[data_idx] += iweights_out[i] * neighbor_data_buf[i * K + data_idx];
                }
            }
            return;
        }
        node_id += skip;
        pu = u;
        pv = v;
        pw = w;
    }
}


template <typename scalar_t, int K>
__device__ __inline__ void query_interp_from_root_bwd(
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> parent_depth,
    const scalar_t* __restrict__ weights,       // Interpolation weights [8]
    const int64_t* __restrict__ neighbor_ids,  // Packed ids of the neighbors [8]
    const scalar_t* __restrict__ grad_output)   // [K]
{
    int32_t i, u, v, w;
    int32_t packed_node_idx;
    for (uint32_t neigh_idx = 0; neigh_idx < 8; ++neigh_idx) {
        packed_node_idx = (int32_t)neighbor_ids[neigh_idx];
        while (packed_node_idx > 0) {  // Loop going up through the tree.
            unpack_idx<2>(packed_node_idx, &i, &u, &v, &w);
            for (uint32_t data_idx = 0; data_idx < K; ++data_idx) {
                atomicAdd(&grad[i][u][v][w][data_idx], grad_output[data_idx] * weights[neigh_idx]);
            }
            packed_node_idx = parent_depth[packed_node_idx][0];
        }
    }
}


}  // namespace device
}  // namespace

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

namespace {
// Get approx number of CUDA cores
__host__ int get_sp_cores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            break;
        default:
            break;
    }
    return cores;
}
}  // namespace


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ inline void atomicMax(float* result, float value){
    unsigned* result_as_u = (unsigned*)result;
    unsigned old = *result_as_u, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_u, assumed,
                __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (old != assumed);
    return;
}

__device__ inline void atomicMax(double* result, double value){
    unsigned long long int* result_as_ull = (unsigned long long int*)result;
    unsigned long long int old = *result_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(result_as_ull, assumed,
                __double_as_longlong(fmaxf(value, __longlong_as_double(assumed))));
    } while (old != assumed);
    return;
}
