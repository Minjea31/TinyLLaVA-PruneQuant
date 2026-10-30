// Native signed INT4 Tensor Core GEMM. No floating-point operand GEMM.
// Compile to a cubin and launch through the CUDA Driver API, not a PyTorch C++ ABI.
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <math.h>

namespace wmma = nvcuda::wmma;
using S4 = wmma::experimental::precision::s4;
constexpr int WARPS = 4;

__device__ int read_original(const uint8_t* codes, long long index) {
    return ((codes[index / 2] >> ((index & 1) * 4)) & 15) - 8;
}

// A group smaller than a hardware K=32 tile is zero-padded independently.
// G=gcd(original K, original weight group size), so original scales are preserved.
extern "C" __global__ void repack_weight(
    const uint8_t* original, uint8_t* output, int n, int k, int group, int physical_group) {
    int padded_n = (n + 7) / 8 * 8;
    int physical_k = k / group * physical_group;
    long long size = (long long)padded_n * physical_k / 2;
    for (long long offset = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         offset < size; offset += (long long)gridDim.x * blockDim.x) {
        int row = offset / (physical_k / 2);
        int col = (offset % (physical_k / 2)) * 2;
        int result = 0;
        #pragma unroll
        for (int lane = 0; lane < 2; ++lane) {
            int physical = col + lane;
            int within = physical % physical_group;
            int logical = physical / physical_group * group + within;
            int q = row < n && within < group ? read_original(original, (long long)row * k + logical) : 0;
            result |= (q & 15) << (lane * 4);
        }
        output[offset] = result;
    }
}

extern "C" __global__ void quantize_activation(
    const half* input, uint8_t* output, half* scales,
    int m, int k, int group, int physical_group) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * WARPS + threadIdx.x / 32;
    int padded_m = (m + 7) / 8 * 8;
    if (row >= padded_m) return;
    int groups = k / group;
    int physical_k = groups * physical_group;
    for (int g = 0; g < groups; ++g) {
        float maximum = 0.f;
        bool invalid = false;
        if (row < m) {
            for (int within = lane; within < group; within += 32) {
                float value = __half2float(input[(long long)row * k + g * group + within]);
                invalid |= !isfinite(value);
                maximum = fmaxf(maximum, fabsf(value));
            }
        }
        unsigned invalid_mask = __ballot_sync(0xffffffff, invalid);
        for (int delta = 16; delta; delta /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
        half stored_scale = __float2half_rn(maximum == 0.f ? 1.f : fmaxf(maximum / 7.f, 0x1p-24f));
        if (invalid_mask) stored_scale = __float2half_rn(nanf(""));
        float scale = __half2float(stored_scale);
        if (lane == 0) scales[(long long)row * groups + g] = stored_scale;
        for (int byte = lane; byte < physical_group / 2; byte += 32) {
            int result = 0;
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                int within = byte * 2 + part;
                int q = 0;
                if (row < m && within < group && !invalid_mask) {
                    float value = __half2float(input[(long long)row * k + g * group + within]) / scale;
                    q = __float2int_rn(fminf(7.f, fmaxf(-7.f, value)));
                }
                result |= (q & 15) << (part * 4);
            }
            output[(long long)row * (physical_k / 2) + g * physical_group / 2 + byte] = result;
        }
    }
}

// Diagnostic path: return the exact INT32 sum for already packed signed INT4.
extern "C" __global__ void raw_int4_gemm(
    const uint8_t* a, const uint8_t* b, int* output, int m, int n, int physical_k) {
    __shared__ __align__(32) int partial[WARPS * 64];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int n_tiles = (n + 7) / 8;
    int tile = blockIdx.x * WARPS + warp;
    if (tile >= (m + 7) / 8 * n_tiles) return;
    int row = tile / n_tiles * 8;
    int col = tile % n_tiles * 8;
    wmma::fragment<wmma::matrix_a, 8, 8, 32, S4, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, S4, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> cf;
    wmma::fill_fragment(cf, 0);
    for (int start = 0; start < physical_k; start += 32) {
        wmma::load_matrix_sync(af, reinterpret_cast<const int*>(a + ((long long)row * physical_k + start) / 2), physical_k);
        wmma::load_matrix_sync(bf, reinterpret_cast<const int*>(b + ((long long)col * physical_k + start) / 2), physical_k);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(partial + warp * 64, cf, 8, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 64; i += 32) {
        int r = row + i / 8, c = col + i % 8;
        if (r < m && c < n) output[(long long)r * n + c] = partial[warp * 64 + i];
    }
}

// Each K-group is accumulated in INT32, then its original FP16 weight scale
// and activation scales are applied in FP32. There is one scale per row/group.
extern "C" __global__ void scaled_int4_gemm(
    const uint8_t* a, const uint8_t* b, const half* a_scales,
    const half* w_scales, const half* bias, half* output,
    int m, int n, int k, int weight_group, int group, int physical_group) {
    __shared__ __align__(32) int partial[WARPS * 64];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int n_tiles = (n + 7) / 8;
    int tile = blockIdx.x * WARPS + warp;
    if (tile >= (m + 7) / 8 * n_tiles) return;
    int row = tile / n_tiles * 8;
    int col = tile % n_tiles * 8;
    int groups = k / group;
    int physical_k = groups * physical_group;
    float sums[2] = {0.f, 0.f};
    wmma::fragment<wmma::matrix_a, 8, 8, 32, S4, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, S4, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> cf;
    for (int g = 0; g < groups; ++g) {
        wmma::fill_fragment(cf, 0);
        for (int start = 0; start < physical_group; start += 32) {
            int kk = g * physical_group + start;
            wmma::load_matrix_sync(af, reinterpret_cast<const int*>(a + ((long long)row * physical_k + kk) / 2), physical_k);
            wmma::load_matrix_sync(bf, reinterpret_cast<const int*>(b + ((long long)col * physical_k + kk) / 2), physical_k);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(partial + warp * 64, cf, 8, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int item = 0; item < 2; ++item) {
            int index = lane + item * 32;
            int c = col + index % 8;
            float scale = c < n ? __half2float(w_scales[((long long)c * k + g * group) / weight_group]) : 0.f;
            int r = row + index / 8;
            float activation_scale = r < m ? __half2float(a_scales[(long long)r * groups + g]) : 0.f;
            sums[item] += (float)partial[warp * 64 + index] * scale * activation_scale;
        }
        __syncwarp();
    }
    #pragma unroll
    for (int item = 0; item < 2; ++item) {
        int index = lane + item * 32;
        int r = row + index / 8, c = col + index % 8;
        if (r < m && c < n) {
            float value = sums[item];
            if (bias) value += __half2float(bias[c]);
            output[(long long)r * n + c] = __float2half_rn(value);
        }
    }
}
