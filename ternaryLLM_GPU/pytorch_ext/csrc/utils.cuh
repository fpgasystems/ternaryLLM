// Author: fuguan@ethz.ch
// Copyrights reserved
#pragma once

#include <stdint.h>
#include <vector_types.h>

/**
 * macro for dispatching
*/

#define DISPATCH_BOOL(condition, kCondition, ...)\
    do {                                        \
        if (condition) {                        \
            constexpr bool kCondition = true;   \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else {                                \
            constexpr bool kCondition = false;  \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        }                                       \
    } while(0)


// TODO: The logic to dispatch M can be fine-tuned for better performance
#define DISPATCH_M(row, vtype, tile_m, ...)     \
    do {                                        \
        if (row % 8 == 0) {                     \
            constexpr int tile_m = 8;           \
            using vtype = float4;               \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else if (row % 4 == 0) {              \
            constexpr int tile_m = 4;           \
            using vtype = float4;               \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else if (row % 2 == 0) {              \
            constexpr int tile_m = 2;           \
            using vtype = float2;               \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else {                                \
            constexpr int tile_m = 1;           \
            using vtype = float;                \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        }                                       \
    } while(0)

// TODO: inner has to be multiple of 64, add assert check here or upper
#define DISPATCH_K(inner, tile_k, fragment_size, ...)\
    do {                                        \
        if (inner <= 512) {                     \
            constexpr int tile_k = 64;          \
            constexpr int fragment_size = 32;   \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else if (inner <= 2048) {             \
            constexpr int tile_k = 256;         \
            constexpr int fragment_size = 32;   \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        } else {                                \
            constexpr int tile_k = 512;         \
            constexpr int fragment_size = 64;   \
            {                                   \
                __VA_ARGS__                     \
            }                                   \
        }                                       \
    } while(0)


// DISPATCH_BOOL(
    //     uniformed,
    //     kUniformed,
    //     DISPATCH_BOOL(
    //         padded,
    //         kPadded,

    //         /* tile size dispatch logic
    //           todo: its quite dumb */

    //         if (inners <= 512) {
    //             if (rows <= 128)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M2, TILE_WIDTH_N64, TILE_WIDTH_K64, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else if (rows <= 512)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M4, TILE_WIDTH_N64, TILE_WIDTH_K64, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M8, TILE_WIDTH_N64, TILE_WIDTH_K64, 32, kUniformed, kPadded, float4, float, int16_t>;
    //         }

    //         else if (inners <= 1024) {
    //             if (rows <= 128)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M2, TILE_WIDTH_N256, TILE_WIDTH_K256, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else if (rows <= 512)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M4, TILE_WIDTH_N256, TILE_WIDTH_K256, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else 
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M8, TILE_WIDTH_N256, TILE_WIDTH_K256, 32, kUniformed, kPadded, float4, float, int16_t>;
    //         }

    //         else if (inners <= 2048) {
    //             if (rows <= 128)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M4, TILE_WIDTH_N256, TILE_WIDTH_K256, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else if (rows <= 512)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M8, TILE_WIDTH_N256, TILE_WIDTH_K256, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else 
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M8, TILE_WIDTH_N256, TILE_WIDTH_K256, 32, kUniformed, kPadded, float4, float, int16_t>;
    //         }

    //         else if (inners <= 4096) {
    //             if (rows <= 128)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K512, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else if (rows <= 512)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M8, TILE_WIDTH_N512, TILE_WIDTH_K512, 32, kUniformed, kPadded, float4, float, int16_t>;
    //             else 
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M16, TILE_WIDTH_N512, TILE_WIDTH_K512, 32, kUniformed, kPadded, float4, float, int16_t>;
    //         }

    //         else if (inners <= 8192) {
    //             if (rows <= 128)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M4, TILE_WIDTH_N512, TILE_WIDTH_K512, 64, kUniformed, kPadded, float4, float, int16_t>;
    //             else if (rows <= 512)
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M8, TILE_WIDTH_N512, TILE_WIDTH_K512, 64, kUniformed, kPadded, float4, float, int16_t>;
    //             else 
    //                 return ter_spmm_kernel_caller<TILE_WIDTH_M16, TILE_WIDTH_N512, TILE_WIDTH_K512, 64, kUniformed, kPadded, float4, float, int16_t>;
    //         }
    //     );  // dispatch padded
    // );  // dispatch uniformed


/**
 * Self-defined vector
*/

struct __align__(8) uchar8 {
    unsigned char x, y, z, w, a, b ,c, d;
};

struct __align__(16) short8 {
    short x, y, z, w, a, b ,c, d;
};

template <typename EleType, uint32_t EleNumPerVec>
struct ToVec {};

template <>
struct ToVec<uint8_t, 4> {
    using vtype = uchar4;
};

template <>
struct ToVec<uint8_t, 8> {
    using vtype = uchar8;
};

template <>
struct ToVec<int16_t, 4> {
    using vtype = short4;
};

template <>
struct ToVec<int16_t, 8> {
    using vtype = short8;
};

template <>
struct ToVec<int32_t, 4> {
    using vtype = int4;
};

template<typename EleType, uint32_t EleNumPerVec>
struct CustomIdxVec
{
    static constexpr int BytesPerVec = EleNumPerVec * sizeof(EleType);
    using VecType = typename ToVec<EleType, EleNumPerVec>::vtype;

    using DataType = union {
        VecType vec;
        EleType ele[EleNumPerVec];
    };

    DataType data;

    inline __device__ EleType ReadAsScalar(int offset) const {
        return this->data.ele[offset];
    }

    inline __device__ void LoadByVec(const void* ptr, int offset) {
        this->data.vec = static_cast<const VecType*>(ptr)[offset];
    }
};


/**
 * Helper device function
*/

__device__ __forceinline__ void StoreFPVectorToArray(float& vec, float* dst, int rows) {
    dst[0] = vec;
}

__device__ __forceinline__ void StoreFPVectorToArray(float2& vec, float* dst, int rows) {
    dst[0] = vec.x;
    dst[rows] = vec.y;
}

__device__ __forceinline__ void StoreFPVectorToArray(float4& vec, float* dst, int rows) {
    dst[0] = vec.x;
    dst[rows] = vec.y;
    dst[rows*2] = vec.z;
    dst[rows*3] = vec.w;
}

__device__ __forceinline__ float SumFPVector(float4& vec) {
    return 0;
}

#define FULL_WARP_MASK 0xFFFFFFFF
template <class T>
/**
 *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
 *  the value of the val variable from the thread at lane X+offset of the same warp.
 *  The data exchange is performed between registers, and more efficient than going
 *  through shared memory, which requires a load, a store and an extra register to
 *  hold the address.
 */
__device__ void warp_reduce_sum(T& val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
}
