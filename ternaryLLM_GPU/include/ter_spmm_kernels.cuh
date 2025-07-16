#pragma once

#include <cuda.h>
#include <assert.h>
#include <cusparse.h>
#include <cublas_v2.h>
// #include <cusparseLt.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "devutil.cuh"


#define TILE_WIDTH_K8   8
#define TILE_WIDTH_N2   TILE_WIDTH_K8/4
#define TILE_WIDTH_K32  32
#define TILE_WIDTH_N8   TILE_WIDTH_K32/4
#define TILE_WIDTH_K64  64
#define TILE_WIDTH_N16  TILE_WIDTH_K64/4
#define TILE_WIDTH_K128 128
#define TILE_WIDTH_N32  TILE_WIDTH_K128/4
#define TILE_WIDTH_K256 256
#define TILE_WIDTH_N64  TILE_WIDTH_K256/4
#define TILE_WIDTH_K512 512
#define TILE_WIDTH_N128  TILE_WIDTH_K512/4
#define TILE_WIDTH_K1024 1024
#define TILE_WIDTH_K2048 2048
#define TILE_WIDTH_N256  256
#define TILE_WIDTH_N512  512
#define TILE_WIDTH_N1024  1024

#define TILE_WIDTH_M1   1
#define TILE_WIDTH_M2   2
#define TILE_WIDTH_M4   4
#define TILE_WIDTH_M8   8
#define TILE_WIDTH_M16  16
#define TILE_WIDTH_M32  32
#define TILE_WIDTH_M64  64

#define CUDA_CALL_CHECK(func_call)              \
    do {                                        \
        const cudaError_t cudaerr = func_call;  \
        assert(cudaerr == cudaSuccess && __FILE__ && __LINE__);         \
    } while(0)

#define CUSPARSE_CALL_CHECK(func_call)                                                   \
{                                                                              \
    cusparseStatus_t status = (func_call);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        assert(false && __FILE__ && __LINE__);                \
    }                                                                          \
}

#define CUBLAS_CALL_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

/**
 * straight forward implentation of sparse ternary gemm
 * X is in column major 
 */
__global__ void ter_spmm_float_baseline(float* X, int8_t* W1, float* result, int rows, int columns, int inners);

/**
 * baseline optimized implentation of sparse ternary gemm in csc format
 * X is in column major
 */
__global__ void ter_csc_spmm_float_baseline(float* X, 
                                            int32_t* w_neg_row_indice, int32_t* w_neg_col_offset, 
                                            int32_t* w_pos_row_indice, int32_t* w_pos_col_offset, 
                                            float* result, 
                                            int rows, int columns, int inners);

/**
 * baseline optimized implentation of sparse ternary gemm in merged csc format
 */
__global__ void ter_mcsc_spmm_float_baseline(const float* __restrict__ X, 
                                            const int32_t* __restrict__ w_merged_row_indice, const int32_t* __restrict__ w_merged_col_offset, 
                                            float* result, 
                                            int rows, int columns, int inners);



/**
 * baseline optimized implentation of sparse ternary gemm in csc format
 */
template<int TILE_M, int TILE_N, int TILE_K, int BURST_SIZE, typename BURST_RES_F> 
__global__ void ter_csc_spmm_float_opt(
    const float* __restrict__ X, 
    const int32_t* __restrict__ w_neg_row_indice, const int32_t* __restrict__ w_neg_col_offset, 
    const int32_t* __restrict__ w_pos_row_indice, const int32_t* __restrict__ w_pos_col_offset, 
    float* result, 
    int rows, int columns, int inners)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= columns || row >= rows)
        return;

    __shared__ float tile_X[TILE_M*TILE_K];
    float res[TILE_M] = {0};
    const int tileNumK = (int)ceil(1.*inners/TILE_K);

    const int2* w_neg_col_offset_int2 = reinterpret_cast<const int2*>(w_neg_col_offset + col*tileNumK*2);
    const int2* w_pos_col_offset_int2 = reinterpret_cast<const int2*>(w_pos_col_offset + col*tileNumK*2);
    const BURST_RES_F* res_burst = reinterpret_cast<const BURST_RES_F*>(result);
    // const F* X_burst = reinterpret_cast<const F*>(X);
    // F* tile_X_burst = reinterpret_cast<F*>(tile_X);

    int row_offset = row*TILE_M;
#pragma unroll
    for (int tileIdK = 0; tileIdK < tileNumK; tileIdK++) {
        /* load into smem */
        // int offset = tileIdK*TILE_K;
        int offset_tx = tileIdK*TILE_K + threadIdx.x;   // single float load offset
#pragma unroll
        for (int j = 0; j < TILE_M; j++) {
            /* burst load: burst size = (TILE_K/TILE_N) */
            // tile_X_burst[j*TILE_K/(TILE_K/TILE_N) + threadIdx.x] = X_burst[row_offset*(inners/(TILE_K/TILE_N)) + offset/(TILE_K/TILE_N)];

            /* single load */
#pragma unroll
            for (int i = 0; i < BURST_SIZE; i++) {
                /* thread loads consecutive element */
                // tile_X[j*TILE_K + threadIdx.x*(TILE_K/TILE_N) + i] = X[rows*(offset + threadIdx.x*(TILE_K/TILE_N) + i) + row*TILE_M + j];

                /* thread loads inconsecutive element */
                float x_data = __ldca(X + rows*(offset_tx + i*TILE_N) + row_offset + j);
                tile_X[j*TILE_K + threadIdx.x + i*TILE_N] = x_data;
            }
        } 
        
        __syncthreads();
        
        int2 tile_neg_range = __ldca(w_neg_col_offset_int2+tileIdK);  // (x, y)=(start, end)
        int2 tile_pos_range = __ldca(w_pos_col_offset_int2+tileIdK);  // (x, y)=(start, end)
        int neg = tile_neg_range.y - tile_neg_range.x;
        int pos = tile_pos_range.y - tile_pos_range.x;
        int min_neg_pos = neg < pos ? neg : pos;

#pragma unroll
        for (int j = 0; j < TILE_M; j++) {
            int tile_offset = j*TILE_K;
            for (int k = 0; k < min_neg_pos; k++) {
                int neg_id = __ldca(w_neg_row_indice + tile_neg_range.x + k);
                int pos_id = __ldca(w_pos_row_indice + tile_pos_range.x + k);
                res[j] -= tile_X[tile_offset + neg_id];
                res[j] += tile_X[tile_offset + pos_id];
            }
            for (int k = min_neg_pos; k < pos; k++) {
                int pos_id = __ldca(w_pos_row_indice + tile_pos_range.x + k);
                res[j] += tile_X[tile_offset + pos_id];
            }
            for (int k = min_neg_pos; k < neg; k++) {
                int neg_id = __ldca(w_neg_row_indice + tile_neg_range.x + k);
                res[j] -= tile_X[tile_offset + neg_id];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < TILE_M; j++)
        result[(row*TILE_M + j) * columns + col] = res[j];
    // result[tx * rows + ty] = res;
}


/**
 * baseline optimized implentation of sparse ternary gemm in csc format
 */
template<int TILE_M, int TILE_N, int TILE_K, typename F> 
__global__ void ter_csc_spmm_float_opt_async(
    const float* __restrict__ X, 
    const int32_t* __restrict__ w_neg_row_indice, const int32_t* __restrict__ w_neg_col_offset, 
    const int32_t* __restrict__ w_pos_row_indice, const int32_t* __restrict__ w_pos_col_offset, 
    float* result, 
    int rows, int columns, int inners)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= columns || row >= rows)
        return;

    /* double buffer in smem */
    __shared__ float tile_X[2][TILE_M*TILE_K];

    /* number of tiles in K dim */
    int tileNumK = (int)ceil(1.*inners/TILE_K);

    /* double buffer ptr */
    int buf = 0;

    /* cast pointers */
    const int2* w_neg_col_offset_int2 = reinterpret_cast<const int2 *>(w_neg_col_offset + col*tileNumK*2);  // int2 pointer for neg col offset
    const int2* w_pos_col_offset_int2 = reinterpret_cast<const int2 *>(w_pos_col_offset + col*tileNumK*2);  // int2 pointer for pos col offset
    const float* data_in_X = (const float*)X;
    // size_t tile_X_smem = __cvta_generic_to_shared(tile_X);  // smem pointer for async mem operation
    // const F* X_float4 = reinterpret_cast<const F*>(X);
    // F* tile_X_float4 = reinterpret_cast<F*>(tile_X);

    float res[TILE_M] = {0};

    /* offsets */
    int offset = 0;
    int offset_tx = offset + threadIdx.x;
    int row_offset = row*TILE_M;

    /* prefetch the first tile into first buffer */
#pragma unroll
    for (int j = 0; j < TILE_M; j++) {
#pragma unroll
        for (int i = 0; i < TILE_K/TILE_N; i++) {
            /* thread loads inconsecutive element */
            float x_data = __ldca(X + rows*(offset_tx + i*TILE_N) + row_offset + j);
            tile_X[buf][j*TILE_K + threadIdx.x + i*TILE_N] = x_data;
        }
    }
    __syncthreads();

    /* async calculations */
    int tileIdK = 0;    // tile id
    do {
        tileIdK++;

        /* check to trigger async copy */
        buf ^= 1;   // switch buffer
        offset = tileIdK*TILE_K;
        offset_tx = offset + threadIdx.x;

        if (tileIdK < tileNumK) {
#pragma unroll
        for (int j = 0; j < TILE_M; j++) {
#pragma unroll
            for (int i = 0; i < TILE_K/TILE_N; i++) {
                /* thread loads inconsecutive element */
                float x_data = __ldca(X + rows*(offset_tx + i*TILE_N) + row_offset + j);
                tile_X[buf][j*TILE_K + threadIdx.x + i*TILE_N] = x_data;

                // size_t tile_X_smem = __cvta_generic_to_shared(tile_X);  // smem pointer for async mem operation
                // asm volatile(
                //     "cp.async.ca.shared.global [%0], [%1], %2;\n"
                //     :
                //     : "l"(tile_X_smem + buf*TILE_M*TILE_K + j*TILE_K + threadIdx.x + i*TILE_N),   // Destination in shared memory
                //     "l"(data_in_X + rows*(offset_tx + i*TILE_N) + row_offset + j),   // Source in global memory
                //     "n"(4)      // Number of bytes to copy (32bit)
                // );
            }
        } 
        
        /* commit into group */
        // asm volatile("cp.async.commit_group;\n");
        }

        /* start tile calculation of prefetched tile */
        buf ^= 1;   // switch buffer
        int2 tile_neg_range = __ldca(w_neg_col_offset_int2+tileIdK-1);  // (x, y)=(start, end)
        int2 tile_pos_range = __ldca(w_pos_col_offset_int2+tileIdK-1);  // (x, y)=(start, end)
        int neg = tile_neg_range.y - tile_neg_range.x;
        int pos = tile_pos_range.y - tile_pos_range.x;
        int min_neg_pos = neg < pos ? neg : pos;
#pragma unroll
        for (int j = 0; j < TILE_M; j++) {

            for (int k = 0; k < min_neg_pos; k++) {
                int neg_id = __ldca(w_neg_row_indice + tile_neg_range.x + k);
                int pos_id = __ldca(w_pos_row_indice + tile_pos_range.x + k);
                res[j] -= tile_X[buf][j*TILE_K + neg_id];
                res[j] += tile_X[buf][j*TILE_K + pos_id];
            }
            for (int k = min_neg_pos; k < pos; k++) {
                int pos_id = __ldca(w_pos_row_indice + tile_pos_range.x + k);
                res[j] += tile_X[buf][j*TILE_K + pos_id];
            }
            for (int k = min_neg_pos; k < neg; k++) {
                int neg_id = __ldca(w_neg_row_indice + tile_neg_range.x + k);
                res[j] -= tile_X[buf][j*TILE_K + neg_id];
            }
        }

        /* wait for async copy to complete */
        buf ^= 1;
        // asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    } while (tileIdK < tileNumK);

#pragma unroll
    for (int j = 0; j < TILE_M; j++)
        result[(row*TILE_M + j) * columns + col] = res[j];
    // result[tx * rows + ty] = res;
}



template<int TILE_M, int TILE_N, int TILE_K, int FRAGMENT_SIZE=32, 
    typename VEC_TYPE_M=float, typename DATA_TYPE=float, 
    typename IDX_VEC_TYPE=int2, typename OFFSET_VEC_TYPE=int4, typename IDX_TYPE=int32_t> 
/**
 * @brief 
 * ThreadBlkY = 1, ThreadBlkX = TILE_N
 * @tparam TILE_M: tile width in M-dim
 * @tparam TILE_N: tile width/block size in N-dim/x-dim
 * @tparam TILE_K: tile width in K-dim
 * @tparam FRAGMENT_SIZE: fragment size for row indices
 * @tparam VEC_SIZE_M: size of vector in M-dim. If scalar, VEC_SIZE_M = 1
 * @tparam VEC_TYPE_M: data_type<BURST_SIZE>
 * @tparam DATA_TYPE: activation and output data type
 * @tparam IDX_VEC_TYPE
 * @tparam OFFSET_VEC_TYPE
 * @tparam IDX_TYPE
 * @param w_merged_row_indice 
 * @param w_merged_col_offset 
 * @param w_col_tile_indices
 * @param result 
 * @param rows 
 * @param columns 
 * @param inners 
 */
__global__ void ter_tiled_mcsc_spmm(
    const DATA_TYPE* __restrict__ X, 
    const IDX_TYPE* __restrict__ w_tiled_merged_row_indice, 
    const IDX_TYPE* __restrict__ w_tiled_merged_col_offset, 
    const int32_t* __restrict__ w_col_tile_indices,
    DATA_TYPE* result, 
    int rows, int columns, int inners)
{
    constexpr int vectorSizeM = sizeof(VEC_TYPE_M)/sizeof(DATA_TYPE);   // size of vector used in m-dim of dense X, normally 4
    constexpr int vectorSizeIdx = sizeof(IDX_VEC_TYPE)/sizeof(IDX_TYPE);   // size of vector used in row indices
    constexpr int threadVecItemM = TILE_M/vectorSizeM;                  // number of vector each thread needs to load in m-dim for dense X
    constexpr int threadVecItemIdx = FRAGMENT_SIZE/vectorSizeIdx;       // number of vector each thread needs to load for row indices
    constexpr int quarterFragmentSize = FRAGMENT_SIZE/4;
    constexpr int threadVecItemQuarterIdx = quarterFragmentSize/vectorSizeIdx;       // number of vector each thread needs to load for row indices
    
    // static_assert(threadVecItemM > 0, "TILE_M has to be larger than vectorSizeM");
    // static_assert(threadVecItemIdx > 0, "FRAGMENT_SIZE has to be larger than vectorSizeIdx");
    // static_assert(quarterFragmentSize > 0, "FRAGMENT_SIZE has to be larger than 4");
    // static_assert(threadVecItemQuarterIdx > 0, "vectorSizeIdx has to be larger than 4");

    const int tileNumK = (inners + TILE_K - 1)/TILE_K;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    w_col_tile_indices += col*tileNumK;

    if (col >= columns || row >= rows)
        return;

    /* smem for dense X */
    __shared__ DATA_TYPE tile_X[TILE_M*TILE_K]; // row major smem tile

    /* fragment for sparse arrays  */
    __align__(16) IDX_TYPE fragment_row_indices[FRAGMENT_SIZE] = {0};
    IDX_VEC_TYPE* fragment_row_indices_vec = reinterpret_cast<IDX_VEC_TYPE*>(fragment_row_indices);

    /* results */
    DATA_TYPE res[TILE_M] = {0};
    DATA_TYPE partial[TILE_M] = {0};
    
    /* vectorized access to gmem */
    const OFFSET_VEC_TYPE* w_tiled_merged_col_offset_vec = reinterpret_cast<const OFFSET_VEC_TYPE*>(w_tiled_merged_col_offset + col*tileNumK*4);
    const VEC_TYPE_M* X_vec = reinterpret_cast<const VEC_TYPE_M*>(X);

    int row_offset = row*threadVecItemM;    // each thread processes TILE_M/vectorSizeM elements once in m-dim
#pragma unroll
    for (int tileCnt = 0; tileCnt < tileNumK; tileCnt++) {
        int tileIdK = w_col_tile_indices[tileCnt];      // swizzled tile to process
        /* load into smem */
        int actual_k_offset = tileIdK*TILE_K + threadIdx.x;   // load offset for smem in k-dim
#pragma unroll
        for (int m_id = 0; m_id < threadVecItemM; m_id++) {
            int tile_offset = m_id*TILE_K*vectorSizeM + threadIdx.x;   // start offset for 2-d tile
            int actual_row_id = row_offset + m_id;         // actually row to load

            /* single load */
            DATA_TYPE* tile_X_base = tile_X + tile_offset;
#pragma unroll
            for (int i = 0; i < TILE_K; i+=TILE_N) {
                /* thread loads consecutive element in column major */
                // tile_X[j*TILE_K + threadIdx.x*(TILE_K/TILE_N) + i] = X[rows*(offset + threadIdx.x*(TILE_K/TILE_N) + i) + row*TILE_M + j];

                /* thread loads inconsecutive element in column major */
                // DATA_TYPE x_data = __ldca(X + rows*(actual_k_offset + i) + actual_row_id);
                VEC_TYPE_M x_data_vec = __ldca(X_vec + rows/vectorSizeM*(actual_k_offset + i) + actual_row_id);

                /* thread loads inconsecutive element in row major */
                // DATA_TYPE x_data = __ldca(X + inners*actual_row_id + actual_k_offset + i);
                // tile_X[tile_offset + i] = x_data;

                // presume the vector is float4
                StoreFPVectorToArray(x_data_vec, tile_X_base + i, TILE_K);
            }
        } 
        
        __syncthreads();
        
        OFFSET_VEC_TYPE col_offset = __ldca(w_tiled_merged_col_offset_vec + tileIdK);
        int col_offset_interleaved_start = col_offset.x;
        int col_offset_remain_start = col_offset.y;
        int nnz_interleaved = col_offset_remain_start - col_offset_interleaved_start;
        int nnz_remain = col_offset.z - col_offset_remain_start;

        /* interleaved part computation */
        const IDX_TYPE* w_tiled_merged_row_indice_base = w_tiled_merged_row_indice + col_offset_interleaved_start;
        const IDX_VEC_TYPE* w_tiled_merged_row_indice_base_vec = reinterpret_cast<const IDX_VEC_TYPE*>(w_tiled_merged_row_indice_base);

        for (; nnz_interleaved >= FRAGMENT_SIZE; nnz_interleaved -= FRAGMENT_SIZE) {
#pragma unroll
            for (int i = 0; i < threadVecItemIdx; i++) 
                fragment_row_indices_vec[i] = __ldca(w_tiled_merged_row_indice_base_vec + i);
            w_tiled_merged_row_indice_base_vec += threadVecItemIdx;   // update global data pointer

#pragma unroll
            for (int j = 0; j < TILE_M; j++) {
                int tile_offset = j*TILE_K;
#pragma unroll
                for (int i = 0; i < FRAGMENT_SIZE; i+=2) {
                    res[j] += tile_X[tile_offset + fragment_row_indices[i]];
                    res[j] -= tile_X[tile_offset + fragment_row_indices[i+1]];
                }
            }
        }

        /* interleaved part is padded to the multiple of quarterFragmentSize, which will increase some memory overheads
            but trival */
        for (; nnz_interleaved >= quarterFragmentSize; nnz_interleaved -= quarterFragmentSize) {
#pragma unroll
            for (int i = 0; i < threadVecItemQuarterIdx; i++) 
                fragment_row_indices_vec[i] = __ldca(w_tiled_merged_row_indice_base_vec + i);
            w_tiled_merged_row_indice_base_vec += threadVecItemQuarterIdx;   // update global data pointer

#pragma unroll
            for (int j = 0; j < TILE_M; j++) {
                int tile_offset = j*TILE_K;
#pragma unroll
                for (int i = 0; i < quarterFragmentSize; i+=2) {
                    res[j] += tile_X[tile_offset + fragment_row_indices[i]];
                    res[j] -= tile_X[tile_offset + fragment_row_indices[i+1]];
                }
            }
        }
        
        /* if padding is not added, turn on residual calculation for interleaved part */
        /* interleaved residual computation */
// #pragma unroll
//         for (int i = 0; i < nnz_interleaved; i+=2) {
//             IDX_VEC_TYPE row_ids = __ldca(w_tiled_merged_row_indice_base_vec + i/2);
// #pragma unroll
//             for (int j = 0; j < TILE_M; j++) {
//                 int tile_offset = j*TILE_K;
//                 res[j] += tile_X[tile_offset + row_ids.x];
//                 res[j] -= tile_X[tile_offset + row_ids.y];
//             }
//         }
        
        /* common part computation */
        w_tiled_merged_row_indice_base = w_tiled_merged_row_indice + col_offset_remain_start;
        w_tiled_merged_row_indice_base_vec = reinterpret_cast<const IDX_VEC_TYPE*>(w_tiled_merged_row_indice_base);
        for (; nnz_remain >= FRAGMENT_SIZE; nnz_remain -= FRAGMENT_SIZE) {
#pragma unroll
            for (int i = 0; i < threadVecItemIdx; i++) 
                fragment_row_indices_vec[i] = __ldca(w_tiled_merged_row_indice_base_vec + i);
            w_tiled_merged_row_indice_base_vec += threadVecItemIdx;   // update global data pointer
#pragma unroll
            for (int m_id = 0; m_id < TILE_M; m_id++) {
                int tile_offset = m_id*TILE_K;
#pragma unroll
                for (int i = 0; i < FRAGMENT_SIZE; i+=1) {
                    partial[m_id] += tile_X[tile_offset + fragment_row_indices[i]];
                }
            }
        }

        for (; nnz_remain >= quarterFragmentSize; nnz_remain -= quarterFragmentSize) {
#pragma unroll
            for (int i = 0; i < threadVecItemQuarterIdx; i++) {
                fragment_row_indices_vec[i] = __ldca(w_tiled_merged_row_indice_base_vec + i);
            }
            w_tiled_merged_row_indice_base_vec += threadVecItemQuarterIdx;   // update global data pointer
#pragma unroll
            for (int m_id = 0; m_id < TILE_M; m_id++) {
                int tile_offset = m_id*TILE_K;
#pragma unroll
                for (int i = 0; i < quarterFragmentSize; i+=1) {
                    partial[m_id] += tile_X[tile_offset + fragment_row_indices[i]];
                }
            }
        }

        /* common residual computation */
        for (int i = 0; i < nnz_remain/threadVecItemIdx; i+=1) {
            IDX_VEC_TYPE common_ids = __ldca(w_tiled_merged_row_indice_base_vec + i);
#pragma unroll
            for (int m_id = 0; m_id < TILE_M; m_id++) {
                int tile_offset = m_id*TILE_K;
                partial[m_id] += tile_X[tile_offset + common_ids.x] + tile_X[tile_offset + common_ids.y] +
                                tile_X[tile_offset + common_ids.z] + tile_X[tile_offset + common_ids.w];
            }
        }

        /* aggregate sum of interleaved part and common part */
#pragma unroll
        for (int m_id = 0; m_id < TILE_M; m_id++) {
            res[m_id] += col_offset.w*partial[m_id];
            partial[m_id] = 0;
        }
        __syncthreads();
    }


#pragma unroll
    for (int j = 0; j < TILE_M; j++)
        result[(row*TILE_M + j) * columns + col] = res[j];
}






template<int TILE_M, int TILE_N, int TILE_K, 
    typename IDX_VEC_TYPE=int4, typename IDX_TYPE=int32_t,
    typename DATA_VEC_TYPE=float4, typename DATA_TYPE=float> 
/**
 * @brief 
 * ThreadBlkY = TILE_M, ThreadBlkX = TILE_N
 * @tparam TILE_M: tile width/block size in M-dim/y-dim
 * @tparam TILE_N: tile width in N-dim/x-dim, default 1
 * @tparam TILE_K: tile width in K-dim
 * @tparam BURST_SIZE: TILE_K/TILE_N
 * @tparam BURST_TYPE: data_type<BURST_SIZE>
 * @tparam DATA_TYPE: activation and output data type
 * @param w_merged_row_indice 
 * @param w_merged_col_offset 
 * @param result 
 * @param rows 
 * @param columns 
 * @param inners 
 */
__global__ void ter_tiled_mcsc_spmm_T(
    const DATA_TYPE* __restrict__ X, 
    const int32_t* __restrict__ w_merged_row_indice, 
    const int32_t* __restrict__ w_merged_col_offset, 
    const int32_t* __restrict__ w_swizzled_col_indices,
    DATA_TYPE* result, 
    int rows, int columns, int inners)
{
//     constexpr int idx_vec_size = sizeof(IDX_VEC_TYPE)/sizeof(IDX_TYPE);
//     constexpr int data_vec_size = sizeof(DATA_VEC_TYPE)/sizeof(DATA_TYPE);
//     constexpr int TILE_M_VEC = TILE_M/idx_vec_size; // tile size M after vectorization
//     constexpr int TILE_K_VEC = TILE_K/idx_vec_size; // tile size K after vectorization
//     constexpr int threadItemK = TILE_K/TILE_M/idx_vec_size;  // number of items to be loaded by each thread for row indices

//     const int blkx = blockIdx.x * blockDim.x;
//     const int blky = blockIdx.y * blockDim.y;
//     const int row_id = blky + threadIdx.y;
//     const int col_id = w_swizzled_col_indices[blkx + threadIdx.x];  // load swizzled col id

//     if (row_id >= rows || col_id >= columns)
//         return;

//     __shared__ int32_t tile_row_indice[TILE_N*TILE_K];

//     DATA_TYPE fragment_X[TILE_K] = {0};
//     DATA_VEC_TYPE* fragment_X_vec = reinterpret_cast<DATA_VEC_TYPE*>(fragment_X);
//     DATA_TYPE fragment_res = 0;
//     DATA_TYPE cache_sum = 0;

//     const int4* w_merged_col_offset_int4 = reinterpret_cast<const int4*>(w_merged_col_offset + col_id*4);
//     int4 col_offset = __ldca(w_merged_col_offset_int4);
//     int col_offset_start = col_offset.x;
//     int col_offset_mid = col_offset.y;
//     int nnz_interleaved = col_offset_mid - col_offset_start;
//     int nnz_remain = col_offset.z - col_offset_mid;

//     /* prepare smem pointers */
//     int32_t tile_base_index = threadIdx.x*TILE_K + threadIdx.y*idx_vec_size;    // scalar index in tile to store
//     int32_t* tile_row_indice_base = tile_row_indice + tile_base_index;          // scalar pointer for tile to store
//     IDX_VEC_TYPE* tile_row_indice_vec_base = reinterpret_cast<IDX_VEC_TYPE*>(tile_row_indice_base);    // vector pointer for tile to store
//     int32_t* tile_row_indice_ld = tile_row_indice + threadIdx.x*TILE_K;          // scalar pointer for tile to load
//     IDX_VEC_TYPE* tile_row_indice_vec_ld = reinterpret_cast<IDX_VEC_TYPE*>(tile_row_indice_ld);    // vector pointer for tile to store
    
//     /* prepare row indices pointers */
//     int32_t row_indices_base_index = col_offset_start + threadIdx.y*idx_vec_size;
//     // const int32_t* row_indices_base = w_merged_row_indice + row_indices_base_index; // scalar pointer for row indices
//     const IDX_VEC_TYPE* row_indices_vec_base = nullptr; // vector pointer for row indices
//     MemoryAligner<IDX_VEC_TYPE, IDX_TYPE, TILE_M> aligner_interleaved(row_indices_base_index, nnz_interleaved);
    
//     /* prepare dense X pointers */
//     const DATA_TYPE* X_base = X + row_id;   // scalar pointer for dense X
//     // const DATA_VEC_TYPE* X_base_vec = reinterpret_cast<const DATA_VEC_TYPE*>(X + row_id*data_vec_size);

//     /* calculate first tile */
//     int nnz_interleaved_alinged = aligner_interleaved.AlignedNonzeros();
//     if (nnz_interleaved_alinged >= TILE_K) {
//         nnz_interleaved = nnz_interleaved_alinged;
//         col_offset_start = aligner_interleaved.AlignedRowOffset();
//         row_indices_vec_base = reinterpret_cast<const IDX_VEC_TYPE*>(w_merged_row_indice + col_offset_start);

//         /* load row indices into smem */
// #pragma unroll
//         for (int k_item_idx = 0, load_id = 0; k_item_idx < threadItemK; ++k_item_idx, load_id+=TILE_M_VEC) {
//             tile_row_indice_vec_base[load_id] = __ldca(row_indices_vec_base);
//             row_indices_vec_base += TILE_M_VEC;             // update global data ptr
//         }
//         __syncthreads();

//         /* mask extra values */
//         aligner_interleaved.MaskPrefix(tile_row_indice_ld);
//         __syncthreads();
        
//         /* load dense X fragment based on row indices */
// #pragma unroll
//         for (int k = 0; k < TILE_K_VEC; k++) {
//             IDX_VEC_TYPE row_ids = tile_row_indice_vec_ld[k];
//             fragment_X_vec[k] = {__ldca(X_base + row_ids.x*rows), __ldca(X_base + row_ids.y*rows),
//                                 __ldca(X_base + row_ids.z*rows), __ldca(X_base + row_ids.w*rows)};
//         }

//         /* compute */
// #pragma unroll
//         for (int k = 0; k < TILE_K; k+=2) {
//             fragment_res += fragment_X[k] - fragment_X[k+1];
//         }

//         nnz_interleaved -= TILE_K;
//     }

//     /* interleaved computation */
//     for (; nnz_interleaved >= TILE_K; nnz_interleaved -= TILE_K) {
//         /* load row indices into smem */
// #pragma unroll
//         for (int k_item_idx = 0, load_id = 0; k_item_idx < threadItemK; ++k_item_idx, load_id+=TILE_M_VEC) {
//             tile_row_indice_vec_base[load_id] = __ldca(row_indices_vec_base);
//             row_indices_vec_base += TILE_M_VEC;             // update global data ptr
//         }
        
//         __syncthreads();
    
//         /* load dense X fragment based on row indices */
// #pragma unroll
//         for (int k = 0; k < TILE_K_VEC; k++) {
//             IDX_VEC_TYPE row_ids = tile_row_indice_vec_ld[k];
//             fragment_X_vec[k] = {__ldca(X_base + row_ids.x*rows), __ldca(X_base + row_ids.y*rows),
//                                 __ldca(X_base + row_ids.z*rows), __ldca(X_base + row_ids.w*rows)};
//         }

//         /* compute */
// #pragma unroll
//         for (int k = 0; k < TILE_K; k+=2) {
//             fragment_res += fragment_X[k] - fragment_X[k+1];
//         }
//     }

//     /* interleaved residual compute */
//     for (; nnz_interleaved >= TILE_M; nnz_interleaved -= TILE_M) {
//         /* load row indices into smem */
// #pragma unroll
//         for (int k_item_idx = 0, load_id = 0; k_item_idx < 1; ++k_item_idx, load_id+=TILE_M_VEC) {
//             tile_row_indice_vec_base[load_id] = __ldca(row_indices_vec_base);
//             row_indices_vec_base += TILE_M_VEC;             // update global data ptr
//         }
        
//         __syncthreads();

//         /* load dense X fragment based on row indices */
// #pragma unroll
//         for (int k = 0; k < TILE_M_VEC; k++) {
//             IDX_VEC_TYPE row_ids = tile_row_indice_vec_ld[k];
//             fragment_X_vec[k] = {__ldca(X_base + row_ids.x*rows), __ldca(X_base + row_ids.y*rows),
//                                 __ldca(X_base + row_ids.z*rows), __ldca(X_base + row_ids.w*rows)};
//         }

//         /* compute */
// #pragma unroll
//         for (int k = 0; k < TILE_M; k+=2) {
//             fragment_res += fragment_X[k] - fragment_X[k+1];
//         }
//     }


//     /* prepare for remaining computation */
//     row_indices_base_index = col_offset_mid + threadIdx.y*idx_vec_size;
//     tile_row_indice_base = tile_row_indice + tile_base_index;
//     MemoryAligner<IDX_VEC_TYPE, IDX_TYPE, TILE_M> aligner_remain(row_indices_base_index, nnz_remain);

//     /* calculate first tile */
//     nnz_interleaved_alinged = aligner_remain.AlignedNonzeros();
//     if (nnz_interleaved_alinged >= TILE_K) {
//         nnz_remain = nnz_interleaved_alinged;
//         col_offset_mid = aligner_remain.AlignedRowOffset();
//         row_indices_vec_base = reinterpret_cast<const IDX_VEC_TYPE*>(w_merged_row_indice + col_offset_mid);
//         /* load row indices into smem */
// #pragma unroll
//         for (int k_item_idx = 0, load_id = 0; k_item_idx < threadItemK; ++k_item_idx, load_id+=TILE_M_VEC) {
//             tile_row_indice_vec_base[load_id] = __ldca(row_indices_vec_base);
//             row_indices_vec_base += TILE_M_VEC;             // update global data ptr
//         }
//         __syncthreads();

//         /* mask extra values */
//         aligner_remain.MaskPrefix(tile_row_indice_ld);
//         __syncthreads();
        
//         /* load dense X fragment based on row indices */
// #pragma unroll
//         for (int k = 0; k < TILE_K_VEC; k++) {
//             IDX_VEC_TYPE row_ids = tile_row_indice_vec_ld[k];
//             fragment_X_vec[k] = {__ldca(X_base + row_ids.x*rows), __ldca(X_base + row_ids.y*rows),
//                                 __ldca(X_base + row_ids.z*rows), __ldca(X_base + row_ids.w*rows)};
//         }

//         /* compute */
// #pragma unroll
//         for (int k = 0; k < TILE_K; k+=1)
//             cache_sum += fragment_X[k];
        
//         nnz_remain -= TILE_K;
//     }


//     /* common computation */
//     for (; nnz_remain >= TILE_K; nnz_remain -= TILE_K) {
// #pragma unroll
//         for (int k_item_idx = 0, load_id = 0; k_item_idx < threadItemK; ++k_item_idx, load_id+=TILE_M_VEC) {
//             tile_row_indice_vec_base[load_id] = __ldca(row_indices_vec_base);
//             row_indices_vec_base += TILE_M_VEC;             // update global data ptr
//         }

//         __syncthreads();

//         /* load dense X fragment based on row indices */
// #pragma unroll
//         for (int k = 0; k < TILE_K_VEC; k++) {
//             IDX_VEC_TYPE row_ids = tile_row_indice_vec_ld[k];
//             fragment_X_vec[k] = {__ldca(X_base + row_ids.x*rows), __ldca(X_base + row_ids.y*rows),
//                                 __ldca(X_base + row_ids.z*rows), __ldca(X_base + row_ids.w*rows)};
//         }

//         /* compute */
// #pragma unroll
//         for (int k = 0; k < TILE_K; k++) {
//             cache_sum += fragment_X[k];
//         }
//     }

//     /* common residual computation */
//     for (; nnz_remain >= TILE_M; nnz_remain -= TILE_M) {
// #pragma unroll
//         for (int k_item_idx = 0, load_id = 0; k_item_idx < 1; ++k_item_idx, load_id+=TILE_M_VEC) {
//             tile_row_indice_vec_base[load_id] = __ldca(row_indices_vec_base);
//             row_indices_vec_base += TILE_M_VEC;             // update global data ptr
//         }

//         __syncthreads();

//         /* load dense X fragment based on row indices */
// #pragma unroll
//         for (int k = 0; k < TILE_M_VEC; k++) {
//             IDX_VEC_TYPE row_ids = tile_row_indice_vec_ld[k];
//             fragment_X_vec[k] = {__ldca(X_base + row_ids.x*rows), __ldca(X_base + row_ids.y*rows),
//                                 __ldca(X_base + row_ids.z*rows), __ldca(X_base + row_ids.w*rows)};
//         }

//         /* compute */
// #pragma unroll
//         for (int k = 0; k < TILE_K; k++) {
//             cache_sum += fragment_X[k];
//         }
//     }

//     /* compute */
//     result[row_id*columns + col_id] = fragment_res + cache_sum*col_offset.w;
}




template<int TILE_M, int TILE_N, int TILE_K, 
    int FRAGMENT_SIZE, typename VEC_TYPE_M=float, typename DATA_TYPE=float, typename IDX_VEC_TYPE=int4, typename IDX_TYPE=int32_t> 
/**
 * @brief 
 * ThreadBlkY = 1, ThreadBlkX = TILE_N
 * @tparam TILE_M: tile width in M-dim
 * @tparam TILE_N: tile width/block size in N-dim/x-dim
 * @tparam TILE_K: tile width in K-dim
 * @tparam FRAGMENT_SIZE: fragment size for row indices
 * @tparam VEC_SIZE_M: size of vector in M-dim. If scalar, VEC_SIZE_M = 1
 * @tparam VEC_TYPE_M: data_type<BURST_SIZE>
 * @tparam DATA_TYPE: activation and output data type
 * @param w_merged_row_indice 
 * @param w_merged_col_offset 
 * @param result 
 * @param rows 
 * @param columns 
 * @param inners 
 */
__global__ void ter_tiled_mcsc_spmm_nofrag(
    const DATA_TYPE* __restrict__ X, 
    const IDX_TYPE* __restrict__ w_tiled_merged_row_indice, 
    const IDX_TYPE* __restrict__ w_tiled_merged_col_offset, 
    DATA_TYPE* result, 
    int rows, int columns, int inners)
{
    constexpr int vectorSizeM = sizeof(VEC_TYPE_M)/sizeof(DATA_TYPE);   // size of vector used in m-dim
    constexpr int threadVecItemM = TILE_M/vectorSizeM;                  // number of vector each thread needs to load in m-dim
    constexpr int quarterFragmentSize = FRAGMENT_SIZE/4;
    const int tileNumK = (inners + TILE_K - 1)/TILE_K;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= columns || row >= rows)
        return;

    __shared__ DATA_TYPE tile_X[TILE_M*TILE_K]; // row major smem tile
    DATA_TYPE res[TILE_M] = {0};
    DATA_TYPE partial[TILE_M] = {0};
    // __align__(16) IDX_TYPE fragment_row_indices[FRAGMENT_SIZE] = {0};
    
    const IDX_VEC_TYPE* w_tiled_merged_col_offset_vec = reinterpret_cast<const IDX_VEC_TYPE*>(w_tiled_merged_col_offset + col*tileNumK*4);
    const VEC_TYPE_M* X_vec = reinterpret_cast<const VEC_TYPE_M*>(X);

    int row_offset = row*threadVecItemM;    // each thread processes TILE_M/vectorSizeM elements once in m-dim
#pragma unroll
    for (int tileIdK = 0; tileIdK < tileNumK; tileIdK++) {
        /* load into smem */
        int actual_k_offset = tileIdK*TILE_K + threadIdx.x;   // load offset for smem in k-dim
#pragma unroll
        for (int m_id = 0; m_id < threadVecItemM; m_id++) {
            int tile_offset = m_id*TILE_K*vectorSizeM + threadIdx.x;   // start offset for 2-d tile
            int actual_row_id = row_offset + m_id;         // actually row to load

            /* single load */
            DATA_TYPE* tile_X_base = tile_X + tile_offset;
#pragma unroll
            for (int i = 0; i < TILE_K; i+=TILE_N) {
                /* thread loads consecutive element in column major */
                // tile_X[j*TILE_K + threadIdx.x*(TILE_K/TILE_N) + i] = X[rows*(offset + threadIdx.x*(TILE_K/TILE_N) + i) + row*TILE_M + j];

                /* thread loads inconsecutive element in column major */
                // DATA_TYPE x_data = __ldca(X + rows*(actual_k_offset + i) + actual_row_id);
                VEC_TYPE_M x_data_vec = __ldca(X_vec + rows/vectorSizeM*(actual_k_offset + i) + actual_row_id);

                /* thread loads inconsecutive element in row major */
                // DATA_TYPE x_data = __ldca(X + inners*actual_row_id + actual_k_offset + i);
                // tile_X[tile_offset + i] = x_data;

                // presume the vector is float4
                StoreFPVectorToArray(x_data_vec, tile_X_base, i, TILE_K);
            }
        } 
        
        __syncthreads();
        
        IDX_VEC_TYPE col_offset = __ldca(w_tiled_merged_col_offset_vec + tileIdK);
        int col_offset_interleaved_start = col_offset.x;
        int col_offset_remain_start = col_offset.y;
        int nnz_interleaved = col_offset_remain_start - col_offset_interleaved_start;
        int nnz_remain = col_offset.z - col_offset_remain_start;

        for (int i = col_offset.x; i < col_offset.y; i+=2) {
            // todo: use vec access
            int pos_id = __ldca(w_tiled_merged_row_indice + i);
            int neg_id = __ldca(w_tiled_merged_row_indice + i + 1);

#pragma unroll
            for (int j = 0; j < TILE_M; j++) {
                res[j] += tile_X[j*TILE_K + pos_id];
                res[j] -= tile_X[j*TILE_K + neg_id];
            }
        }

        DATA_TYPE partial_sum[TILE_M] = {0};
        for (int i = col_offset.y; i < col_offset.z; i++) {
            int common_id = __ldca(w_tiled_merged_row_indice + i);
#pragma unroll
            for (int j = 0; j < TILE_M; j++) {
                partial_sum[j] += tile_X[j*TILE_K + common_id];
            }
        }

        __syncthreads();
    }


#pragma unroll
    for (int j = 0; j < TILE_M; j++)
        result[(row*TILE_M + j) * columns + col] = res[j];
}