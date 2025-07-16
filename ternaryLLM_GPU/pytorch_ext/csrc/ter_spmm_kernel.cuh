// Author: fuguan@ethz.ch
// Copyrights reserved
#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"
#include <iostream>
#include <cassert>


template<int TILE_K, bool UNIFORMED, typename DATA_TYPE, typename IDX_TYPE>
/**
 * @brief ternary spmv kernel for tile-wise ternary csc format
 * 
 * @tparam TILE_K
 * @tparam DATA_TYPE 
 * @param X 
 * @param w_tiled_merged_row_indice 
 * @param w_tiled_merged_col_offset 
 * @param result 
 */
__global__ void ter_tiled_mcsc_spmv(
    const DATA_TYPE* __restrict__ X, 
    const IDX_TYPE* __restrict__ w_tiled_merged_row_indice, 
    const int32_t* __restrict__ w_tiled_merged_col_offset,
    DATA_TYPE* result,
    const unsigned int rows, 
    const unsigned int columns, 
    const unsigned int inners
)
{
    using DataScalar = DATA_TYPE;
    using OffsetVec = CustomIdxVec<int32_t, 4>;
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;   // thread id in a grid
    const unsigned int warp_id = thread_id / 32;    // warp id
    const unsigned int lane_id = thread_id % 32;    // thread id in a warp
    const unsigned int col = warp_id; // one warp per column
    const int tileNumK = (inners + TILE_K - 1)/TILE_K;

    const OffsetVec* w_tiled_merged_col_offset_vec = reinterpret_cast<const OffsetVec*>(w_tiled_merged_col_offset + col*tileNumK*4);

    DataScalar sum = 0;
    if (col < columns) {
        for (int tileIdK = 0; tileIdK < tileNumK; tileIdK++) {
            OffsetVec col_offset;
            col_offset.LoadByVec(w_tiled_merged_col_offset_vec, tileIdK);
            int32_t col_offset_interleaved_start = col_offset.ReadAsScalar(0);
            int32_t col_offset_remain_start = col_offset.ReadAsScalar(1);
            int x_offset = tileIdK*TILE_K;

            for (int i = col_offset_interleaved_start + lane_id*2; i < col_offset_remain_start; i+=2*32) {
                int pos_id = __ldca(w_tiled_merged_row_indice + i);
                int neg_id = __ldca(w_tiled_merged_row_indice + i + 1);
                sum += __ldca(X + x_offset + pos_id) - __ldca(X + x_offset + neg_id);
            }

            if constexpr (!UNIFORMED) {
                DataScalar partial_sum = 0;
                for (int i = col_offset_remain_start + lane_id; i < col_offset.ReadAsScalar(2); i+=32) {
                    int common_id = __ldca(w_tiled_merged_row_indice + i);
                    partial_sum += X[common_id];
                }

                sum += col_offset.ReadAsScalar(3)*partial_sum;
            }
        }
    }

    /* reduce among the warp */
    warp_reduce_sum(sum);

    /* one thread in a warp writes back */
    if (lane_id == 0 && col < columns) {
        result[col] = sum;
    }
}






template<int TILE_M, int TILE_N, int TILE_K, int FRAGMENT_SIZE=32, bool UNIFORMED=false, bool PADDED=false,
    typename VEC_TYPE_M=float, typename DATA_TYPE=float, typename IDX_TYPE=int32_t> 
/**
 * @brief ternary spmm kernel for tile-wise ternary csc format
 * ThreadBlkY = 1, ThreadBlkX = TILE_N
 * @tparam TILE_M: tile width in M-dim
 * @tparam TILE_N: tile width/block size in N-dim/x-dim
 * @tparam TILE_K: tile width in K-dim
 * @tparam FRAGMENT_SIZE: fragment size for row indices
 * @tparam VEC_SIZE_M: size of vector in M-dim. If scalar, VEC_SIZE_M = 1
 * @tparam VEC_TYPE_M: data_type<BURST_SIZE>
 * @tparam DATA_TYPE: activation and output data type
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
    const int32_t* __restrict__ w_tiled_merged_col_offset, 
    const IDX_TYPE* __restrict__ w_col_tile_indices,
    DATA_TYPE* result, 
    int rows, int columns, int inners
)
{
    using DataScalar = DATA_TYPE;
    using DataVecM = VEC_TYPE_M;
    using IdxScalar = IDX_TYPE;
    using IdxVec = CustomIdxVec<IDX_TYPE, 4>;
    using OffsetVec = CustomIdxVec<int32_t, 4>;
    constexpr int vectorSizeM = sizeof(DataVecM)/sizeof(DataScalar);    // size of vector used in m-dim of dense X, normally 4
    constexpr int vectorSizeIdx = 4;                                    // size of vector used in row indices
    constexpr int threadVecItemM = TILE_M/vectorSizeM;                  // number of vector each thread needs to load in m-dim for dense X
    constexpr int threadVecItemIdx = FRAGMENT_SIZE/vectorSizeIdx;       // number of vector each thread needs to load for row indices
    constexpr int quarterFragmentSize = FRAGMENT_SIZE/4;
    constexpr int threadVecItemQuarterIdx = quarterFragmentSize/vectorSizeIdx;       // number of vector each thread needs to load for row indices
    constexpr int TILE_M_ = UNIFORMED ? 1 : TILE_M;
    // static_assert(threadVecItemM > 0, "TILE_M has to be larger than vectorSizeM");
    // static_assert(threadVecItemIdx > 0, "FRAGMENT_SIZE has to be larger than vectorSizeIdx");
    // static_assert(quarterFragmentSize > 0, "FRAGMENT_SIZE has to be larger than 4");
    // static_assert(threadVecItemQuarterIdx > 0, "vectorSizeIdx has to be larger than 4");

    const int tileNumK = (inners + TILE_K - 1)/TILE_K;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    w_col_tile_indices += col*tileNumK;

    /* todo: remove this branch */
    if (col >= columns || row*threadVecItemM >= rows)
        return;

    /* smem for dense X */
    __shared__ DataScalar tile_X[TILE_M*TILE_K]; // row major smem tile

    /* fragment for sparse arrays  */
    __align__(16) IdxScalar fragment_row_indices[FRAGMENT_SIZE] = {0};
    IdxVec* fragment_row_indices_vec = reinterpret_cast<IdxVec*>(fragment_row_indices);

    /* results */
    DataScalar res[TILE_M] = {0};    // each thread processes TILE_M * 1 results
    DataScalar partial[TILE_M_] = {0};
    
    /* vectorized access to gmem */
    const OffsetVec* w_tiled_merged_col_offset_vec = reinterpret_cast<const OffsetVec*>(w_tiled_merged_col_offset + col*tileNumK*4);
    const DataVecM* X_vec = reinterpret_cast<const DataVecM*>(X);

    int row_offset = row*threadVecItemM;    // each thread processes TILE_M/vectorSizeM elements once in m-dim
#pragma unroll
    for (int tileCnt = 0; tileCnt < tileNumK; tileCnt++) {
        int tileIdK = tileCnt;
        // int tileIdK = w_col_tile_indices[tileCnt];      // swizzled tile to process
        /* load into smem */
        int actual_k_offset = tileIdK*TILE_K + threadIdx.x;   // load offset for smem in k-dim
#pragma unroll
        for (int m_id = 0; m_id < threadVecItemM; m_id++) {
            int tile_offset = m_id*TILE_K*vectorSizeM + threadIdx.x;   // start offset for 2-d tile
            int actual_row_id = row_offset + m_id;         // actual row to load

            /* single load */
            DataScalar* tile_X_base = tile_X + tile_offset;
#pragma unroll
            for (int i = 0; i < TILE_K; i+=TILE_N) {
                /* thread loads inconsecutive element in column major */
                DataVecM x_data_vec = __ldca(X_vec + rows/vectorSizeM*(actual_k_offset + i) + actual_row_id);

                /* thread loads inconsecutive element in row major */
                // DATA_TYPE x_data = __ldca(X + inners*actual_row_id + actual_k_offset + i);
                // tile_X[tile_offset + i] = x_data;

                // store by vector
                StoreFPVectorToArray(x_data_vec, tile_X_base + i, TILE_K);
            }
        } 
        
        __syncthreads();
        
        OffsetVec col_offset;
        col_offset.LoadByVec(w_tiled_merged_col_offset_vec, tileIdK);
        int32_t col_offset_interleaved_start = col_offset.ReadAsScalar(0);
        int32_t col_offset_remain_start = col_offset.ReadAsScalar(1);
        int nnz_interleaved = col_offset_remain_start - col_offset_interleaved_start;
        int nnz_remain = col_offset.ReadAsScalar(2) - col_offset_remain_start;

        /* interleaved part computation */
        const IdxScalar* w_tiled_merged_row_indice_base = w_tiled_merged_row_indice + col_offset_interleaved_start;
        const IdxVec* w_tiled_merged_row_indice_base_vec = reinterpret_cast<const IdxVec*>(w_tiled_merged_row_indice_base);

        for (; nnz_interleaved >= FRAGMENT_SIZE; nnz_interleaved -= FRAGMENT_SIZE) {
#pragma unroll
            for (int i = 0; i < threadVecItemIdx; i++) 
                fragment_row_indices_vec[i].LoadByVec(w_tiled_merged_row_indice_base_vec, i);
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

        for (; nnz_interleaved >= quarterFragmentSize; nnz_interleaved -= quarterFragmentSize) {
#pragma unroll
            for (int i = 0; i < threadVecItemQuarterIdx; i++) 
                fragment_row_indices_vec[i].LoadByVec(w_tiled_merged_row_indice_base_vec, i);
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
        
        /* interleaved residual computation */
        /* if padding is not added, turn on residual calculation for interleaved part */
        /* interleaved part can be padded to the multiple of quarterFragmentSize, normally 8, 
            which will increase some memory overheads */
        if constexpr (!PADDED) {
#pragma unroll
        for (int i = 0; i < nnz_interleaved/vectorSizeIdx; i+=1) {
            IdxVec row_ids = w_tiled_merged_row_indice_base_vec[i];
#pragma unroll
            for (int j = 0; j < TILE_M; j++) {
                int tile_offset = j*TILE_K;
                res[j] += tile_X[tile_offset + row_ids.ReadAsScalar(0)] + tile_X[tile_offset + row_ids.ReadAsScalar(2)];
                res[j] -= tile_X[tile_offset + row_ids.ReadAsScalar(1)] + tile_X[tile_offset + row_ids.ReadAsScalar(3)];
            }
        }
        }
        
        /* common part computation */
        if constexpr (!UNIFORMED) {
        w_tiled_merged_row_indice_base = w_tiled_merged_row_indice + col_offset_remain_start;
        w_tiled_merged_row_indice_base_vec = reinterpret_cast<const IdxVec*>(w_tiled_merged_row_indice_base);
        for (; nnz_remain >= FRAGMENT_SIZE; nnz_remain -= FRAGMENT_SIZE) {
#pragma unroll
            for (int i = 0; i < threadVecItemIdx; i++) 
                fragment_row_indices_vec[i] = w_tiled_merged_row_indice_base_vec[i];
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
                fragment_row_indices_vec[i] = w_tiled_merged_row_indice_base_vec[i];
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
        for (int i = 0; i < nnz_remain/vectorSizeIdx; i+=1) {
            IdxVec common_ids = w_tiled_merged_row_indice_base_vec[i];
#pragma unroll
            for (int m_id = 0; m_id < TILE_M; m_id++) {
                int tile_offset = m_id*TILE_K;
                partial[m_id] += tile_X[tile_offset + common_ids.ReadAsScalar(0)] + tile_X[tile_offset + common_ids.ReadAsScalar(2)] +
                                tile_X[tile_offset + common_ids.ReadAsScalar(1)] + tile_X[tile_offset + common_ids.ReadAsScalar(3)];
            }
        }
        
        /* aggregate sum of interleaved part and common part */
#pragma unroll
        for (int m_id = 0; m_id < TILE_M; m_id++) {
            res[m_id] += col_offset.ReadAsScalar(3) * partial[m_id];
            partial[m_id] = 0;
        }
        }
        __syncthreads();
    }


#pragma unroll
    for (int j = 0; j < TILE_M; j++)
        result[(row*TILE_M + j) * columns + col] = res[j];
}




/**
 * @brief Caller of spmm kernel
 */
template<int TILE_M, int TILE_N, int TILE_K, int FRAGMENT_SIZE, bool UNIFORMED, bool PADDED,
    typename VEC_TYPE_M=float4, typename DATA_TYPE=float, typename IDX_TYPE=int16_t> 
__host__ void ter_spmm_kernel_caller(
    const DATA_TYPE* X,
    const int16_t* w_tiled_merged_row_indice,
    const int32_t* w_tiled_merged_col_offset,
    const int16_t* w_col_tile_indices,
    DATA_TYPE* result, 
    int batch_size, int rows, int columns, int inners
)
{
    dim3 blkSize = dim3(TILE_N, 1);
    dim3 gridSize = dim3((int)ceil(1.*columns/TILE_N), (int)ceil(1.*rows/1/TILE_M));

    if (batch_size%2 == 1) {
        ter_tiled_mcsc_spmm<TILE_M, TILE_N, TILE_K, FRAGMENT_SIZE, UNIFORMED, PADDED, VEC_TYPE_M, DATA_TYPE, IDX_TYPE><<<gridSize, blkSize>>>
        (
            X,
            w_tiled_merged_row_indice, 
            w_tiled_merged_col_offset, 
            w_col_tile_indices,
            result,
            rows, columns, inners
        );
        batch_size--;
        X += rows*inners;
        result += rows*columns;
    }

    cudaStream_t stream_0, stream_1;
    cudaStreamCreate(&stream_0);
    cudaStreamCreate(&stream_1);
    for (int b = 0; b < batch_size; b += 2) {
        ter_tiled_mcsc_spmm<TILE_M, TILE_N, TILE_K, FRAGMENT_SIZE, UNIFORMED, PADDED, VEC_TYPE_M, DATA_TYPE, IDX_TYPE><<<gridSize, blkSize, 0, stream_0>>>
        (
            X,
            w_tiled_merged_row_indice, 
            w_tiled_merged_col_offset, 
            w_col_tile_indices,
            result,
            rows, columns, inners
        );
        X += rows*inners;
        result += rows*columns;

        ter_tiled_mcsc_spmm<TILE_M, TILE_N, TILE_K, FRAGMENT_SIZE, UNIFORMED, PADDED, VEC_TYPE_M, DATA_TYPE, IDX_TYPE><<<gridSize, blkSize, 0, stream_1>>>
        (
            X,
            w_tiled_merged_row_indice, 
            w_tiled_merged_col_offset, 
            w_col_tile_indices,
            result,
            rows, columns, inners
        );
        X += rows*inners;
        result += rows*columns;
    }
    cudaStreamDestroy(stream_0);
    cudaStreamDestroy(stream_1);
}



/**
 * @brief Caller of spmv kernel
 */
template<int TILE_K, bool UNIFORMED, typename DATA_TYPE, typename IDX_TYPE>
__host__ void ter_spmv_kernel_caller(
    const DATA_TYPE* X,
    const int16_t* w_tiled_merged_row_indice,
    const int32_t* w_tiled_merged_col_offset,
    const int16_t*,
    DATA_TYPE* result, 
    int batch_size, int rows, int columns, int inners
)
{
    dim3 blkSize = dim3(256);
    dim3 gridSize = dim3((int)ceil(1.*32*columns/256));

    if (batch_size%2 == 1) {
        ter_tiled_mcsc_spmv<TILE_K, UNIFORMED, DATA_TYPE, IDX_TYPE><<<gridSize, blkSize>>>
        (
            X,
            w_tiled_merged_row_indice, 
            w_tiled_merged_col_offset, 
            result,
            rows, columns, inners
        );
        batch_size--;
        X += rows*inners;
        result += rows*columns;
    }

    cudaStream_t stream_0, stream_1;
    cudaStreamCreate(&stream_0);
    cudaStreamCreate(&stream_1);
    for (int b = 0; b < batch_size; b += 2) {
        ter_tiled_mcsc_spmv<TILE_K, UNIFORMED, DATA_TYPE, IDX_TYPE><<<gridSize, blkSize, 0, stream_0>>>
        (
            X,
            w_tiled_merged_row_indice, 
            w_tiled_merged_col_offset, 
            result,
            rows, columns, inners
        );
        X += rows*inners;
        result += rows*columns;

        ter_tiled_mcsc_spmv<TILE_K, UNIFORMED, DATA_TYPE, IDX_TYPE><<<gridSize, blkSize, 0, stream_1>>>
        (
            X,
            w_tiled_merged_row_indice, 
            w_tiled_merged_col_offset, 
            result,
            rows, columns, inners
        );
        X += rows*inners;
        result += rows*columns;
    }
    cudaStreamDestroy(stream_0);
    cudaStreamDestroy(stream_1);
}

/* function signature for kernel caller */
using KernelCaller = void(*)( 
    const float*, const int16_t*, const int32_t*, const int16_t*, float*, 
    int, int, int, int
);

/**
 * @brief Selecter of template function
 * 
 * @param batch_size 
 * @param rows 
 * @param columns 
 * @param inners 
 * @return KernelCaller 
 */
KernelCaller kernel_dispatcher(
    int batch_size, int rows, int columns, int inners,
    bool uniformed, bool padded
)
{
    assert((inners % 256 == 0) && "K-dim has be to multiple of 256");
    assert((columns % 8 == 0) && "N-dim has be to multiple of 8");
    if (rows > 1) {
        DISPATCH_M(
            rows,
            kVecTypeM,
            kTileM,
            DISPATCH_K(
                inners,
                kTileK,
                kFragment,
                DISPATCH_BOOL(
                    uniformed,
                    kUniformed,
                    DISPATCH_BOOL(
                        padded,
                        kPadded,

                        /* currently we use the same size for tile_k and tile_n */
                        return ter_spmm_kernel_caller<kTileM, kTileK, kTileK, kFragment, kUniformed, kPadded, kVecTypeM, float, int16_t>;
                    );  // dispatch padded
                );  // dispatch uniformed
            );  // dispatch k
        );  // dispatch m
    } else {
        DISPATCH_K(
            inners,
            kTileK,
            kFragment,
            DISPATCH_BOOL(
                uniformed,
                kUniformed,

                return ter_spmv_kernel_caller<kTileK, kUniformed, float, int16_t>;
            ); // dispatch uniformed
        ); // dispatch k
    }
}

/**
 * @brief Wrapper exposed to python
 * 
 * @tparam DATA_TYPE 
 * @param X 
 * @param w_tiled_merged_row_indice 
 * @param w_tiled_merged_col_offset 
 * @param result 
 * @param rows 
 * @param columns 
 * @param inners 
 */
template<typename DATA_TYPE=float> 
void ter_spmm_wrapper(
    const DATA_TYPE* X,
    const int16_t* w_tiled_merged_row_indice,
    const int32_t* w_tiled_merged_col_offset,
    const int16_t* w_col_tile_indices,
    DATA_TYPE* result, 
    int batch_size, int rows, int columns, int inners,
    bool uniformed, bool padded
) 
{

    auto fn = kernel_dispatcher(batch_size, rows, columns, inners, uniformed, padded);
    fn(
        X,
        w_tiled_merged_row_indice,
        w_tiled_merged_col_offset,
        w_col_tile_indices,
        result, 
        batch_size, rows, columns, inners
    );
}