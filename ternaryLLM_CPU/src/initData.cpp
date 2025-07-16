#include"initData.hpp"

//********************* A simple fused SiLU Implementation: SiLU(X) dot XB  **************************************

void Naive_SiLU_Dot_Unroll_AVX2(int LEN, float* X, const float* XB) {
    const int STEP = 64;
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zeros = _mm256_setzero_ps();
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
        __m256 a0 = _mm256_load_ps(&X[i + 0]);
        __m256 a1 = _mm256_load_ps(&X[i + 8]);
        __m256 a2 = _mm256_load_ps(&X[i + 16]);
        __m256 a3 = _mm256_load_ps(&X[i + 24]);
        __m256 a4 = _mm256_load_ps(&X[i + 32]);
        __m256 a5 = _mm256_load_ps(&X[i + 40]);
        __m256 a6 = _mm256_load_ps(&X[i + 48]);
        __m256 a7 = _mm256_load_ps(&X[i + 56]);
        __m256 b0 = _mm256_load_ps(&XB[i + 0]);
        __m256 b1 = _mm256_load_ps(&XB[i + 8]);
        __m256 b2 = _mm256_load_ps(&XB[i + 16]);
        __m256 b3 = _mm256_load_ps(&XB[i + 24]);
        __m256 b4 = _mm256_load_ps(&XB[i + 32]);
        __m256 b5 = _mm256_load_ps(&XB[i + 40]);
        __m256 b6 = _mm256_load_ps(&XB[i + 48]);
        __m256 b7 = _mm256_load_ps(&XB[i + 56]);
        __m256 d0 = _mm256_sub_ps(zeros, a0);
        __m256 d1 = _mm256_sub_ps(zeros, a1);
        __m256 d2 = _mm256_sub_ps(zeros, a2);
        __m256 d3 = _mm256_sub_ps(zeros, a3);
        __m256 d4 = _mm256_sub_ps(zeros, a4);
        __m256 d5 = _mm256_sub_ps(zeros, a5);
        __m256 d6 = _mm256_sub_ps(zeros, a6);
        __m256 d7 = _mm256_sub_ps(zeros, a7);
        a0 = _mm256_mul_ps(a0, b0);
        a1 = _mm256_mul_ps(a1, b1);
        a2 = _mm256_mul_ps(a2, b2);
        a3 = _mm256_mul_ps(a3, b3);
        a4 = _mm256_mul_ps(a4, b4);
        a5 = _mm256_mul_ps(a5, b5);
        a6 = _mm256_mul_ps(a6, b6);
        a7 = _mm256_mul_ps(a7, b7);
        d0 = _mm256_exp_ps(d0);
        d1 = _mm256_exp_ps(d1);
        d2 = _mm256_exp_ps(d2);
        d3 = _mm256_exp_ps(d3);
        d4 = _mm256_exp_ps(d4);
        d5 = _mm256_exp_ps(d5);
        d6 = _mm256_exp_ps(d6);
        d7 = _mm256_exp_ps(d7);
        d0 = _mm256_add_ps(d0, ones);
        d1 = _mm256_add_ps(d1, ones);
        d2 = _mm256_add_ps(d2, ones);
        d3 = _mm256_add_ps(d3, ones);
        d4 = _mm256_add_ps(d4, ones);
        d5 = _mm256_add_ps(d5, ones);
        d6 = _mm256_add_ps(d6, ones);
        d7 = _mm256_add_ps(d7, ones);
        d0 = _mm256_rcp_ps(d0);
        d1 = _mm256_rcp_ps(d1);
        d2 = _mm256_rcp_ps(d2);
        d3 = _mm256_rcp_ps(d3);
        d4 = _mm256_rcp_ps(d4);
        d5 = _mm256_rcp_ps(d5);
        d6 = _mm256_rcp_ps(d6);
        d7 = _mm256_rcp_ps(d7);
        d0 = _mm256_mul_ps(a0, d0);
        d1 = _mm256_mul_ps(a1, d1);
        d2 = _mm256_mul_ps(a2, d2);
        d3 = _mm256_mul_ps(a3, d3);
        d4 = _mm256_mul_ps(a4, d4);
        d5 = _mm256_mul_ps(a5, d5);
        d6 = _mm256_mul_ps(a6, d6);
        d7 = _mm256_mul_ps(a7, d7);
        _mm256_store_ps(&X[i + 0], d0);
        _mm256_store_ps(&X[i + 8], d1);
        _mm256_store_ps(&X[i + 16], d2);
        _mm256_store_ps(&X[i + 24], d3);
        _mm256_store_ps(&X[i + 32], d4);
        _mm256_store_ps(&X[i + 40], d5);
        _mm256_store_ps(&X[i + 48], d6);
        _mm256_store_ps(&X[i + 56], d7);
    }
};


void Naive_SiLU_Dot_Unroll_AVX512(int LEN, float* X, const float* XB) {
    const int STEP = 128;
    const __m512 ones = _mm512_set1_ps(1.0f);
    const __m512 zeros = _mm512_setzero_ps();
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
        __m512 a0 = _mm512_load_ps(&X[i + 0]);
        __m512 a1 = _mm512_load_ps(&X[i + 16]);
        __m512 a2 = _mm512_load_ps(&X[i + 32]);
        __m512 a3 = _mm512_load_ps(&X[i + 48]);
        __m512 a4 = _mm512_load_ps(&X[i + 64]);
        __m512 a5 = _mm512_load_ps(&X[i + 80]);
        __m512 a6 = _mm512_load_ps(&X[i + 96]);
        __m512 a7 = _mm512_load_ps(&X[i + 112]);
        __m512 b0 = _mm512_load_ps(&XB[i + 0]);
        __m512 b1 = _mm512_load_ps(&XB[i + 16]);
        __m512 b2 = _mm512_load_ps(&XB[i + 32]);
        __m512 b3 = _mm512_load_ps(&XB[i + 48]);
        __m512 b4 = _mm512_load_ps(&XB[i + 64]);
        __m512 b5 = _mm512_load_ps(&XB[i + 80]);
        __m512 b6 = _mm512_load_ps(&XB[i + 96]);
        __m512 b7 = _mm512_load_ps(&XB[i + 112]);
        __m512 d0 = _mm512_sub_ps(zeros, a0);
        __m512 d1 = _mm512_sub_ps(zeros, a1);
        __m512 d2 = _mm512_sub_ps(zeros, a2);
        __m512 d3 = _mm512_sub_ps(zeros, a3);
        __m512 d4 = _mm512_sub_ps(zeros, a4);
        __m512 d5 = _mm512_sub_ps(zeros, a5);
        __m512 d6 = _mm512_sub_ps(zeros, a6);
        __m512 d7 = _mm512_sub_ps(zeros, a7);
        a0 = _mm512_mul_ps(a0, b0);
        a1 = _mm512_mul_ps(a1, b1);
        a2 = _mm512_mul_ps(a2, b2);
        a3 = _mm512_mul_ps(a3, b3);
        a4 = _mm512_mul_ps(a4, b4);
        a5 = _mm512_mul_ps(a5, b5);
        a6 = _mm512_mul_ps(a6, b6);
        a7 = _mm512_mul_ps(a7, b7);
        d0 = _mm512_exp_ps(d0);
        d1 = _mm512_exp_ps(d1);
        d2 = _mm512_exp_ps(d2);
        d3 = _mm512_exp_ps(d3);
        d4 = _mm512_exp_ps(d4);
        d5 = _mm512_exp_ps(d5);
        d6 = _mm512_exp_ps(d6);
        d7 = _mm512_exp_ps(d7);
        d0 = _mm512_add_ps(d0, ones);
        d1 = _mm512_add_ps(d1, ones);
        d2 = _mm512_add_ps(d2, ones);
        d3 = _mm512_add_ps(d3, ones);
        d4 = _mm512_add_ps(d4, ones);
        d5 = _mm512_add_ps(d5, ones);
        d6 = _mm512_add_ps(d6, ones);
        d7 = _mm512_add_ps(d7, ones);
        d0 = _mm512_rcp14_ps(d0);
        d1 = _mm512_rcp14_ps(d1);
        d2 = _mm512_rcp14_ps(d2);
        d3 = _mm512_rcp14_ps(d3);
        d4 = _mm512_rcp14_ps(d4);
        d5 = _mm512_rcp14_ps(d5);
        d6 = _mm512_rcp14_ps(d6);
        d7 = _mm512_rcp14_ps(d7);
        d0 = _mm512_mul_ps(a0, d0);
        d1 = _mm512_mul_ps(a1, d1);
        d2 = _mm512_mul_ps(a2, d2);
        d3 = _mm512_mul_ps(a3, d3);
        d4 = _mm512_mul_ps(a4, d4);
        d5 = _mm512_mul_ps(a5, d5);
        d6 = _mm512_mul_ps(a6, d6);
        d7 = _mm512_mul_ps(a7, d7);
        _mm512_store_ps(&X[i + 0], d0);
        _mm512_store_ps(&X[i + 16], d1);
        _mm512_store_ps(&X[i + 32], d2);
        _mm512_store_ps(&X[i + 48], d3);
        _mm512_store_ps(&X[i + 64], d4);
        _mm512_store_ps(&X[i + 80], d5);
        _mm512_store_ps(&X[i + 96], d6);
        _mm512_store_ps(&X[i + 112], d7);
    }
};


//********************* A simple Matrix Transposing Implementation **************************************

// Fast 8x8 transpose using AVX2 intrinsics
inline void transpose_8x8_avx2(const float* src, float* dst, int src_stride, int dst_stride) {
    // Load 8 rows of 8 floats each
    __m256 row0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 row1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 row2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 row3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 row4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 row5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 row6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 row7 = _mm256_loadu_ps(src + 7 * src_stride);

    // Stage 1: Interleave adjacent pairs
    __m256 tmp0 = _mm256_unpacklo_ps(row0, row1);
    __m256 tmp1 = _mm256_unpackhi_ps(row0, row1);
    __m256 tmp2 = _mm256_unpacklo_ps(row2, row3);
    __m256 tmp3 = _mm256_unpackhi_ps(row2, row3);
    __m256 tmp4 = _mm256_unpacklo_ps(row4, row5);
    __m256 tmp5 = _mm256_unpackhi_ps(row4, row5);
    __m256 tmp6 = _mm256_unpacklo_ps(row6, row7);
    __m256 tmp7 = _mm256_unpackhi_ps(row6, row7);

    // Stage 2: Interleave 64-bit blocks
    __m256 tmp8 = _mm256_shuffle_ps(tmp0, tmp2, 0x44);
    __m256 tmp9 = _mm256_shuffle_ps(tmp0, tmp2, 0xEE);
    __m256 tmp10 = _mm256_shuffle_ps(tmp1, tmp3, 0x44);
    __m256 tmp11 = _mm256_shuffle_ps(tmp1, tmp3, 0xEE);
    __m256 tmp12 = _mm256_shuffle_ps(tmp4, tmp6, 0x44);
    __m256 tmp13 = _mm256_shuffle_ps(tmp4, tmp6, 0xEE);
    __m256 tmp14 = _mm256_shuffle_ps(tmp5, tmp7, 0x44);
    __m256 tmp15 = _mm256_shuffle_ps(tmp5, tmp7, 0xEE);

    // Stage 3: Interleave 128-bit lanes
    row0 = _mm256_permute2f128_ps(tmp8, tmp12, 0x20);
    row1 = _mm256_permute2f128_ps(tmp9, tmp13, 0x20);
    row2 = _mm256_permute2f128_ps(tmp10, tmp14, 0x20);
    row3 = _mm256_permute2f128_ps(tmp11, tmp15, 0x20);
    row4 = _mm256_permute2f128_ps(tmp8, tmp12, 0x31);
    row5 = _mm256_permute2f128_ps(tmp9, tmp13, 0x31);
    row6 = _mm256_permute2f128_ps(tmp10, tmp14, 0x31);
    row7 = _mm256_permute2f128_ps(tmp11, tmp15, 0x31);

    // Store transposed result
    _mm256_storeu_ps(dst + 0 * dst_stride, row0);
    _mm256_storeu_ps(dst + 1 * dst_stride, row1);
    _mm256_storeu_ps(dst + 2 * dst_stride, row2);
    _mm256_storeu_ps(dst + 3 * dst_stride, row3);
    _mm256_storeu_ps(dst + 4 * dst_stride, row4);
    _mm256_storeu_ps(dst + 5 * dst_stride, row5);
    _mm256_storeu_ps(dst + 6 * dst_stride, row6);
    _mm256_storeu_ps(dst + 7 * dst_stride, row7);
}

// Optimized 16x16 block transpose using 8x8 micro-blocks
inline void transpose_16x16_block(const float* src, float* dst, int src_stride, int dst_stride) {
    // Process 4 8x8 micro-blocks within the 16x16 block
    transpose_8x8_avx2(src, dst, src_stride, dst_stride);
    transpose_8x8_avx2(src + 8, dst + 8 * dst_stride, src_stride, dst_stride);
    transpose_8x8_avx2(src + 8 * src_stride, dst + 8, src_stride, dst_stride);
    transpose_8x8_avx2(src + 8 * src_stride + 8, dst + 8 * dst_stride + 8, src_stride, dst_stride);
}

// Optimized 32x32 block transpose using 16x16 micro-blocks
inline void transpose_32x32_block(const float* src, float* dst, int src_stride, int dst_stride) {
    // Process 4 16x16 micro-blocks within the 32x32 block
    transpose_16x16_block(src, dst, src_stride, dst_stride);
    transpose_16x16_block(src + 16, dst + 16 * dst_stride, src_stride, dst_stride);
    transpose_16x16_block(src + 16 * src_stride, dst + 16, src_stride, dst_stride);
    transpose_16x16_block(src + 16 * src_stride + 16, dst + 16 * dst_stride + 16, src_stride, dst_stride);
}

// Handle remaining rows (when H is not divisible by block size)
inline void transpose_remaining_rows(const float* X, float* XT, int start_row, int H, int W) {
    for (int i = start_row; i < H; ++i) {
        const float* src_row = X + i * W;
        // Process 8 elements at a time using AVX2
        int j = 0;
        for (; j + 7 < W; j += 8) {
            __m256 data = _mm256_loadu_ps(src_row + j);
            for (int k = 0; k < 8; ++k) {
                XT[(j + k) * H + i] = ((float*)&data)[k];
            }
        }
        // Handle remaining elements
        for (; j < W; ++j) {
            XT[j * H + i] = src_row[j];
        }
    }
}

// Main fast matrix transpose function
void FastMatrixTranspose(const float* X, float* XT, const int H, const int W) {
    constexpr int BLOCK_SIZE = 32;

    // Process full 32x32 blocks
    int blocks_h = H / BLOCK_SIZE;
    int blocks_w = W / BLOCK_SIZE;

    // Transpose full blocks
    for (int block_i = 0; block_i < blocks_h; ++block_i) {
#pragma omp parallel for
        for (int block_j = 0; block_j < blocks_w; ++block_j) {
            const float* src_block = X + block_i * BLOCK_SIZE * W + block_j * BLOCK_SIZE;
            float* dst_block = XT + block_j * BLOCK_SIZE * H + block_i * BLOCK_SIZE;

            transpose_32x32_block(src_block, dst_block, W, H);
        }
    }

    // Handle remaining rows (partial height blocks)
    int remaining_rows = H % BLOCK_SIZE;
    if (remaining_rows > 0) {
        int start_row = blocks_h * BLOCK_SIZE;

        // Process remaining rows with full width blocks
        for (int block_j = 0; block_j < blocks_w; ++block_j) {
            int col_start = block_j * BLOCK_SIZE;

            // Process remaining rows in this column block
            for (int i = start_row; i < H; ++i) {
                const float* src_row = X + i * W + col_start;

                // Process 8 elements at a time using AVX2
                int j = 0;
                for (; j + 7 < BLOCK_SIZE; j += 8) {
                    __m256 data = _mm256_loadu_ps(src_row + j);
                    // Manual unroll for better performance
                    XT[(col_start + j + 0) * H + i] = ((float*)&data)[0];
                    XT[(col_start + j + 1) * H + i] = ((float*)&data)[1];
                    XT[(col_start + j + 2) * H + i] = ((float*)&data)[2];
                    XT[(col_start + j + 3) * H + i] = ((float*)&data)[3];
                    XT[(col_start + j + 4) * H + i] = ((float*)&data)[4];
                    XT[(col_start + j + 5) * H + i] = ((float*)&data)[5];
                    XT[(col_start + j + 6) * H + i] = ((float*)&data)[6];
                    XT[(col_start + j + 7) * H + i] = ((float*)&data)[7];
                }
                // Handle remaining elements in the block
                for (; j < BLOCK_SIZE; ++j) {
                    XT[(col_start + j) * H + i] = src_row[j];
                }
            }
        }
    }
}