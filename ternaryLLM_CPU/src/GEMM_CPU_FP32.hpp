#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <tuple>
#include <sstream> 
#include <immintrin.h>

void GEMM_CPU_FP32_colMajor_Direct_OpenMP(float* X, int8_t* W1, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_rowMajor_Direct_OpenMP(float* X, int8_t* W1, float* result, int rows, int columns, int inners);

template <typename T>
void GEMM_CPU_colMajor_Direct_OpenMP(T* X, T* W1, T* result, int rows, int columns, int inners) {
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
#pragma omp simd
            for (int k = 0; k < inners; k++) {
                result[j * rows + i] += X[k * rows + i] * W1[k * columns + j];
            }
        }
    }
}

template <typename T>
void GEMM(T* X, T* W, T* b, T* Y, int M, int N, int K) {
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			T y = 0;
			for (int k = 0; k < K; k++) {
				y += X[m * K + k] * W[k * N + n];
			}
			Y[m * N + n] = y + b[n];
		}
	}
};

void GEMM_CPU_FP32_colMajor_TCSC_Naive(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2_OpenMP(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);

void GEMM_CPU_FP32_colMajor_TCSC_Merged_8x1_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);

void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG16_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG16_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG32_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Aligned_GroupMax_8xG4_AVX2_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);

void GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG4_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG16_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_8xG32_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG4_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG16_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG32_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
// AVX-512
void GEMM_CPU_FP32_colMajor_TCSC_Merged_16xG1_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG4_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG8_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG16_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_16xG32_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_32xG1_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG4_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG8_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG16_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_32xG32_AVX512_OpenMP(const float* X, const int32_t* metadata, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);

void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG1_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG4_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG8_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG16_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_16xG32_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG1_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG4_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG8_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG16_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_32xG32_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_FP32_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int16_t* row_index, float* result, const int M_ROW, const int N_COL, const int K);

/* RowMajor*/

void GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_OpenMP(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);
void GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_4x4_Unroll_OpenMP(float* X, int16_t* w_neg_row_ind, int32_t* w_neg_col_ptr, int16_t* w_pos_row_ind, int32_t* w_pos_col_ptr, float* result, int rows, int columns, int inners);

void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPij(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPijj(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPji(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMPjii(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);

void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG8_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG8_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG8_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG8_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);

void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG16_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG16_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG16_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG16_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG16_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);

void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG32_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG32_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG32_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG32_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG32_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);

void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG64_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_2xG64_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_4xG64_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_8xG64_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);
void GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_16xG64_AVX2_OpenMP(float* X, int32_t* metadata, int32_t* row_index, float* result, int M_ROW, int N_COL, int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_CS8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_CS8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD1_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD2_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD3_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS8_AVX2_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_CS8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_CS8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_CS8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_CS8_AVX2_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_SIMD3_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G32_CS16_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS16_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD1_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD2_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G128_CS16_SIMD3_AVX512_OpenMP(const float* X, const int32_t* metadata, const int32_t* row_index, float* result, const int N_COL, const int K);
 
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G16_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);

void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G32_CS16_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G64_CS16_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);
void GEMV_CPU_FP32_rowMajor_TCSC_Uniform_G128_CS16_AVX512_OpenMP(const float* X, const int32_t NonZeroPerCol, const int32_t* row_index, float* result, const int N_COL, const int K);