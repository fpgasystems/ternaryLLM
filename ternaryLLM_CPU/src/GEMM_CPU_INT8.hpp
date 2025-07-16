#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <tuple>
#include <sstream> 
#include <immintrin.h>

void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_32xG1_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_32xG4_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_128xG4_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_128xG8_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_128xG16_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_128xG32_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);

void GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX2_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG4_AVX2_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG8_AVX2_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG16_AVX2_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG32_AVX2_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);

void GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG4_AVX512_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG8_AVX512_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG16_AVX512_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
void GEMM_CPU_INT8_colMajor_TCSC_Uniform_128xG32_AVX512_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K);
