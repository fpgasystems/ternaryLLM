#include "GEMM_CPU_INT8.hpp"

void GEMM_CPU_INT8_colMajor_TCSC_Merged_GroupMin_64xG4_AVX2_OpenMP(const int8_t* X, const int32_t* metadata, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K) {
#pragma omp parallel for
    for (int j = 0; j < N_COL / 4; j++) {
        const int* groupData = &metadata[j * 10];
        for (int i = 0; i < M_ROW; i += 64) {
            __m256i res00 = _mm256_setzero_si256();
            __m256i res01 = _mm256_setzero_si256();
            __m256i res10 = _mm256_setzero_si256();
            __m256i res11 = _mm256_setzero_si256();
            __m256i res20 = _mm256_setzero_si256();
            __m256i res21 = _mm256_setzero_si256();
            __m256i res30 = _mm256_setzero_si256();
            __m256i res31 = _mm256_setzero_si256();
            for (int k = groupData[0]; k < groupData[1]; k += 8) {
                __m256i pos00 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 0] * M_ROW + i + 0));
                __m256i pos10 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 1] * M_ROW + i + 0));
                __m256i pos20 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 2] * M_ROW + i + 0));
                __m256i pos30 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 3] * M_ROW + i + 0));
                __m256i pos01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 0] * M_ROW + i + 32));
                __m256i pos11 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 1] * M_ROW + i + 32));
                __m256i pos21 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 2] * M_ROW + i + 32));
                __m256i pos31 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 3] * M_ROW + i + 32));
                __m256i neg00 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 4] * M_ROW + i + 0));
                __m256i neg10 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 5] * M_ROW + i + 0));
                __m256i neg20 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 6] * M_ROW + i + 0));
                __m256i neg30 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 7] * M_ROW + i + 0));
                __m256i neg01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 4] * M_ROW + i + 32));
                __m256i neg11 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 5] * M_ROW + i + 32));
                __m256i neg21 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 6] * M_ROW + i + 32));
                __m256i neg31 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 7] * M_ROW + i + 32));
                res00 = _mm256_add_epi8(res00, _mm256_sub_epi8(pos00, neg00));
                res10 = _mm256_add_epi8(res10, _mm256_sub_epi8(pos10, neg10));
                res20 = _mm256_add_epi8(res20, _mm256_sub_epi8(pos20, neg20));
                res30 = _mm256_add_epi8(res30, _mm256_sub_epi8(pos30, neg30));
                res01 = _mm256_add_epi8(res01, _mm256_sub_epi8(pos01, neg01));
                res11 = _mm256_add_epi8(res11, _mm256_sub_epi8(pos11, neg11));
                res21 = _mm256_add_epi8(res21, _mm256_sub_epi8(pos21, neg21));
                res31 = _mm256_add_epi8(res31, _mm256_sub_epi8(pos31, neg31));
            }
            for (int k = groupData[1]; k < groupData[2]; k++) {
                res00 = _mm256_add_epi8(res00, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res01 = _mm256_add_epi8(res01, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[2]; k < groupData[3]; k++) {
                res00 = _mm256_sub_epi8(res00, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res01 = _mm256_sub_epi8(res01, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[3]; k < groupData[4]; k++) {
                res10 = _mm256_add_epi8(res10, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res11 = _mm256_add_epi8(res11, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[4]; k < groupData[5]; k++) {
                res10 = _mm256_sub_epi8(res10, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res11 = _mm256_sub_epi8(res11, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[5]; k < groupData[6]; k++) {
                res20 = _mm256_add_epi8(res20, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res21 = _mm256_add_epi8(res21, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[6]; k < groupData[7]; k++) {
                res20 = _mm256_sub_epi8(res20, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res21 = _mm256_sub_epi8(res21, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[7]; k < groupData[8]; k++) {
                res30 = _mm256_add_epi8(res30, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res31 = _mm256_add_epi8(res31, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            for (int k = groupData[8]; k < groupData[9]; k++) {
                res30 = _mm256_sub_epi8(res30, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 0)));
                res31 = _mm256_sub_epi8(res31, _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k] * M_ROW + i + 32)));
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 0) * M_ROW + i + 0), res00);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 0) * M_ROW + i + 32), res01);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 1) * M_ROW + i + 0), res10);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 1) * M_ROW + i + 32), res11);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 2) * M_ROW + i + 0), res20);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 2) * M_ROW + i + 32), res21);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 3) * M_ROW + i + 0), res30);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 3) * M_ROW + i + 32), res31);
        }
    }
}

void GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX2_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K) {
#pragma omp parallel for
    for (int j = 0; j < N_COL / 4; j++) {
        for (int i = 0; i < M_ROW; i += 64) {
            __m256i res00 = _mm256_setzero_si256();
            __m256i res01 = _mm256_setzero_si256();
            __m256i res10 = _mm256_setzero_si256();
            __m256i res11 = _mm256_setzero_si256();
            __m256i res20 = _mm256_setzero_si256();
            __m256i res21 = _mm256_setzero_si256();
            __m256i res30 = _mm256_setzero_si256();
            __m256i res31 = _mm256_setzero_si256();
            for (int k = j * 4 * NonZeroPerCol; k < (j + 1) * 4 * NonZeroPerCol; k += 8) {
                __m256i pos00 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 0] * M_ROW + i + 0));
                __m256i pos01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 0] * M_ROW + i + 32));
                __m256i pos10 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 1] * M_ROW + i + 0));
                __m256i pos11 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 1] * M_ROW + i + 32));
                __m256i pos20 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 2] * M_ROW + i + 0));
                __m256i pos21 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 2] * M_ROW + i + 32));
                __m256i pos30 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 3] * M_ROW + i + 0));
                __m256i pos31 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 3] * M_ROW + i + 32));
                __m256i neg00 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 4] * M_ROW + i + 0));
                __m256i neg01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 4] * M_ROW + i + 32));
                __m256i neg10 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 5] * M_ROW + i + 0));
                __m256i neg11 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 5] * M_ROW + i + 32));
                __m256i neg20 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 6] * M_ROW + i + 0));
                __m256i neg21 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 6] * M_ROW + i + 32));
                __m256i neg30 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 7] * M_ROW + i + 0));
                __m256i neg31 = _mm256_load_si256(reinterpret_cast<const __m256i*>(X + row_index[k + 7] * M_ROW + i + 32));
                res00 = _mm256_add_epi8(res00, _mm256_sub_epi8(pos00, neg00));
                res01 = _mm256_add_epi8(res01, _mm256_sub_epi8(pos01, neg01));
                res10 = _mm256_add_epi8(res10, _mm256_sub_epi8(pos10, neg10));
                res11 = _mm256_add_epi8(res11, _mm256_sub_epi8(pos11, neg11));
                res20 = _mm256_add_epi8(res20, _mm256_sub_epi8(pos20, neg20));
                res21 = _mm256_add_epi8(res21, _mm256_sub_epi8(pos21, neg21));
                res30 = _mm256_add_epi8(res30, _mm256_sub_epi8(pos30, neg30));
                res31 = _mm256_add_epi8(res31, _mm256_sub_epi8(pos31, neg31));
            }
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 0) * M_ROW + i + 0), res00);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 0) * M_ROW + i + 32), res01);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 1) * M_ROW + i + 0), res10);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 1) * M_ROW + i + 32), res11);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 2) * M_ROW + i + 0), res20);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 2) * M_ROW + i + 32), res21);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 3) * M_ROW + i + 0), res30);
            _mm256_store_si256(reinterpret_cast<__m256i*>(result + (j * 4 + 3) * M_ROW + i + 32), res31);
        }
    }
}

void GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP(const int8_t* X, const int32_t NonZeroPerCol, const int16_t* row_index, int8_t* result, const int M_ROW, const int N_COL, const int K) {
#pragma omp parallel for
    for (int j = 0; j < N_COL / 4; j++) {
        for (int i = 0; i < M_ROW; i += 64) {
            __m512i res00 = _mm512_setzero_si512();
            __m512i res10 = _mm512_setzero_si512();
            __m512i res20 = _mm512_setzero_si512();
            __m512i res30 = _mm512_setzero_si512();
            for (int k = j * 4 * NonZeroPerCol; k < (j + 1) * 4 * NonZeroPerCol; k += 8) {
                __m512i pos00 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 0] * M_ROW + i + 0));
                __m512i pos10 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 1] * M_ROW + i + 0));
                __m512i pos20 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 2] * M_ROW + i + 0));
                __m512i pos30 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 3] * M_ROW + i + 0));
                __m512i neg00 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 4] * M_ROW + i + 0));
                __m512i neg10 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 5] * M_ROW + i + 0));
                __m512i neg20 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 6] * M_ROW + i + 0));
                __m512i neg30 = _mm512_load_si512(reinterpret_cast<const __m512i*>(X + row_index[k + 7] * M_ROW + i + 0));
                res00 = _mm512_add_epi8(res00, _mm512_sub_epi8(pos00, neg00));
                res10 = _mm512_add_epi8(res10, _mm512_sub_epi8(pos10, neg10));
                res20 = _mm512_add_epi8(res20, _mm512_sub_epi8(pos20, neg20));
                res30 = _mm512_add_epi8(res30, _mm512_sub_epi8(pos30, neg30));
            }
            _mm512_store_si512(reinterpret_cast<__m512i*>(result + (j * 4 + 0) * M_ROW + i + 0), res00);
            _mm512_store_si512(reinterpret_cast<__m512i*>(result + (j * 4 + 1) * M_ROW + i + 0), res10);
            _mm512_store_si512(reinterpret_cast<__m512i*>(result + (j * 4 + 2) * M_ROW + i + 0), res20);
            _mm512_store_si512(reinterpret_cast<__m512i*>(result + (j * 4 + 3) * M_ROW + i + 0), res30);
        }
    }
}


