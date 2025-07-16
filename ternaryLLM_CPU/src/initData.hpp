#pragma once
#include<vector>
#include<random>
#include<iostream>
#include<immintrin.h>
#include<memory>
#include<chrono>
#include<cstdlib> 
#include<algorithm>
#include<cstring>
using namespace std;

/* Initialize the Activation X
* @param LEN:   the vector length
* @param Range: the initial values range from (-Range, +Range)
* Return the initialided random vector in given data type
*/
template <typename T>
vector<T> initX(int LEN, int Range) {
    vector<T> X(LEN, 0);
    mt19937 generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
    uniform_int_distribution<int> range(-Range, Range);
#pragma omp parallel for
    for (int i = 0; i < LEN; i++) {
        X[i] = range(generator);
    }
    return X;
};


/* Initialize the ternary weights at given sparsity
* @param ROW
* @param COL
* @param Sparsity:  overall persentage of zeros in the matrix
* @param Variation: the variation of non-zero values at each column
* @param Aligned:   if same number of +1 and -1 at a column, but different columns can have different number of +1/-1 
* @param Uniformed: if all columns have the same number of +1 and -1, and the non-zero values are uniformly fdistributed within each column
* Return a ternary weight matrix in gicen data type
*/
template <typename T>
vector<T> sparseWeight(int ROW, int COL, float Sparsity, float Variation, bool Aligned, bool Uniformed) {
    vector<T> W(ROW * COL, 0); // Normal row-major matrix, but the data are initialized by each column
    mt19937 generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

    // Generate Aligned Weights:  # of +1 and -1 are the same in a Column
    if (Aligned) { // same +1 and -1 in one column.
        int nonZero = ROW * (1 - Sparsity); // Non-zero per column
        uniform_int_distribution<int> alignRange(0, ROW - 1);
        uniform_int_distribution<int> variRange(0, max(int(ROW * Variation - 1), 0)); // The variation among different columns

        // # of +1 and -1 are the same in a Column, but two Columns may differ. For example, col 0 may have 50 +1 and 50 -1, but col 1 may have 60 +1 and 60 -1
        if (Variation > 0) {
#pragma omp parallel for
            for (int i = 0; i < COL / 2; i++) {
                int vari = variRange(generator);
                int col0Pos = nonZero / 2 + vari;
                int col0Neg = nonZero / 2 + vari;
                int col1Pos = nonZero / 2 - vari;
                int col1Neg = nonZero / 2 - vari;
                // First column: (posOrNegPerCol + vari) +1 and  (posOrNegPerCol + vari)  -1
                int num = 0;
                while (num < col0Pos) {
                    int index = alignRange(generator);
                    if (W[index * COL + i * 2 + 0] == 0) {
                        W[index * COL + i * 2 + 0] = +1;
                        num++;
                    }
                }
                num = 0;
                while (num < col0Neg) {
                    int index = alignRange(generator);
                    if (W[index * COL + i * 2 + 0] == 0) {
                        W[index * COL + i * 2 + 0] = -1;
                        num++;
                    }
                }
                // Second column: (posOrNegPerCol - vari) +1 and  (posOrNegPerCol - vari)  -1
                num = 0;
                while (num < col1Pos) {
                    int index = alignRange(generator);
                    if (W[index * COL + i * 2 + 1] == 0) {
                        W[index * COL + i * 2 + 1] = +1;
                        num++;
                    }
                }
                num = 0;
                while (num < col1Neg) {
                    int index = alignRange(generator);
                    if (W[index * COL + i * 2 + 1] == 0) {
                        W[index * COL + i * 2 + 1] = -1;
                        num++;
                    }
                }
            }
        }
        // # of +1 and -1 are the same in all Columns. For example, if col 0 have 50 +1 and 50 -1, col 1 must have 50 +1 and 50 -1
        else {
#pragma omp parallel for
            for (int i = 0; i < COL; i++) {
                int num = 0;
                while (num < nonZero / 2) {
                    int index = alignRange(generator);
                    if (W[index * COL + i] == 0) {
                        W[index * COL + i] = +1;
                        num++;
                    }
                }
                num = 0;
                while (num < nonZero / 2) {
                    int index = alignRange(generator);
                    if (W[index * COL + i] == 0) {
                        W[index * COL + i] = -1;
                        num++;
                    }
                }
            }
        }
    }
    // # of +1 and -1 are the same in all Columns. In addition, there will be one +1 and one -1 in every small bock (block size = StepSice below)  
    else if (Uniformed) {
        int steps = ROW * (1 - Sparsity) / 2;
        int stepSize = ROW / steps;
        uniform_int_distribution<int> uniformRange(0, stepSize - 1);
#pragma omp parallel for
        for (int j = 0; j < COL; j++) {
            for (int i = 0; i < steps; i++) {
                // +1
                int index = uniformRange(generator) + i * stepSize;
                W[index * COL + j] = +1;
                // -1
                index = uniformRange(generator) + i * stepSize;
                while (W[index * COL + j] != 0) {
                    index = uniformRange(generator) + i * stepSize;
                }
                W[index * COL + j] = -1;
            }
        }
    }
    // General Sparsity. Every column may have different numbers of +1 and -1
    else {
        int nonZero = ROW * (1 - Sparsity);
        uniform_int_distribution<int> range(0, ROW - 1);
        uniform_int_distribution<int> dynamicCOL(0, min(int(Variation * ROW) + 1, ROW - nonZero - 2));
        uniform_int_distribution<int> dynamicPosNeg(0, int(Variation * ROW * (1 - Sparsity)) + 1);
#pragma omp parallel for
        for (int j = 0; j < COL; j += 2) {
            // The variation is biased on +1 for easier debug, e.g., there will always more +1 than -1 at each column
            int colVari = dynamicCOL(generator);
            int posVari = dynamicPosNeg(generator);
            int col0Pos = (nonZero + colVari) / 2 + posVari;
            int col0Neg = (nonZero + colVari) / 2 - posVari;
            int col1Pos = (nonZero - colVari) / 2 + posVari;
            int col1Neg = (nonZero - colVari) / 2 - posVari;
            int num = 0;
            while (num < col0Pos) {
                int i = range(generator);
                if (W[i * COL + j] == 0) {
                    W[i * COL + j] = +1;
                    num++;
                }
            }
            num = 0;
            while (num < col0Neg) {
                int i = range(generator);
                if (W[i * COL + j] == 0) {
                    W[i * COL + j] = -1;
                    num++;
                }
            }
            num = 0;
            while (num < col1Pos) {
                int i = range(generator);
                if (W[i * COL + j + 1] == 0) {
                    W[i * COL + j + 1] = +1;
                    num++;
                }
            }
            num = 0;
            while (num < col1Neg) {
                int i = range(generator);
                if (W[i * COL + j + 1] == 0) {
                    W[i * COL + j + 1] = -1;
                    num++;
                }
            }
        }
    }

    return W;
};


//********************* Simple SiLU Implementations: SiLU(X), SiLU(X) dot XB **************************************

template <typename T>
void Naive_SiLU(int LEN, T* X) { 
#pragma omp parallel for simd
    for (int i = 0; i < LEN; i++) {
        X[i] = X[i] / (1 + std::exp(-X[i]));
    } 
};

// Fused SiLU and pointwise matrix multiplication: SiLU(X) dot XB 
template <typename T>
void Naive_SiLU_Dot(int LEN, T* X, const T* XB) {
#pragma omp parallel for
    for (int i = 0; i < LEN; i++) { 
        X[i] = X[i] * XB[i] / (1 + std::exp(-X[i]));
    }
};

template <typename T>
void Naive_SiLU_Dot_Unroll(int LEN, T* X, const T* XB) {
    const int STEP = 16;
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
        X[i +  0] *= XB[i +  0] / (1 + std::exp(-X[i +  0]));
        X[i +  1] *= XB[i +  1] / (1 + std::exp(-X[i +  1]));
        X[i +  2] *= XB[i +  2] / (1 + std::exp(-X[i +  2]));
        X[i +  3] *= XB[i +  3] / (1 + std::exp(-X[i +  3]));
        X[i +  4] *= XB[i +  4] / (1 + std::exp(-X[i +  4]));
        X[i +  5] *= XB[i +  5] / (1 + std::exp(-X[i +  5]));
        X[i +  6] *= XB[i +  6] / (1 + std::exp(-X[i +  6]));
        X[i +  7] *= XB[i +  7] / (1 + std::exp(-X[i +  7]));
        X[i +  8] *= XB[i +  8] / (1 + std::exp(-X[i +  8]));
        X[i +  9] *= XB[i +  9] / (1 + std::exp(-X[i +  9]));
        X[i + 10] *= XB[i + 10] / (1 + std::exp(-X[i + 10]));
        X[i + 11] *= XB[i + 11] / (1 + std::exp(-X[i + 11]));
        X[i + 12] *= XB[i + 12] / (1 + std::exp(-X[i + 12]));
        X[i + 13] *= XB[i + 13] / (1 + std::exp(-X[i + 13]));
        X[i + 14] *= XB[i + 14] / (1 + std::exp(-X[i + 14]));
        X[i + 15] *= XB[i + 15] / (1 + std::exp(-X[i + 15]));       
    }
};

template <typename T>
void Naive_SiLU_Dot_Unroll_LCS(int LEN, T* X, const T* XB) {
    const int STEP = 16;
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
        T a0 = X[i + 0];
        T a1 = X[i + 1];
        T a2 = X[i + 2];
        T a3 = X[i + 3];
        T a4 = X[i + 4];
        T a5 = X[i + 5];
        T a6 = X[i + 6];
        T a7 = X[i + 7];
        T a8 = X[i + 8];
        T a9 = X[i + 9];
        T a10 = X[i + 10];
        T a11 = X[i + 11];
        T a12 = X[i + 12];
        T a13 = X[i + 13];
        T a14 = X[i + 14];
        T a15 = X[i + 15];
        T b0 = XB[i + 0];
        T b1 = XB[i + 1];
        T b2 = XB[i + 2];
        T b3 = XB[i + 3];
        T b4 = XB[i + 4];
        T b5 = XB[i + 5];
        T b6 = XB[i + 6];
        T b7 = XB[i + 7];
        T b8 = XB[i + 8];
        T b9 = XB[i + 9];
        T b10 = XB[i + 10];
        T b11 = XB[i + 11];
        T b12 = XB[i + 12];
        T b13 = XB[i + 13];
        T b14 = XB[i + 14];
        T b15 = XB[i + 15];
        T d0 = 1 + std::exp(-a0);
        T d1 = 1 + std::exp(-a1);
        T d2 = 1 + std::exp(-a2);
        T d3 = 1 + std::exp(-a3);
        T d4 = 1 + std::exp(-a4);
        T d5 = 1 + std::exp(-a5);
        T d6 = 1 + std::exp(-a6);
        T d7 = 1 + std::exp(-a7);
        T d8 = 1 + std::exp(-a8);
        T d9 = 1 + std::exp(-a9);
        T d10 = 1 + std::exp(-a10);
        T d11 = 1 + std::exp(-a11);
        T d12 = 1 + std::exp(-a12);
        T d13 = 1 + std::exp(-a13);
        T d14 = 1 + std::exp(-a14);
        T d15 = 1 + std::exp(-a15);
        a0 = a0 * b0 / d0;
        a1 = a1 * b1 / d1;
        a2 = a2 * b2 / d2;
        a3 = a3 * b3 / d3;
        a4 = a4 * b4 / d4;
        a5 = a5 * b5 / d5;
        a6 = a6 * b6 / d6;
        a7 = a7 * b7 / d7;
        a8 = a8 * b8 / d8;
        a9 = a9 * b9 / d9;
        a10 = a10 * b10 / d10;
        a11 = a11 * b11 / d11;
        a12 = a12 * b12 / d12;
        a13 = a13 * b13 / d13;
        a14 = a14 * b14 / d14;
        a15 = a15 * b15 / d15;
        X[i + 0] = a0;
        X[i + 1] = a1;
        X[i + 2] = a2;
        X[i + 3] = a3;
        X[i + 4] = a4;
        X[i + 5] = a5;
        X[i + 6] = a6;
        X[i + 7] = a7;
        X[i + 8] = a8;
        X[i + 9] = a9;
        X[i + 10] = a10;
        X[i + 11] = a11;
        X[i + 12] = a12;
        X[i + 13] = a13;
        X[i + 14] = a14;
        X[i + 15] = a15;
    }
};

template <typename T>
void Naive_SiLU_Dot_STEP(int LEN, T* X, const T* XB) {
    const int STEP = 16;
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
#pragma omp simd
        for (int j = 0; j < STEP; j++) {
            T div = 1 + std::exp(-X[i + j]);
            X[i + j] = X[i + j] * XB[i + j] / div;
        }
    }
};

template <typename T>
void Naive_SiLU_Dot_Unroll_SIMD(int LEN, T* X, const T* XB) {
    const int STEP = 256;
    const int Unroll = 16;
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
#pragma omp simd
        for (int j = 0; j < STEP; j+=Unroll) {
            X[i + j + 0] *= XB[i + j + 0] / (1 + std::exp(-X[i + j + 0]));
            X[i + j + 1] *= XB[i + j + 1] / (1 + std::exp(-X[i + j + 1]));
            X[i + j + 2] *= XB[i + j + 2] / (1 + std::exp(-X[i + j + 2]));
            X[i + j + 3] *= XB[i + j + 3] / (1 + std::exp(-X[i + j + 3]));
            X[i + j + 4] *= XB[i + j + 4] / (1 + std::exp(-X[i + j + 4]));
            X[i + j + 5] *= XB[i + j + 5] / (1 + std::exp(-X[i + j + 5]));
            X[i + j + 6] *= XB[i + j + 6] / (1 + std::exp(-X[i + j + 6]));
            X[i + j + 7] *= XB[i + j + 7] / (1 + std::exp(-X[i + j + 7]));
            X[i + j + 8] *= XB[i + j + 8] / (1 + std::exp(-X[i + j + 8]));
            X[i + j + 9] *= XB[i + j + 9] / (1 + std::exp(-X[i + j + 9]));
            X[i + j + 10] *= XB[i + j + 10] / (1 + std::exp(-X[i + j + 10]));
            X[i + j + 11] *= XB[i + j + 11] / (1 + std::exp(-X[i + j + 11]));
            X[i + j + 12] *= XB[i + j + 12] / (1 + std::exp(-X[i + j + 12]));
            X[i + j + 13] *= XB[i + j + 13] / (1 + std::exp(-X[i + j + 13]));
            X[i + j + 14] *= XB[i + j + 14] / (1 + std::exp(-X[i + j + 14]));
            X[i + j + 15] *= XB[i + j + 15] / (1 + std::exp(-X[i + j + 15]));
        }
    }
};

template <typename T>
void Naive_SiLU_Dot_Unroll_SIMD_LCS(int LEN, T* X, const T* XB) {
    const int STEP = 256;
    const int Unroll = 16;
#pragma omp parallel for 
    for (int i = 0; i < LEN; i += STEP) {
#pragma omp simd
        for (int j = 0; j < STEP; j += Unroll) {
            T a0 = X[i + j + 0];
            T a1 = X[i + j + 1];
            T a2 = X[i + j + 2];
            T a3 = X[i + j + 3];
            T a4 = X[i + j + 4];
            T a5 = X[i + j + 5];
            T a6 = X[i + j + 6];
            T a7 = X[i + j + 7];
            T a8 = X[i + j + 8];
            T a9 = X[i + j + 9];
            T a10 = X[i + j + 10];
            T a11 = X[i + j + 11];
            T a12 = X[i + j + 12];
            T a13 = X[i + j + 13];
            T a14 = X[i + j + 14];
            T a15 = X[i + j + 15];
            T b0 = XB[i + j + 0];
            T b1 = XB[i + j + 1];
            T b2 = XB[i + j + 2];
            T b3 = XB[i + j + 3];
            T b4 = XB[i + j + 4];
            T b5 = XB[i + j + 5];
            T b6 = XB[i + j + 6];
            T b7 = XB[i + j + 7];
            T b8 = XB[i + j + 8];
            T b9 = XB[i + j + 9];
            T b10 = XB[i + j + 10];
            T b11 = XB[i + j + 11];
            T b12 = XB[i + j + 12];
            T b13 = XB[i + j + 13];
            T b14 = XB[i + j + 14];
            T b15 = XB[i + j + 15];
            T d0 = std::exp(-a0);
            T d1 = std::exp(-a1);
            T d2 = std::exp(-a2);
            T d3 = std::exp(-a3);
            T d4 = std::exp(-a4);
            T d5 = std::exp(-a5);
            T d6 = std::exp(-a6);
            T d7 = std::exp(-a7);
            T d8 = std::exp(-a8);
            T d9 = std::exp(-a9);
            T d10 = std::exp(-a10);
            T d11 = std::exp(-a11);
            T d12 = std::exp(-a12);
            T d13 = std::exp(-a13);
            T d14 = std::exp(-a14);
            T d15 = std::exp(-a15);
            d0 = d0 + 1;
            d1 = d1 + 1;
            d2 = d2 + 1;
            d3 = d3 + 1;
            d4 = d4 + 1;
            d5 = d5 + 1;
            d6 = d6 + 1;
            d7 = d7 + 1;
            d8 = d8 + 1;
            d9 = d9 + 1;
            d10 = d10 + 1;
            d11 = d11 + 1;
            d12 = d12 + 1;
            d13 = d13 + 1;
            d14 = d14 + 1;
            d15 = d15 + 1;
            a0 = a0 * b0 / d0;
            a1 = a1 * b1 / d1;
            a2 = a2 * b2 / d2;
            a3 = a3 * b3 / d3;
            a4 = a4 * b4 / d4;
            a5 = a5 * b5 / d5;
            a6 = a6 * b6 / d6;
            a7 = a7 * b7 / d7;
            a8 = a8 * b8 / d8;
            a9 = a9 * b9 / d9;
            a10 = a10 * b10 / d10;
            a11 = a11 * b11 / d11;
            a12 = a12 * b12 / d12;
            a13 = a13 * b13 / d13;
            a14 = a14 * b14 / d14;
            a15 = a15 * b15 / d15;
            X[i + j + 0] = a0;
            X[i + j + 1] = a1;
            X[i + j + 2] = a2;
            X[i + j + 3] = a3;
            X[i + j + 4] = a4;
            X[i + j + 5] = a5;
            X[i + j + 6] = a6;
            X[i + j + 7] = a7;
            X[i + j + 8] = a8;
            X[i + j + 9] = a9;
            X[i + j + 10] = a10;
            X[i + j + 11] = a11;
            X[i + j + 12] = a12;
            X[i + j + 13] = a13;
            X[i + j + 14] = a14;
            X[i + j + 15] = a15;
        }
    }
};


void Naive_SiLU_Dot_Unroll_AVX2(int LEN, float* X, const float* XB);


void Naive_SiLU_Dot_Unroll_AVX512(int LEN, float* X, const float* XB);


//********************* A simple Matrix Transposing Implementation **************************************

/* Transpose the matrix X and return a new transposed matrix XT */
template <typename T>
vector<T> transposeVector(T* X, int ROW, int COL) {
    vector<T> XT(ROW * COL, 0);
#pragma omp parallel for
    for (int i = 0; i < ROW; i++) {
#pragma omp simd
        for (int j = 0; j < COL; j++) {
            XT[j * ROW + i] = X[i * COL + j];
        }
    }
    return XT;
};

void FastMatrixTranspose(const float* X, float* XT, const int H, const int W);


//********************* A simple data comparing to verify the results **************************************

template <typename T>
bool compare_results(T* result, T* groundTruth, int H, int W) {
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int i = h * W + w;
            if (abs(result[i] - groundTruth[i]) > 10e-6) {
                cout << "Error at: H=" << h << ", W=" << w << ", result=" << result[i] << ", groundTruth=" << groundTruth[i] << endl;
                return false;
            }
        }
    }
    return true;
};

template <typename T>
bool compare_results(T* result, T* groundTruth, int H, float mini) {
    for (int i = 0; i < H; i++) {
        if (abs(result[i] - groundTruth[i]) > mini) {
            cout << "Error at: H=" << i << ", result=" << result[i] << ", groundTruth=" << groundTruth[i] << endl;
            return false;
        }
    }
    return true;
};