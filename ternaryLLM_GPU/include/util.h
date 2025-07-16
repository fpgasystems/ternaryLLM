#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstring>

#define INFO(msg) do {     \
    std::cout << "[INFO]: " << msg << std::endl;    \
} while (0)

#define ERROR(msg) do {     \
    std::cerr << "[ERROR]: " << msg << std::endl;   \
} while (0)

template<typename XType>
struct TerSparseDataWrap {
public:
    TerSparseDataWrap(int rows, int columns, int inners, float sparsity): rows(rows), columns(columns), inners(inners), sparsity(sparsity),
                                                        size_x(rows*inners), size_w1(columns*inners), size_res(rows*columns) {
        // allocate space for host data
        host_x.resize(size_x,0);
        host_w1.resize(size_w1,0);
        host_w1_xtype.resize(size_w1,0);
        host_res.resize(size_res,0);
    }

    ~TerSparseDataWrap() {
        // free(w1_merged_row_indice);
        // free(w1_merged_col_offset);
        // free(w1_merged_col_offset_tiled);
        // free(w1_merged_row_indice_tiled);
        // free(w1_neg_col_offset_tiled);
        // free(w1_pos_col_offset_tiled);
        // free(w1_neg_row_indice);
        // free(w1_pos_row_indice);
        // free(w1_neg_col_offset);
        // free(w1_pos_col_offset);
        // free(w1_neg_col_offset_pad);
        // free(w1_pos_col_offset_pad);
        // free(w1_row_indice);
        // free(w1_col_offset);
        // free(w1_values);
    }

    void duplicate_w() {
        for (int i = 0; i < size_w1; i++)
            host_w1_xtype.push_back((float)host_w1.data()[i]);
    }   

    void inter_tile_swizzle(int columns, int tile_num, int32_t* col_tile_indices) {
        for (int i = 0; i < columns; i++)
            std::iota(col_tile_indices + i*tile_num, col_tile_indices + (i+1)*tile_num, 0);
    }

    void inter_tile_swizzle(int columns, int tile_num, int32_t* col_offsets, int32_t* col_tile_indices) {
        for (int i = 0; i < columns; i++) {
            std::vector<int> swizzle_staging(tile_num);
            std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

            // col_offsets += i*tile_num*4;

            std::sort(swizzle_staging.begin(), swizzle_staging.end(),
                [i, tile_num, &col_offsets](int idx_a, int idx_b) {
                int length_a_interleaved = col_offsets[i*tile_num*4 + idx_a*4 + 1] - col_offsets[i*tile_num*4 + idx_a*4 + 0];
                int length_a_remain = col_offsets[idx_a*4 + 2] - col_offsets[idx_a*4 + 1];
                int length_b_interleaved = col_offsets[i*tile_num*4 + idx_b*4 + 1] - col_offsets[i*tile_num*4 + idx_b*4];
                int length_b_remain = col_offsets[idx_b*4 + 2] - col_offsets[idx_b*4 + 1];

                return length_a_interleaved+length_a_remain < length_b_interleaved+length_b_remain;
            });

            std::memcpy(col_tile_indices+i*tile_num, swizzle_staging.data(), sizeof(int) * tile_num);
        }
    }

    void inter_col_swizzle(int columns, int32_t* col_indices) {
        std::iota(col_indices, col_indices + columns, 0);
    }

    void inter_col_swizzle(int columns, const int32_t* col_offsets,  int32_t* col_indices) {
        std::vector<int> swizzle_staging(columns);
        std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);
        /* Argsort the col indices based on their length. */
        std::sort(swizzle_staging.begin(), swizzle_staging.end(),
                    [&col_offsets](int idx_a, int idx_b) {
                    int length_a_interleaved = col_offsets[idx_a*4 + 1] - col_offsets[idx_a*4];
                    int length_a_remain = col_offsets[idx_a*4 + 2] - col_offsets[idx_a*4 + 1];
                    int length_b_interleaved = col_offsets[idx_b*4 + 1] - col_offsets[idx_b*4];
                    int length_b_remain = col_offsets[idx_b*4 + 2] - col_offsets[idx_b*4 + 1];

                    // Interleave bits of a
                    // length_a_interleaved = (length_a_interleaved | (length_a_interleaved << 8)) & 0x00FF00FF;
                    // length_a_interleaved = (length_a_interleaved | (length_a_interleaved << 4)) & 0x0F0F0F0F;
                    // length_a_interleaved = (length_a_interleaved | (length_a_interleaved << 2)) & 0x33333333;
                    // length_a_interleaved = (length_a_interleaved | (length_a_interleaved << 1)) & 0x55555555;

                    // length_a_remain = (length_a_remain | (length_a_remain << 8)) & 0x00FF00FF;
                    // length_a_remain = (length_a_remain | (length_a_remain << 4)) & 0x0F0F0F0F;
                    // length_a_remain = (length_a_remain | (length_a_remain << 2)) & 0x33333333;
                    // length_a_remain = (length_a_remain | (length_a_remain << 1)) & 0x55555555;

                    // uint32_t a_code = (length_a_interleaved << 1) | length_a_remain;

                    // // Interleave bits of a
                    // length_b_interleaved = (length_b_interleaved | (length_b_interleaved << 8)) & 0x00FF00FF;
                    // length_b_interleaved = (length_b_interleaved | (length_b_interleaved << 4)) & 0x0F0F0F0F;
                    // length_b_interleaved = (length_b_interleaved | (length_b_interleaved << 2)) & 0x33333333;
                    // length_b_interleaved = (length_b_interleaved | (length_b_interleaved << 1)) & 0x55555555;

                    // length_b_remain = (length_b_remain | (length_b_remain << 8)) & 0x00FF00FF;
                    // length_b_remain = (length_b_remain | (length_b_remain << 4)) & 0x0F0F0F0F;
                    // length_b_remain = (length_b_remain | (length_b_remain << 2)) & 0x33333333;
                    // length_b_remain = (length_b_remain | (length_b_remain << 1)) & 0x55555555;

                    // uint32_t b_code = (length_a_interleaved << 1) | length_a_remain;

                    if ((length_a_interleaved/32) != (length_b_interleaved/32))
                        return (length_a_interleaved/32) < (length_b_interleaved/32);
                    
                    if ((length_a_interleaved%32) != (length_b_interleaved%32))
                        return (length_a_interleaved%32) < (length_b_interleaved%32);
                    
                    if ((length_a_remain/32) != (length_b_remain/32))
                        return (length_a_remain/32) < (length_b_remain/32);
                    
                    return (length_a_remain%32) < (length_b_remain%32);
                });
        
        std::memcpy(col_indices, swizzle_staging.data(), sizeof(int) * columns);

        // for (int i = 0; i < columns; i++) {
        //     std::cout << col_offsets[col_indices[i]*4 + 1] - col_offsets[col_indices[i]*4] << ":";
        //     std::cout << col_offsets[col_indices[i]*4 + 2] - col_offsets[col_indices[i]*4 + 1] << ", ";
        // }
        // std::cout << "\n";
    }

    /**
     * compress sparse matrix into ternary csc format
     */
    void compress_ter_csc() {
        size_csc_col_offset = columns+1;
        w1_neg_row_indice = (int *)calloc(w1_cnt_neg, sizeof(int));
        w1_pos_row_indice = (int *)calloc(w1_cnt_pos, sizeof(int));
        w1_neg_col_offset = (int *)calloc(size_csc_col_offset, sizeof(int));
        w1_pos_col_offset = (int *)calloc(size_csc_col_offset, sizeof(int));

        w1_neg_col_offset_pad = (int *)calloc(columns*2, sizeof(int));
        w1_pos_col_offset_pad = (int *)calloc(columns*2, sizeof(int));

        //Compress W1
        int nb_neg = 0;
        int nb_pos = 0;
        for (int y = 0; y < columns; y++) {
            w1_neg_col_offset[y] = nb_neg;
            w1_pos_col_offset[y] = nb_pos;
            for (int x = 0; x < inners; x++) {
                if(host_w1.data()[x * columns + y] == -1){
                    w1_neg_row_indice[nb_neg] = x;
                    nb_neg++;
                }
                else if(host_w1.data()[x * columns + y] == 1){
                    w1_pos_row_indice[nb_pos] = x;
                    nb_pos++;
                }
            }
        }
        w1_neg_col_offset[columns] = nb_neg;
        w1_pos_col_offset[columns] = nb_pos;

        /* compress padding col offset */
        w1_neg_col_offset_pad[0] = w1_neg_col_offset[0];
        w1_pos_col_offset_pad[0] = w1_pos_col_offset[0];
        w1_neg_col_offset_pad[columns*2-1] = w1_neg_col_offset[columns];
        w1_pos_col_offset_pad[columns*2-1] = w1_pos_col_offset[columns];
        for (int i = 1; i < columns; i++) {
            w1_neg_col_offset_pad[i*2-1] = w1_neg_col_offset[i];
            w1_neg_col_offset_pad[i*2] = w1_neg_col_offset[i];
            w1_pos_col_offset_pad[i*2-1] = w1_pos_col_offset[i];
            w1_pos_col_offset_pad[i*2] = w1_pos_col_offset[i];
        }

        return;
    }

    /**
     * compress sparse matrix into ternary merged csc format
     */
    void compress_merged_ter_csc() {
        compress_ter_csc();

        size_csc_col_offset_merged = columns*4;
        w1_cnt_nnz = w1_cnt_neg + w1_cnt_pos;
        // std::cout << w1_cnt_nnz << "\n";
        w1_merged_row_indice = (int*)calloc(w1_cnt_nnz, sizeof(int));
        w1_merged_col_offset = (int*)calloc(columns*4, sizeof(int));


        int prev_cnt = 0;
        int prev_nnz = 0;
        for (int i = 0; i < columns; i++) {
            int neg_start = w1_neg_col_offset[i];
            int neg_end = w1_neg_col_offset[i+1];
            int pos_start = w1_pos_col_offset[i];
            int pos_end = w1_pos_col_offset[i+1];

            int neg = neg_end - neg_start;
            int pos = pos_end - pos_start;
            bool flag_min_neg = neg <= pos;
            int min_neg_pos = flag_min_neg ? neg : pos;
            int max_neg_pos = flag_min_neg ? pos : neg;

            int cnt_interleave = min_neg_pos*2;
            for (int k = 0; k < min_neg_pos; k++) {
                w1_merged_row_indice[prev_nnz + k*2 + 0] = w1_pos_row_indice[pos_start + k]; 
                w1_merged_row_indice[prev_nnz + k*2 + 1] = w1_neg_row_indice[neg_start + k];
            }
            

            int cnt_remain = max_neg_pos - min_neg_pos;
            for (int k = min_neg_pos; k < max_neg_pos; k++) {
                // positive number remains
                if (flag_min_neg) {
                    w1_merged_row_indice[prev_nnz + k + min_neg_pos] = w1_pos_row_indice[pos_start + k];
                }
                // negative number remains
                else {
                    w1_merged_row_indice[prev_nnz + k + min_neg_pos] = w1_neg_row_indice[neg_start + k];
                }
            }

            w1_merged_col_offset[i*4 + 0] = prev_cnt;
            w1_merged_col_offset[i*4 + 1] = prev_cnt + cnt_interleave;
            w1_merged_col_offset[i*4 + 2] = prev_cnt + cnt_interleave + cnt_remain;
            w1_merged_col_offset[i*4 + 3] = flag_min_neg ? 1 : -1;
            // std::cout << prev_cnt << ":" << prev_cnt + cnt_interleave << ":" << prev_cnt + cnt_interleave + cnt_remain << ":" << w1_merged_col_offset[i*4 + 3] << "\n";
            prev_cnt += cnt_interleave + cnt_remain;
            prev_nnz += neg + pos;
        }

        /* */
        inter_col_swizzle(columns, w1_merged_col_offset, col_indices_);
    }

    /**
     * compress sparse matrix into ternary tiled csc format with tile size TILE_K
     */
    template<int TILE_K> void compress_tiled_csc() {
        compress_ter_csc();

        size_csc_col_offset_tiled = columns*ceil(1.*inners/TILE_K)*2;
        w1_neg_col_offset_tiled = (int *)calloc(size_csc_col_offset_tiled, sizeof(int));
        w1_pos_col_offset_tiled = (int *)calloc(size_csc_col_offset_tiled, sizeof(int));

        /* compress tiled col offset */
        int nb_neg = 0;
        int nb_pos = 0;
        int tileNumK = ceil(1.*inners/TILE_K);
        int col_offset_x = tileNumK*2;  // each tile needs 2 offset [start, end)
        // compress column
        for (int y = 0; y < columns; y++) {
            int* neg_col_offset_start = w1_neg_col_offset_tiled + y*col_offset_x;
            int* pos_col_offset_start = w1_pos_col_offset_tiled + y*col_offset_x;
            // compress tile
            for (int tileId = 0; tileId < tileNumK; tileId++) {
                int offset = tileId*TILE_K;
                neg_col_offset_start[tileId*2 + 0] = nb_neg;
                pos_col_offset_start[tileId*2 + 0] = nb_pos;
                for (int x = 0; x < TILE_K; x++) {
                    int id = (tileId*TILE_K+x)*columns + y;
                    if (host_w1.data()[id] == -1) {
                        w1_neg_row_indice[nb_neg] -= offset;
                        nb_neg++;
                    } else if (host_w1.data()[id] == 1) {
                        w1_pos_row_indice[nb_pos] -= offset;
                        nb_pos++;
                    }
                }
                neg_col_offset_start[tileId*2 + 1] = nb_neg;
                pos_col_offset_start[tileId*2 + 1] = nb_pos;
            }
        }
    }

    /**
     * compress sparse matrix into tiled and merged csc format with tile size TILE_K
     */
    template<int TILE_K> void compress_tiled_mcsc() {
        if (tile_merged)    return;
        compress_tiled_csc<TILE_K>();
        tile_merged = true;
        // print<int, int>(columns, 2*ceil(1.*inners/TILE_K), w1_neg_col_offset_tiled);

        // std::cout << "start compress\n";
        size_mcsc_col_offset_tiled = columns*4*ceil(1.*inners/TILE_K);
        w1_cnt_nnz = w1_cnt_neg + w1_cnt_pos;
        w1_merged_row_indice_tiled = (int*)calloc(w1_cnt_nnz, sizeof(int));
        w1_merged_col_offset_tiled = (int*)calloc(size_mcsc_col_offset_tiled, sizeof(int));
        col_tile_indices_ = (int*)calloc(size_mcsc_col_offset_tiled/4, sizeof(int));

        /* compress merged tiled col offset */
        int tileNumK = ceil(1.*inners/TILE_K);  
        int col_offset_tiled_csc = tileNumK*2;  // each tile needs 2 offset [start, end) in tiled csc
        int col_offset_tiled_mcsc = tileNumK*4; // each tile needs 4 offset [pos_neg_start, pos_neg_end, common_end, sign] in tiled mcsc
        int prev_cnt = 0;
        int prev_nnz = 0;
        // compress column
        for (int y = 0; y < columns; y++) {
            // compress tile
            for (int tileId = 0; tileId < tileNumK; tileId++) {
                int tile_offset = tileId*TILE_K;
                int tile_start_id = y*col_offset_tiled_csc + 2*tileId;
                
                int tile_neg_start = w1_neg_col_offset_tiled[tile_start_id];
                int tile_neg_end = w1_neg_col_offset_tiled[tile_start_id+1];
                int tile_pos_start = w1_pos_col_offset_tiled[tile_start_id];
                int tile_pos_end = w1_pos_col_offset_tiled[tile_start_id+1];

                int neg = tile_neg_end - tile_neg_start;
                int pos = tile_pos_end - tile_pos_start;
                bool flag_min_neg = neg <= pos;
                int min_neg_pos = flag_min_neg ? neg : pos;
                int max_neg_pos = flag_min_neg ? pos : neg;

                
                /* compress interleaved number */
                int cnt_interleave = min_neg_pos*2;
                for (int k = 0; k < min_neg_pos; k++) {
                    w1_merged_row_indice_tiled[prev_nnz + k*2 + 0] = w1_pos_row_indice[tile_pos_start + k]; 
                    w1_merged_row_indice_tiled[prev_nnz + k*2 + 1] = w1_neg_row_indice[tile_neg_start + k];
                }

                
                /* compress common number */
                int cnt_remain = max_neg_pos - min_neg_pos;
                for (int k = min_neg_pos; k < max_neg_pos; k++) {
                    // positive number remains
                    if (flag_min_neg) {
                        w1_merged_row_indice_tiled[prev_nnz + k + min_neg_pos] = w1_pos_row_indice[tile_pos_start + k];
                    }
                    // negative number remains
                    else {
                        w1_merged_row_indice_tiled[prev_nnz + k + min_neg_pos] = w1_neg_row_indice[tile_neg_start + k];
                    }
                }

                if (cnt_remain%2 == 1) {
                    cnt_remain--;
                }

                if (cnt_interleave%4 != 0) {
                    cnt_interleave-=2;
                }

                if (cnt_remain%4 != 0) {
                    cnt_remain-=2;
                }

                
                tile_start_id = y*col_offset_tiled_mcsc + 4*tileId;
                w1_merged_col_offset_tiled[tile_start_id + 0] = prev_cnt;
                w1_merged_col_offset_tiled[tile_start_id + 1] = prev_cnt + cnt_interleave;
                w1_merged_col_offset_tiled[tile_start_id + 2] = prev_cnt + cnt_interleave + cnt_remain;
                w1_merged_col_offset_tiled[tile_start_id + 3] = flag_min_neg ? 1 : -1;

                prev_cnt += cnt_interleave + cnt_remain;
                prev_nnz += neg + pos;

                // std::cout << "tile: " << cnt_interleave << ":" << cnt_remain << "\n";
            }
        }

        inter_tile_swizzle(columns, (int)ceil(1.*inners/TILE_K), w1_merged_col_offset_tiled, col_tile_indices_);
        // for (int i = 0; i < columns; i++) {
        //     for (int j = 0; j < ceil(1.*inners/TILE_K); j++)
        //         std::cout << col_tile_indices_[i*(int)ceil(1.*inners/TILE_K) + j] << ", ";
        //     std::cout << "\n";
        // }
    }

    /**
     * compress into standard csc format
     */
    void compress_normal_csc() {
        w1_cnt_nnz = w1_cnt_neg + w1_cnt_pos;
        w1_row_indice = (int*)calloc(w1_cnt_nnz, sizeof(int));
        w1_col_offset = (int*)calloc(columns+1, sizeof(int));
        w1_values = (XType*)calloc(w1_cnt_nnz, sizeof(XType));

        int nb_nzz = 0;
        for (int y = 0; y < columns; y++) {
            w1_col_offset[y] = nb_nzz;
            for (int x = 0; x < inners; x++) {
                int8_t val = host_w1.data()[x * columns + y];
                if(val == 1 || val == -1){
                    w1_row_indice[nb_nzz] = x;
                    w1_values[nb_nzz] = val;
                    nb_nzz++;
                }
            }
        }

        w1_col_offset[columns] = nb_nzz;
        return;
    }

    template<typename T> 
    void generate_random_array(T* dst, int size) {
        for (int i = 0; i < size; i++) {
            int weight = rand();
            dst[i] = (T)((T)weight / 10);
        }
    }

    void generate_random_ternary_array(int8_t* W, int ROW, int COL, int& cnt_neg, int& cnt_pos, float Sparsity) {
        std::mt19937 generator(0);
        int nonZero = ROW * (1 - Sparsity);
        float Variation = 0.05;
        std::uniform_int_distribution<int> range(0, ROW - 1);
        std::uniform_int_distribution<int> dynamicCOL(0, min(int(Variation * ROW) + 1, int(nonZero / 2)));
        std::uniform_int_distribution<int> dynamicPosNeg(0, int(Variation * nonZero) + 1);
        for (int j = 0; j < COL; j += 2) {
            int colVari = dynamicCOL(generator);
            int posVari = dynamicPosNeg(generator);
            int col0Pos = (nonZero + colVari) / 2 + posVari;
            int col0Neg = (nonZero + colVari) / 2 - posVari;
            int col1Pos = (nonZero - colVari) / 2 + posVari;
            int col1Neg = (nonZero - colVari) / 2 - posVari;
            int num = 0;
            while (num < col0Pos) {
                int i = range(generator);
                if (W[i * COL + j] != +1) {
                    W[i * COL + j] = +1;
                    num++;
                }
            }
            num = 0;
            while (num < col0Neg) {
                int i = range(generator);
                if ((W[i * COL + j] != +1) && (W[i * COL + j] != -1)) {
                    W[i * COL + j] = -1;
                    num++;
                }
            }
            num = 0;
            while (num < col1Pos) {
                int i = range(generator);
                if (W[i * COL + j] != +1) {
                    W[i * COL + j + 1] = +1;
                    num++;
                }
            }
            num = 0;
            while (num < col1Neg) {
                int i = range(generator);
                if ((W[i * COL + j] != +1) && (W[i * COL + j] != -1)) {
                    W[i * COL + j + 1] = -1;
                    num++;
                }
            }
            cnt_pos += col0Pos + col1Pos;
            cnt_neg += col0Neg + col1Neg;
        }
        std::cout << "Ternary General. NNZ: " << (cnt_neg+cnt_pos) << ", Density: " << (float)(cnt_neg+cnt_pos)/(float)(ROW*COL) << "\n";
    }

    void generate_random_uniform_ternary_array(
        int8_t* W,
        int ROW,                // Number of rows
        int COL,                // Number of columns
        int& cnt_neg, 
        int& cnt_pos,
        double Sparsity,      // Fraction of zeros (0 < sparsity < 1)
        unsigned int seed=0   // Random seed, default 0 for reproducibility
    )
    {
        std::mt19937 generator(seed);
        int steps = ROW * (1 - Sparsity) / 2;
        int stepSize = ROW / steps;
        std::uniform_int_distribution<int> uniformRange(0, stepSize - 1);
        for (int j = 0; j < COL; j++) {
            for (int i = 0; i < steps; i++) {
                // +1
                int index = uniformRange(generator) + i * stepSize;
                W[index * COL + j] = +1;
                cnt_pos++;
                // -1
                index = uniformRange(generator) + i * stepSize;
                while (W[index * COL + j] == +1) {
                    index = uniformRange(generator) + i * stepSize;
                }
                W[index * COL + j] = -1;
                cnt_neg++;
            }
        }
        std::cout << "Ternary Uniform. NNZ: " << (cnt_neg + cnt_pos) << ", Density: " << (float)(cnt_neg + cnt_pos) / (float)(ROW * COL) << "\n";
    }

    template<typename MT, typename OT> 
    void print(int H, int W, const MT* matrix) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                    std::cout << (OT)matrix[i*W+j] << ", ";
            }
            std::cout << std::endl;
        }
    }
    
    std::vector<XType> host_x;
    std::vector<int8_t> host_w1;
    std::vector<XType> host_res;
    std::vector<XType> host_w1_xtype;

    XType* dev_x = 0;
    int8_t* dev_w1 = 0;
    XType* dev_res = 0;

    int rows;
    int columns;
    int inners;
    float sparsity;
    int size_x; // rows*inners
    int size_w1;// columns*inners
    int size_res;// rows*columns

    // ter sp csc format
    int w1_cnt_pos = 0;
    int w1_cnt_neg = 0;
    int size_csc_col_offset = 0;        // size of ter csc col offset: column+1
    int* w1_neg_col_offset_pad = 0;
    int* w1_pos_col_offset_pad = 0;
    int* w1_neg_row_indice = 0;         // w1_cnt_neg
    int* w1_pos_row_indice = 0;         // w1_cnt_pos
    int* w1_neg_col_offset = 0;         // column+1
    int* w1_pos_col_offset = 0;         // column+1

    // tiled ter sp csc format: tiled csc
    int* w1_neg_col_offset_tiled = 0;   // column*(inner/TILE_K)*2
    int* w1_pos_col_offset_tiled = 0;   // column*(inner/TILE_K)*2
    int size_csc_col_offset_tiled = 0;  // size of tiled col offset: columns*2*ceil(1.*inners/TILE_K)

    // merged ter sp csc format: mcsc
    int* w1_merged_row_indice = 0;  // merged ter sp csc format: mcsc
    int* w1_merged_col_offset = 0;  // merged ter sp csc format: mcsc
    int size_csc_col_offset_merged = 0; // size of merged col offset: columns*4
    int* col_indices_ = new int[columns];

    // merged tiled ter csc format: tiled mcsc
    int* w1_merged_row_indice_tiled = 0;
    int* w1_merged_col_offset_tiled = 0;
    int size_mcsc_col_offset_tiled = 0; // size of merged tiled col offset: columns*ceil(1.*inners/TILE_K)*4
    int* col_tile_indices_ = 0;
    bool tile_merged = false;

    // normal sp csc format
    int w1_cnt_nnz = 0;
    int* w1_col_offset = 0;
    int* w1_row_indice = 0;
    XType* w1_values;

    // grouped csc format
};



struct SpMMStat
{
    typedef std::map<float, std::vector<double>> SparsitySpanMap;   // sparsity -> function runtime
    typedef std::map<int, SparsitySpanMap> SPMMConfigSpanMap;      // matrix dims -> sparsity -> function runtime

    typedef std::map<float, double> SparsityMemUseMap;   // matrix dims -> mem usage
    typedef std::map<int, SparsityMemUseMap> SPMMConfigMemUseMap;      // sparsity -> matrix dims -> mem usage

    SPMMConfigSpanMap fn_spans;    // ms, sparsity -> function runtime
    SPMMConfigSpanMap kn_spans;    // ms, sparsity -> kernel runtime
    SPMMConfigSpanMap prekn_spans;  // ms, sparsity -> pre-kernel runtime
    SPMMConfigSpanMap postkn_spans; // ms, sparsity -> post-kernel runtime
    SPMMConfigMemUseMap fn_mem_use;
    int curr_config;

    void dump(std::string path) {

        /* dump kernel spans */
        for (auto& kn : kn_spans) {
            int config_num = kn.first;
            std::ofstream fs(path+"kn_config_"+std::to_string(config_num)); 
            if (!fs) {
                ERROR("Cannot open " << path << "kn_config_"+std::to_string(config_num));
                return;
            }

            auto& sparsity_spans = kn.second;
            for (auto& sparsity_span : sparsity_spans) {
                float sparsity = sparsity_span.first;
                auto& spans = sparsity_span.second;
                fs << sparsity << " ";
                for (auto t : spans)
                    fs << std::to_string(t) << " ";
                fs << "\n";
            }
            fs.close();
        }

        /* dump mem use */
        for (auto& kn : fn_mem_use) {
            int config_num = kn.first;
            std::ofstream fs(path+"mem_config_"+std::to_string(config_num)); 
            if (!fs) {
                ERROR("Cannot open " << path << "kn_config_");
                return;
            }

            auto& mem_uses = kn.second;
            for (auto& memuse : mem_uses) {
                float sparsity = memuse.first;
                double mem = memuse.second;
                fs << sparsity << " ";
                fs << std::to_string(mem) << " ";
                fs << "\n";
            }
            fs.close();
        }
    }
};