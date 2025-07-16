// Author: fuguan@ethz.ch
// Copyrights reserved
#include "ter_spmm_kernel.cuh"
#include <torch/extension.h>
#include <vector>
#include "pybind11/pybind11.h"
#include <iostream>

void inter_tile_swizzle(int columns, int tile_num_per_col, const int32_t* col_offsets, int32_t* col_tile_indices) {
    for (int i = 0; i < columns; i++) {
        std::vector<int> swizzle_staging(tile_num_per_col);
        std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

        std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [i, tile_num_per_col, &col_offsets](int idx_a, int idx_b) {
            int length_a_interleaved = col_offsets[i*tile_num_per_col*4 + idx_a*4 + 1] - col_offsets[i*tile_num_per_col*4 + idx_a*4 + 0];
            int length_a_remain = col_offsets[i*tile_num_per_col*4 + idx_a*4 + 2] - col_offsets[i*tile_num_per_col*4 + idx_a*4 + 1];
            
            int length_b_interleaved = col_offsets[i*tile_num_per_col*4 + idx_b*4 + 1] - col_offsets[i*tile_num_per_col*4 + idx_b*4];
            int length_b_remain = col_offsets[i*tile_num_per_col*4 + idx_b*4 + 2] - col_offsets[i*tile_num_per_col*4 + idx_b*4 + 1];

            return (length_a_interleaved+length_a_remain) < (length_b_interleaved+length_b_remain);
        });

        std::memcpy(col_tile_indices+i*tile_num_per_col, swizzle_staging.data(), sizeof(int) * tile_num_per_col);
    }
}

/**
 * @brief 
 * TESTED
 * @param w 
 * @param columns 
 * @param inners 
 * @param w_cnt_pos_nnz 
 * @param w_cnt_neg_nnz 
 * @return std::vector<std::vector<int32_t>> 
 */
std::vector<std::vector<int32_t>> convert_to_ter_csc(
    const at::Tensor& w, 
    const int columns,
    const int inners,
    const int w_cnt_pos_nnz,
    const int w_cnt_neg_nnz
)
{
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CPU);

    const int size_csc_col_offset = columns+1;

    std::vector<int32_t> w1_neg_row_indice(w_cnt_neg_nnz, 0);
    std::vector<int32_t> w1_pos_row_indice(w_cnt_pos_nnz, 0);
    std::vector<int32_t> w1_neg_col_offset(size_csc_col_offset, 0);
    std::vector<int32_t> w1_pos_col_offset(size_csc_col_offset, 0);
    const int32_t* w_ptr = w.data_ptr<int32_t>();

    //Compress W1
    int nb_neg = 0;
    int nb_pos = 0;
    for (int y = 0; y < columns; y++) {
        w1_neg_col_offset[y] = nb_neg;
        w1_pos_col_offset[y] = nb_pos;
        for (int x = 0; x < inners; x++) {
            if(w_ptr[x * columns + y] == -1){
                w1_neg_row_indice[nb_neg] = x;
                nb_neg++;
            }
            else if(w_ptr[x * columns + y] == 1){
                w1_pos_row_indice[nb_pos] = x;
                nb_pos++;
            }
        }
    }
    w1_neg_col_offset[columns] = nb_neg;
    w1_pos_col_offset[columns] = nb_pos;

    return std::vector<std::vector<int32_t>>{w1_neg_row_indice, w1_neg_col_offset, w1_pos_row_indice, w1_pos_col_offset};
}


/**
 * @brief 
 * TESTED
 * @param w 
 * @param columns 
 * @param inners 
 * @param w_cnt_pos_nnz 
 * @param w_cnt_neg_nnz 
 * @param TILE_K 
 * @return std::vector<std::vector<int32_t>> 
 */
std::vector<std::vector<int32_t>> convert_to_ter_tiled_csc(
    const at::Tensor& w, 
    const int columns,
    const int inners,
    const int w_cnt_pos_nnz,
    const int w_cnt_neg_nnz,
    const int TILE_K
)
{
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CPU);

    std::vector<std::vector<int32_t>> tmp = convert_to_ter_csc(w, columns, inners, w_cnt_pos_nnz, w_cnt_neg_nnz);

    std::vector<int32_t>& w1_neg_row_indice = tmp[0];
    std::vector<int32_t>& w1_pos_row_indice = tmp[2];

    const int size_csc_col_offset_tiled = columns*ceil(1.*inners/TILE_K)*2;
    // at::Tensor w1_neg_col_offset_tiled = torch::empty(size_csc_col_offset_tiled, torch::TensorOptions().requires_grad(false)).toType(torch::kI32);
    // at::Tensor w1_pos_col_offset_tiled = torch::empty(size_csc_col_offset_tiled, torch::TensorOptions().requires_grad(false)).toType(torch::kI32);

    std::vector<int32_t> w1_neg_col_offset_tiled(size_csc_col_offset_tiled, 0);
    std::vector<int32_t> w1_pos_col_offset_tiled(size_csc_col_offset_tiled, 0);

    // int32_t* w1_neg_row_indice_ptr = w1_neg_row_indice.data_ptr<int32_t>();
    // int32_t* w1_pos_row_indice_ptr = w1_pos_row_indice.data_ptr<int32_t>();
    // int32_t* w1_neg_col_offset_tiled_ptr = w1_neg_col_offset_tiled.data_ptr<int32_t>();
    // int32_t* w1_pos_col_offset_tiled_ptr = w1_pos_col_offset_tiled.data_ptr<int32_t>();
    const int32_t* w_ptr = w.data_ptr<int32_t>();

    /* compress tiled col offset */
    int nb_neg = 0;
    int nb_pos = 0;
    int tileNumK = ceil(1.*inners/TILE_K);
    int col_offset_x = tileNumK*2;  // each tile needs 2 offset [start, end)

    // compress column
    for (int y = 0; y < columns; y++) {
        // int* neg_col_offset_start = w1_neg_col_offset_tiled_ptr + y*col_offset_x;
        // int* pos_col_offset_start = w1_pos_col_offset_tiled_ptr + y*col_offset_x;
        // compress tile
        for (int tileId = 0; tileId < tileNumK; tileId++) {
            int offset = tileId*TILE_K;
            w1_neg_col_offset_tiled[y*col_offset_x + tileId*2 + 0] = nb_neg;
            w1_pos_col_offset_tiled[y*col_offset_x + tileId*2 + 0] = nb_pos;
            for (int x = 0; x < TILE_K; x++) {
                int id = (tileId*TILE_K+x)*columns + y;
                if (w_ptr[id] == -1) {
                    w1_neg_row_indice[nb_neg] -= offset;
                    nb_neg++;
                } else if (w_ptr[id] == 1) {
                    w1_pos_row_indice[nb_pos] -= offset;
                    nb_pos++;
                }
            }
            w1_neg_col_offset_tiled[y*col_offset_x + tileId*2 + 1] = nb_neg;
            w1_pos_col_offset_tiled[y*col_offset_x + tileId*2 + 1] = nb_pos;
        }
    }

    return std::vector<std::vector<int32_t>>{w1_neg_row_indice, w1_neg_col_offset_tiled, w1_pos_row_indice, w1_pos_col_offset_tiled};
}



std::vector<std::vector<int32_t>> convert_to_ter_tiled_mcsc(
    const at::Tensor& w, 
    const int columns,
    const int inners,
    const int w_cnt_pos_nnz,
    const int w_cnt_neg_nnz,
    const int TILE_K,
    const bool padding,
    const int padding_size
) 
{
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CPU);

    std::vector<std::vector<int32_t>> tmp = convert_to_ter_tiled_csc(w, columns, inners, w_cnt_pos_nnz, w_cnt_neg_nnz, TILE_K);
    
    std::vector<int32_t>& w1_neg_row_indice = tmp[0];
    std::vector<int32_t>& w1_neg_col_offset_tiled = tmp[1];
    std::vector<int32_t>& w1_pos_row_indice = tmp[2];
    std::vector<int32_t>& w1_pos_col_offset_tiled = tmp[3];


    const int size_mcsc_col_offset_tiled = (int)columns*4*ceil(1.*inners/TILE_K);
    const int w_cnt_nnz = w_cnt_pos_nnz + w_cnt_neg_nnz;

    std::vector<int32_t> w1_merged_row_indice_tiled(w_cnt_nnz, 0);
    std::vector<int32_t> w1_merged_col_offset_tiled(size_mcsc_col_offset_tiled, 0);

    const int32_t* w_ptr = w.data_ptr<int32_t>();
    int zero_ele_id = 0;
    for (; zero_ele_id < columns*inners; zero_ele_id++) {
        if (w_ptr[zero_ele_id] == 0)
            break;
    }

    /* compress merged tiled col offset */
    int tileNumK = ceil(1.*inners/TILE_K);  
    int col_offset_tiled_csc = tileNumK*2;  // each tile needs 2 offset [start, end) in tiled csc
    int col_offset_tiled_mcsc = tileNumK*4; // each tile needs 4 offset [pos_neg_start, pos_neg_end, common_end, sign] in tiled mcsc
    int prev_cnt = 0;
    int prev_nnz = 0;

    // calculate padded size and resize
    if (padding) {
        int w_cnt_nnz_padded = 0;
        for (int y = 0; y < columns; y++) {
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

                /* calculate padded size */
                int cnt_interleave = min_neg_pos*2;
                int cnt_interleave_padded = ceil(1.*cnt_interleave/padding_size)*padding_size;
                int interleave_padding_offset = cnt_interleave_padded - cnt_interleave;

                int cnt_remain = max_neg_pos - min_neg_pos;
                int cnt_remain_padded = ceil(1.*cnt_remain/padding_size)*padding_size;
                int remain_padding_offset = cnt_remain_padded - cnt_remain;

                int nnz_padded = cnt_interleave_padded + cnt_remain_padded;
                w_cnt_nnz_padded += nnz_padded;
            }
        }
        w1_merged_row_indice_tiled.resize(w_cnt_nnz_padded, 0);
    }

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
            int cnt_interleave_padded = padding ? ceil(1.*cnt_interleave/padding_size)*padding_size : cnt_interleave;
            for (int k = 0; k < min_neg_pos; k++) {
                w1_merged_row_indice_tiled[prev_nnz + k*2 + 0] = w1_pos_row_indice[tile_pos_start + k]; 
                w1_merged_row_indice_tiled[prev_nnz + k*2 + 1] = w1_neg_row_indice[tile_neg_start + k];
            }

            /* add paddings */
            for (int k = cnt_interleave; k < cnt_interleave_padded; k++) {
                w1_merged_row_indice_tiled[prev_nnz + k] = w1_pos_row_indice[tile_pos_start]; 
            }

            /* compress common number */
            int cnt_remain = max_neg_pos - min_neg_pos;
            int cnt_remain_padded = padding ? ceil(1.*cnt_remain/padding_size)*padding_size : cnt_remain;
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

            /* todo: tentative methods for memory padding and alignment */
            /* make cnt_remain even */
            if (cnt_remain%2 == 1) {
                cnt_remain--;
            }

            /* make cnt_interleave multiple of 4 */
            if (cnt_interleave%4 != 0) {
                cnt_interleave-=2;
            }

            /* make cnt_remain multiple of 4 */
            if (cnt_remain%4 != 0) {
                cnt_remain-=2;
            }

            tile_start_id = y*col_offset_tiled_mcsc + 4*tileId;
            w1_merged_col_offset_tiled[tile_start_id + 0] = prev_cnt;
            w1_merged_col_offset_tiled[tile_start_id + 1] = padding ? prev_cnt + cnt_interleave_padded : prev_cnt + cnt_interleave;
            w1_merged_col_offset_tiled[tile_start_id + 2] = padding ? prev_cnt + cnt_interleave_padded + cnt_remain : prev_cnt + cnt_interleave + cnt_remain;
            w1_merged_col_offset_tiled[tile_start_id + 3] = flag_min_neg ? 1 : -1;

            prev_cnt += padding ? (cnt_interleave_padded + cnt_remain_padded) : (cnt_interleave + cnt_remain);
            prev_nnz = prev_cnt;
            // prev_nnz += neg + pos;
        }
    }
    
    /* swizzle column indices based on tile */
    // std::vector<int32_t> col_tile_indices_(columns*ceil(1.*inners/TILE_K), 0);
    int total_tile_num = (int)columns*ceil(1.*inners/TILE_K);
    int tile_num_per_col = (int)ceil(1.*inners/TILE_K);
    int32_t* col_tile_indices_ = new int32_t[total_tile_num];
    inter_tile_swizzle(columns, tile_num_per_col, w1_merged_col_offset_tiled.data(), col_tile_indices_);

    
    std::vector<int32_t> col_tile_indices_vec_(col_tile_indices_, col_tile_indices_ + total_tile_num);
    return std::vector<std::vector<int32_t>>{w1_merged_row_indice_tiled, w1_merged_col_offset_tiled, col_tile_indices_vec_};
}


torch::Tensor ter_spmm(
    torch::Tensor& X, 
    torch::Tensor& w_tiled_merged_row_indice, 
    torch::Tensor& w_tiled_merged_col_offset, 
    torch::Tensor& w_col_tile_indices,
    int batch_size, int rows, int columns, int inners,
    bool uniformed, bool padded)
{
    TORCH_INTERNAL_ASSERT(X.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w_tiled_merged_row_indice.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w_tiled_merged_col_offset.device().type() == at::DeviceType::CUDA);
    
	const float* X_ptr = X.data_ptr<float>();
    const int16_t* w_tiled_merged_row_indice_ptr = w_tiled_merged_row_indice.data_ptr<int16_t>();
    const int32_t* w_tiled_merged_col_offset_ptr = w_tiled_merged_col_offset.data_ptr<int32_t>();
    const int16_t* w_col_tile_indices_ptr = w_col_tile_indices.data_ptr<int16_t>();
    at::Tensor result = torch::empty({batch_size, rows, columns}, at::device(at::kCUDA).dtype(at::kFloat));
    float* result_ptr = result.data_ptr<float>();

    ter_spmm_wrapper<float>(
        X_ptr, 
        w_tiled_merged_row_indice_ptr,
        w_tiled_merged_col_offset_ptr,
        w_col_tile_indices_ptr,
        result_ptr,
        batch_size, rows, columns, inners,
        uniformed, padded
    );
    
	return result;
}

PYBIND11_MODULE(ter_spmm, m) {
    m.def("ter_spmm", &ter_spmm, "ternary sparse matrix multiplication");
    m.def("convert_to_ter_csc", &convert_to_ter_csc, "convert to naive ternary csc format");
    m.def("convert_to_ter_tiled_csc", &convert_to_ter_tiled_csc, "convert to tile-wise ternary csc format");
    m.def("convert_to_ter_tiled_mcsc", &convert_to_ter_tiled_mcsc, "convert to tile-wise merged ternary csc format");
}