#pragma once
#include<vector>;
#include<iostream>;
using namespace std;

class SparseFormat {
public:
	vector<int> col_start_pos;
	vector<int> col_start_neg;
	vector<int16_t> row_index_pos;
	vector<int16_t> row_index_neg;

	SparseFormat(const int8_t* matrix, int K, int N) { // K ROWS, N COLUMNS
		int column_start_pos = 0;
		int column_start_neg = 0;
// #pragma omp parallel for
		for (int n = 0; n < N; n++) {
			this->col_start_pos.push_back(column_start_pos);
			this->col_start_neg.push_back(column_start_neg);
			for (int k = 0; k < K; k++) {
				if (matrix[k * N + n] >= 1) {
					column_start_pos++;
					this->row_index_pos.push_back(k);
				}
				else if (matrix[k * N + n] <= -1) {
					column_start_neg++;
					this->row_index_neg.push_back(k);
				}
			}
		}
		this->col_start_pos.push_back(column_start_pos);
		this->col_start_neg.push_back(column_start_neg);
	}
};

class MergedTCSC {
public:
	vector<int> metadata; // algin_start, align_end, remain_end, remain_value
	vector<int16_t> row_index;

	MergedTCSC(const SparseFormat naiveTCSC, int K, int N) {
		
		int remain_end = 0;
		int remain_val = 0;
		int idx_pos = 0;
		int idx_neg = 0;
// #pragma omp parallel for
		for (int n = 0; n < N; n++) {
			int align_end = 0;
			int col_pos = naiveTCSC.col_start_pos[n + 1] - naiveTCSC.col_start_pos[n];
			int col_neg = naiveTCSC.col_start_neg[n + 1] - naiveTCSC.col_start_neg[n];
			if (col_pos >= col_neg) {
				align_end += col_neg * 2;
				remain_val = 1;
				if (col_pos == col_neg) {
					remain_val = 0;
				}
				for (int k = 0; k < col_neg; k++) {
					row_index.push_back(naiveTCSC.row_index_pos[idx_pos]);
					idx_pos++;
					row_index.push_back(naiveTCSC.row_index_neg[idx_neg]);
					idx_neg++;
				}
				for (int k = 0; k < col_pos - col_neg; k++) {
					row_index.push_back(naiveTCSC.row_index_pos[idx_pos]);
					idx_pos++;
				}
			}
			else {
				align_end += col_pos * 2;
				remain_val = -1;
				for (int k = 0; k < col_pos; k++) {
					row_index.push_back(naiveTCSC.row_index_pos[idx_pos]);
					idx_pos++;
					row_index.push_back(naiveTCSC.row_index_neg[idx_neg]);
					idx_neg++;
				}
				for (int k = 0; k < col_neg - col_pos; k++) {
					row_index.push_back(naiveTCSC.row_index_neg[idx_neg]);
					idx_neg++;
				}
			}
			metadata.push_back(remain_end); // remain_end is also the algin_start of the next column 
			metadata.push_back(remain_end+align_end);
			remain_end += col_pos + col_neg;
			metadata.push_back(remain_end);
			metadata.push_back(remain_val);
		}
	}
};

class MergedTCSC_Group {
public:
	vector<int> metadata; // align_start, algin_end, (remain_end_pos, remain_end_neg) x group_size in min grouping
						  // align_start, algin_end, (remain_end, remain_val) x group_size in mid grouping
						  // align_start, algin_end in max grouping
	vector<int16_t> row_index;
	
	MergedTCSC_Group(SparseFormat naiveTCSC, int K, int N, int group_size, string group_method, bool interleaved) {
		int align_pointer = 0; // to increase with the total number of non-zero elements
// #pragma omp parallel for
		for (int n = 0; n < N; n += group_size) {
			vector<int> col_pos(group_size, 0); // number of pos values at each column
			vector<int> col_neg(group_size, 0);
			int align_min = 65536; // Count the align of min(min(pos,neg)...group_size)
			int align_mid = 0; // Count the align of max(min(pos,neg)...group_size)
			int align_max = 0; // Count the align of max(max(pos,neg)...group_size)
			int align_chosen = 0; // The final chosen align method: one of three above.
			int mid_longest_align = 0; // Find the position of align parts, so other cols follow it in mid grouping method
			int max_longest_nzero = 0; // Find the position of nzero parts, so other cols follow it in max grouping method
			int longest_align = 0;
			int longest_nzero = 0;
			for (int i = 0; i < group_size; i++) {
				col_pos[i] = naiveTCSC.col_start_pos[n + i + 1] - naiveTCSC.col_start_pos[n + i];
				col_neg[i] = naiveTCSC.col_start_neg[n + i + 1] - naiveTCSC.col_start_neg[n + i];
				align_min = min(min(col_pos[i], col_neg[i]), align_min);
				align_mid = max(min(col_pos[i], col_neg[i]), align_mid);
				align_max = max(max(col_pos[i], col_neg[i]), align_max);
				if (min(col_pos[i], col_neg[i]) > longest_align) {
					longest_align = min(col_pos[i], col_neg[i]);
					mid_longest_align = i;
				}
				if (max(col_pos[i], col_neg[i]) > longest_nzero) {
					longest_nzero = max(col_pos[i], col_neg[i]);
					max_longest_nzero = i;
				}
			}
			// Copy data
			if (interleaved) {
				for (int k = 0; k < align_min; k++) {
					for (int i = 0; i < group_size; i++) {
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
					}
				}
			}
			else {
				for (int k = 0; k < align_min; k++) {
					for (int i = 0; i < group_size; i++) 
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator						
					for (int i = 0; i < group_size; i++) 
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);					
				}
			}
			
			if (group_method == "min") { // min grouping copies all pos and neg values at each column to the row_index, each row needs two for() loops on remainders 
				align_chosen = align_min;
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
				for (int i = 0; i < group_size; i++) {
					/* for (int k = align_chosen; k < naiveTCSC.col_start_pos[n + i + 1]; k++) {
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]);
					}*/
					// k: align_chosen ~ col_end // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[startPos], &naiveTCSC.row_index_pos[endPos]);
					int startPos = naiveTCSC.col_start_pos[n + i] + align_chosen;
					int endPos = naiveTCSC.col_start_pos[n + i + 1];
					for (int k = startPos; k < endPos; k++)
						row_index.push_back(naiveTCSC.row_index_pos[k]);
					align_pointer += endPos - startPos;
					metadata.push_back(align_pointer); // remain_end_pos
					/* for (int k = align_chosen; k < naiveTCSC.col_start_neg[n + i + 1]; k++) { // k: align_chosen ~ col_end
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					}*/
					int startNeg = naiveTCSC.col_start_neg[n + i] + align_chosen;
					int endNeg = naiveTCSC.col_start_neg[n + i + 1];
					for (int k = startNeg; k < endNeg; k++)
						row_index.push_back(naiveTCSC.row_index_neg[k]);
					// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[startNeg], &naiveTCSC.row_index_neg[endNeg]);
					align_pointer += endNeg - startNeg;
					metadata.push_back(align_pointer); // remain_end_neg
				}
			}
			else if (group_method == "mid") { // mid is put all aligned parts together, and put zeros to other columns, so that each col only has one for() loop on remainders
				align_chosen = align_mid;
				for (int k = align_min; k < align_chosen; k++) {
					for (int i = 0; i < group_size; i++) {
						if (k < min(col_pos[i], col_neg[i])) {
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
						}
						else {
							int same_as_longest_align = naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + mid_longest_align] + k]; // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(same_as_longest_align); // two same values produces 0: a-a=0
							row_index.push_back(same_as_longest_align);
						}
					}
				}
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
				// Copy one remainder per column
				for (int i = 0; i < group_size; i++) {
					// k: align_chosen ~ col_end
					if (col_pos[i] > col_neg[i]) {						
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + col_neg[i]], &naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i + 1] -1]);
						int startPos = naiveTCSC.col_start_pos[n + i] + col_neg[i];
						int endPos = naiveTCSC.col_start_pos[n + i + 1];
						for (int k = startPos; k < endPos; k++)
							row_index.push_back(naiveTCSC.row_index_pos[k]);
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[startPos], &naiveTCSC.row_index_pos[endPos]);
						align_pointer += endPos - startPos;
						metadata.push_back(align_pointer); // remain_end
						metadata.push_back(+1); // remain_val
					}
					else {
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + col_pos[i]], &naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i + 1] -1]);
						int startNeg = naiveTCSC.col_start_neg[n + i] + col_pos[i];
						int endNeg = naiveTCSC.col_start_neg[n + i + 1];
						for (int k = startNeg; k < endNeg; k++)
							row_index.push_back(naiveTCSC.row_index_neg[k]);
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[startNeg], &naiveTCSC.row_index_neg[endNeg]);
						align_pointer += endNeg - startNeg;
						metadata.push_back(align_pointer); // remain_end
						metadata.push_back(-1);  // remain_val
					}
				}
			}
			else if (group_method == "max") { // max refer to the semi-structured sparsity: each column has the same number of 1s and -1s, no for() loop on remainders
				align_chosen = align_mid;
				for (int k = align_min; k < align_chosen; k++) {
					for (int i = 0; i < group_size; i++) {
						if (k < min(col_pos[i], col_neg[i])) {
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
						}
						else {
							int same_as_longest_align = naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + mid_longest_align] + k]; // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(same_as_longest_align); // two same values produces 0: a-a=0
							row_index.push_back(same_as_longest_align);
						}
					}
				}
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
			}
			else { // other cases returns a fully aligned matrix: same as the unified TCSC
				// Pass group_size == N. No additional processing is needed.
				if(n==0)
					metadata.push_back(align_min * 2); // as the non-zero elements per column 
			}
		}
	}


	MergedTCSC_Group(SparseFormat naiveTCSC, int K, int N, int group_size, string group_method, bool interleaved, int SIMD_SIZE) {
		int align_pointer = 0; // to increase with the total number of non-zero elements
// #pragma omp parallel for
		for (int n = 0; n < N; n += group_size) {
			vector<int> col_pos(group_size, 0); // number of pos values at each column
			vector<int> col_neg(group_size, 0);
			int align_min = 65536; // Count the align of min(min(pos,neg)...group_size)
			int align_mid = 0; // Count the align of max(min(pos,neg)...group_size)
			int align_max = 0; // Count the align of max(max(pos,neg)...group_size)
			int align_chosen = 0; // The final chosen align method: one of three above.
			int mid_longest_align = 0; // Find the position of align parts, so other cols follow it in mid grouping method
			int max_longest_nzero = 0; // Find the position of nzero parts, so other cols follow it in max grouping method
			int longest_align = 0;
			int longest_nzero = 0;
			for (int i = 0; i < group_size; i++) {
				col_pos[i] = naiveTCSC.col_start_pos[n + i + 1] - naiveTCSC.col_start_pos[n + i];
				col_neg[i] = naiveTCSC.col_start_neg[n + i + 1] - naiveTCSC.col_start_neg[n + i];
				align_min = min(min(col_pos[i], col_neg[i]), align_min);
				align_mid = max(min(col_pos[i], col_neg[i]), align_mid);
				align_max = max(max(col_pos[i], col_neg[i]), align_max);
				if (min(col_pos[i], col_neg[i]) > longest_align) {
					longest_align = min(col_pos[i], col_neg[i]);
					mid_longest_align = i;
				}
				if (max(col_pos[i], col_neg[i]) > longest_nzero) {
					longest_nzero = max(col_pos[i], col_neg[i]);
					max_longest_nzero = i;
				}
			}
			// Copy data
			if (interleaved) {
				for (int k = 0; k < align_min; k++) {
					for (int i = 0; i < group_size; i+=SIMD_SIZE) {						
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator						
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);						 
					}
				}
			}
			else {
				if (SIMD_SIZE > 1) { // SIMD_SIZE interleaved & continue within SIMD_SIZE
					for (int k = 0; k < align_min; k++) {
						for (int i = 0; i < group_size; i += SIMD_SIZE) {
							for (int s = 0; s < SIMD_SIZE; s++) {
								row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i + s] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							}
							for (int s = 0; s < SIMD_SIZE; s++) {
								row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i + s] + k]);
							}
						}
					}
				}
				else {
					for (int k = 0; k < align_min; k++) {
						for (int i = 0; i < group_size; i++)
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator						
						for (int i = 0; i < group_size; i++)
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
					}
				}				
			}

			if (group_method == "min") { // min grouping copies all pos and neg values at each column to the row_index, each row needs two for() loops on remainders 
				align_chosen = align_min;
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
				for (int i = 0; i < group_size; i++) {
					/* for (int k = align_chosen; k < naiveTCSC.col_start_pos[n + i + 1]; k++) {
						row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]);
					}*/
					// k: align_chosen ~ col_end // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[startPos], &naiveTCSC.row_index_pos[endPos]);
					int startPos = naiveTCSC.col_start_pos[n + i] + align_chosen;
					int endPos = naiveTCSC.col_start_pos[n + i + 1];
					for (int k = startPos; k < endPos; k++)
						row_index.push_back(naiveTCSC.row_index_pos[k]);
					align_pointer += endPos - startPos;
					metadata.push_back(align_pointer); // remain_end_pos
					/* for (int k = align_chosen; k < naiveTCSC.col_start_neg[n + i + 1]; k++) { // k: align_chosen ~ col_end
						row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
					}*/
					int startNeg = naiveTCSC.col_start_neg[n + i] + align_chosen;
					int endNeg = naiveTCSC.col_start_neg[n + i + 1];
					for (int k = startNeg; k < endNeg; k++)
						row_index.push_back(naiveTCSC.row_index_neg[k]);
					// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[startNeg], &naiveTCSC.row_index_neg[endNeg]);
					align_pointer += endNeg - startNeg;
					metadata.push_back(align_pointer); // remain_end_neg
				}
			}
			else if (group_method == "mid") { // mid is put all aligned parts together, and put zeros to other columns, so that each col only has one for() loop on remainders
				align_chosen = align_mid;
				for (int k = align_min; k < align_chosen; k++) {
					for (int i = 0; i < group_size; i++) {
						if (k < min(col_pos[i], col_neg[i])) {
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
						}
						else {
							int same_as_longest_align = naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + mid_longest_align] + k]; // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(same_as_longest_align); // two same values produces 0: a-a=0
							row_index.push_back(same_as_longest_align);
						}
					}
				}
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
				// Copy one remainder per column
				for (int i = 0; i < group_size; i++) {
					// k: align_chosen ~ col_end
					if (col_pos[i] > col_neg[i]) {
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + col_neg[i]], &naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i + 1] -1]);
						int startPos = naiveTCSC.col_start_pos[n + i] + col_neg[i];
						int endPos = naiveTCSC.col_start_pos[n + i + 1];
						for (int k = startPos; k < endPos; k++)
							row_index.push_back(naiveTCSC.row_index_pos[k]);
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_pos[startPos], &naiveTCSC.row_index_pos[endPos]);
						align_pointer += endPos - startPos;
						metadata.push_back(align_pointer); // remain_end
						metadata.push_back(+1); // remain_val
					}
					else {
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + col_pos[i]], &naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i + 1] -1]);
						int startNeg = naiveTCSC.col_start_neg[n + i] + col_pos[i];
						int endNeg = naiveTCSC.col_start_neg[n + i + 1];
						for (int k = startNeg; k < endNeg; k++)
							row_index.push_back(naiveTCSC.row_index_neg[k]);
						// row_index.insert(row_index.end(), &naiveTCSC.row_index_neg[startNeg], &naiveTCSC.row_index_neg[endNeg]);
						align_pointer += endNeg - startNeg;
						metadata.push_back(align_pointer); // remain_end
						metadata.push_back(-1);  // remain_val
					}
				}
			}
			else if (group_method == "max") { // max refer to the semi-structured sparsity: each column has the same number of 1s and -1s, no for() loop on remainders
				align_chosen = align_mid;
				for (int k = align_min; k < align_chosen; k++) {
					for (int i = 0; i < group_size; i++) {
						if (k < min(col_pos[i], col_neg[i])) {
							row_index.push_back(naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + i] + k]); // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(naiveTCSC.row_index_neg[naiveTCSC.col_start_neg[n + i] + k]);
						}
						else {
							int same_as_longest_align = naiveTCSC.row_index_pos[naiveTCSC.col_start_pos[n + mid_longest_align] + k]; // Start point of this column: naiveTCSC.col_start_pos[n + i], k is the iterator
							row_index.push_back(same_as_longest_align); // two same values produces 0: a-a=0
							row_index.push_back(same_as_longest_align);
						}
					}
				}
				metadata.push_back(align_pointer); // align_start
				align_pointer += align_chosen * group_size * 2;
				metadata.push_back(align_pointer); // align_end
			}
			else { // other cases returns a fully aligned matrix: same as the unified TCSC
				// Pass group_size == N. No additional processing is needed.
				if (n == 0)
					metadata.push_back(align_min * 2); // as the non-zero elements per column 
			}
		}
	}
};

//******************** A direct sparse GEMM based on the naive TCSC ***************************************

template <typename T>
void sparseGEMM(T* X, SparseFormat W, T* bias, T* Y, int M, int N, int K) {
	for (int m = 0; m < M; m++) {
#pragma omp parallel for
		for (int n = 0; n < N; n++) {
			T y = 0;
#pragma omp simd
			for (int k = W.col_start_pos[n]; k < W.col_start_pos[n + 1]; k++) {
				y += X[m * K + W.row_index_pos[k]];
			}
#pragma omp simd
			for (int k = W.col_start_neg[n]; k < W.col_start_neg[n + 1]; k++) {
				y -= X[m * K + W.row_index_neg[k]];
			}
			Y[m * N + n] = y + bias[n];
		}
	}
};