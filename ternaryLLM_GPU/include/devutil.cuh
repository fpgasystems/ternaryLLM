#pragma once

template <typename VECTOR_TYPE, typename DATA_TYPE, int kBlockWidth>
struct MemoryAligner
{
    //
    /// Static members.
    //

    // The number of values we need to align the pointers to.
    static constexpr int kValueAlignment = sizeof(VECTOR_TYPE) / sizeof(DATA_TYPE);

    // Pre-calculated mask used to efficiently align the row offset.
    static constexpr uint32_t kAlignmentMask = ~(kValueAlignment - 1);

    // The maximum number of values and indices that we could have to mask.
    static constexpr int kMaxValuesToMask = kValueAlignment - 1;

    static constexpr int kMaskSteps =
        (kMaxValuesToMask + kBlockWidth - 1) / kBlockWidth;


    //
    /// Member variables.
    //

    // The row offset in the sparse matrix values & column indices buffers.
    int row_offset_;

    // The number of nonzeros in this row of the sparse matrix.
    int nonzeros_;

    // The number of values we need to mask out at the start of the first
    // computed tile.
    int values_to_mask_;

    // Constructor. Save the row offset and initialize the masked region size.
    __device__ __forceinline__ MemoryAligner(int row_offset, int nonzeros)
    {
        row_offset_ = row_offset;
        nonzeros_ = nonzeros;

        // NOTE: kValueAlignment is guaranteed to be 2 or 4, so we can express
        // modulo by kValueAlignment in this way. Switching to this expression
        // produced much cleaner code than relying on the compiler to optimize
        // away the modulo.
        values_to_mask_ = row_offset & (kValueAlignment - 1);
    }

    /**
     * @brief Potentially align the sparse matrix pointers to the vector width.
     *
     * NOTE: This code effectively reduces the row offset to the nearest 128
     * or 64-byte aligned value. All memory allocated with cudaMalloc is 128-
     * byte aligned, thus this code will never cause our kernels to issue out-
     * of-bounds memory accesses to the region before the allocations used to
     * store the sparse matrix.
     */
    __device__ __forceinline__ int AlignedRowOffset()
    {
        return row_offset_ & kAlignmentMask;
    }

    __device__ __forceinline__ int AlignedNonzeros()
    {
        return nonzeros_ + values_to_mask_;
    }

    __device__ __forceinline__ int AlignedExtra()
    {
        return values_to_mask_;
    }

    __device__ __forceinline__ void MaskPrefix(int* row_indices_tile_si) {
        // NOTE: The below masking code is data type agnostic. Cast input pointers
        // to float/int so that we efficiently operate on 4-byte words.
        int* row_indices_tile = reinterpret_cast<int*>(row_indices_tile_si);

        int mask_idx = threadIdx.y;
    #pragma unroll
        for (int mask_step = 0; mask_step < kMaskSteps; ++mask_step) {
            if (mask_idx < values_to_mask_) {
                // NOTE: We set the column index for these out-of-bounds values to
                // a dummy values of zero. This will trigger a superfluous load into
                // the dense right-hand side matrix, but will never be out-of-bounds.
                // We set the value for this index to 0 s.t. the superfluous rhs value
                // is not accumulated into the output.
                row_indices_tile[mask_idx] = 0;
                mask_idx += kBlockWidth;
            }
        }
    }
};

__device__ __forceinline__ void ConvertINTVectorToColOffset(int4& vec, int32_t* dst) {
    dst[0] = vec.x;
    dst[1] = vec.y;
    dst[2] = vec.z;
    dst[3] = vec.w;
}


__device__ __forceinline__ void ConvertINTVectorToColOffset(int2& vec, int16_t* dst) {
    dst[0] = vec.x & 0xFF;
    dst[1] = (vec.x >> 8) & 0xFF;
    dst[2] = (vec.y) & 0xFF;
    dst[3] = (vec.y >> 8) & 0xFF;
}

__device__ __forceinline__ void StoreFPVectorToArray(float& vec, float* dst, int rows) {
    dst[0] = vec;
}

__device__ __forceinline__ void StoreFPVectorToArray(float2& vec, float* dst, int rows) {
    dst[0] = vec.x;
    dst[rows] = vec.y;
}


__device__ __forceinline__ void StoreFPVectorToArray(float4& vec, float* dst, int rows) {
    dst[0] = vec.x;
    dst[rows] = vec.y;
    dst[rows*2] = vec.z;
    dst[rows*3] = vec.w;
}