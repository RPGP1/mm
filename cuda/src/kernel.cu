#include "kernel.hpp"

__global__ void kernel_impl(__restrict__ Element* lhs, __restrict__ Element* rhs, __restrict__ Element* result,
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    size_t lhs_pitch, size_t rhs_pitch, size_t result_pitch)
{
    for (uint32_t lhs_row{0}; lhs_row < lhs_rows; lhs_row++) {
        auto const& result_row = lhs_row;

        for (uint32_t rhs_col{0}; rhs_col < rhs_cols; rhs_col++) {
            auto const& result_col = rhs_col;

            for (uint32_t lhs_col{0}; lhs_col < lhs_cols; lhs_col++) {
                auto const& rhs_row = lhs_col;

                result[result_row * result_pitch + result_col]
                    += lhs[lhs_row * lhs_pitch + lhs_col] * rhs[rhs_row * rhs_pitch + rhs_col];
            }
        }
    }
}


void kernel(
    Element* lhs, Element* rhs, Element* result,
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    size_t lhs_pitch, size_t rhs_pitch, size_t result_pitch,
    dim3 blocks, dim3 threads, size_t shared_memory_size, cudaStream_t stream)
{
    kernel_impl<<<blocks, threads, shared_memory_size, stream>>>(lhs, rhs, result, lhs_rows, lhs_cols, rhs_cols, lhs_pitch, rhs_pitch, result_pitch);
}
