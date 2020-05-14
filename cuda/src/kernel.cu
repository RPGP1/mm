#include "kernel.hpp"

__global__ void kernel(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    Element* lhs, Element* rhs, Element* result)
{
    auto const& result_cols = rhs_cols;

    for (uint32_t lhs_row{0}; lhs_row < lhs_rows; lhs_row++) {
        auto const& result_row = lhs_row;

        for (uint32_t rhs_col{0}; rhs_col < rhs_cols; rhs_col++) {
            auto const& result_col = rhs_col;

            result[result_row * result_cols + result_col] = 0;

            for (uint32_t lhs_col{0}; lhs_col < lhs_cols; lhs_col++) {
                auto const& rhs_row = lhs_col;

                result[result_row * result_cols + result_col]
                    += lhs[lhs_row * lhs_cols + lhs_col] * rhs[rhs_row * rhs_cols + rhs_col];
            }
        }
    }
}


void cudaGemm(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    Element* lhs, Element* rhs, Element* result)
{
    kernel<<<1, 1>>>(lhs_rows, lhs_cols, rhs_cols, lhs, rhs, result);
}
