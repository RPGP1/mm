#include "kernel.hpp"

#include "gemm.hpp"
#include "size.hpp"
template void CudaMM::gemm<LhsRows, LhsCols, RhsCols>(
    CudaMM::DeviceData<LhsRows, LhsCols, RhsCols>&,
    const Element* lhs, const Element* rhs, Element* result);


namespace CudaMM
{

namespace Kernel
{

__global__ void gemm_impl(__restrict__ const Element* lhs, __restrict__ const Element* rhs, __restrict__ Element* result,
    const uint32_t lhs_rows, const uint32_t lhs_cols, const uint32_t rhs_cols,
    size_t lhs_pitch, size_t rhs_pitch, size_t result_pitch)
{
    auto const result_row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const result_col = blockIdx.x * blockDim.x + threadIdx.x;

    auto const& lhs_row = result_row;
    auto const& rhs_col = result_col;

    __shared__ Element result_cache[BlockLength][BlockLength];

    if (lhs_row >= lhs_rows || rhs_col >= rhs_cols) {
        return;
    }

    result_cache[threadIdx.y][threadIdx.x] = Element{0};

    for (uint32_t lhs_col{0}; lhs_col < lhs_cols; lhs_col++) {
        auto const& rhs_row = lhs_col;

        result_cache[threadIdx.y][threadIdx.x]
            += lhs[lhs_row * lhs_pitch + lhs_col] * rhs[rhs_row * rhs_pitch + rhs_col];
    }

    result[result_row * result_pitch + result_col] = result_cache[threadIdx.y][threadIdx.x];
}

}  // namespace Kernel

}  // namespace CudaMM