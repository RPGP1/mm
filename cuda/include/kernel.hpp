#pragma once

#include "gemm_fwd.hpp"

#include <cstdint>


namespace CudaMM
{

namespace Kernel
{

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void gemm(
    ProblemData<LhsRows, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result);

__global__ void gemm_impl(__restrict__ const Element* lhs, __restrict__ const Element* rhs, __restrict__ Element* result,
    const uint32_t lhs_rows, const uint32_t lhs_cols, const uint32_t rhs_cols,
    size_t lhs_pitch, size_t rhs_pitch, size_t result_pitch);

}  // namespace Kernel

}  // namespace CudaMM

#ifdef __CUDACC__
#include "kernel.ipp"
#endif
