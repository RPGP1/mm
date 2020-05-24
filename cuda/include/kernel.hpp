#pragma once

#include "gemm_fwd.hpp"

#include <cstdint>


namespace CudaMM
{

namespace Kernel
{

template <uint32_t Rows_, uint32_t Cols_>
struct Matrix2D {
    static constexpr uint32_t Rows = Rows_, Cols = Cols_;

    size_t pitch;

    static constexpr uint32_t rows = Rows_, cols = Cols_;

    explicit Matrix2D(CudaMM::Matrix2D<Rows_, Cols_> const&);
};

template <uint32_t Rows_>
struct Matrix2D<Rows_, 0> {
    static constexpr uint32_t Rows = Rows_, Cols = 0;

    size_t pitch;

    static constexpr uint32_t rows = Rows_;
    uint32_t cols;

    explicit Matrix2D(CudaMM::Matrix2D<Rows_, 0> const&);
};

template <uint32_t Cols_>
struct Matrix2D<0, Cols_> {
    static constexpr uint32_t Rows = 0, Cols = Cols_;

    size_t pitch;

    uint32_t rows;
    static constexpr uint32_t cols = Cols_;

    explicit Matrix2D(CudaMM::Matrix2D<0, Cols_> const&);
};

template <>
struct Matrix2D<0, 0> {
    static constexpr uint32_t Rows = 0, Cols = 0;

    size_t pitch;

    uint32_t rows, cols;

    explicit Matrix2D(CudaMM::Matrix2D<0, 0> const&);
};

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
struct ProblemData {
    Matrix2D<LhsRows, LhsCols> lhs;
    Matrix2D<LhsCols, RhsCols> rhs;
    Matrix2D<LhsRows, RhsCols> result;

    explicit ProblemData(CudaMM::ProblemData<LhsRows, LhsCols, RhsCols> const&);
};

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
ProblemData<LhsRows, LhsCols, RhsCols> toKernel(CudaMM::ProblemData<LhsRows, LhsCols, RhsCols> const& problem)
{
    return ProblemData<LhsRows, LhsCols, RhsCols>{problem};
}


template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void gemm(
    CudaMM::ProblemData<LhsRows, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result);

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
__global__ void gemm_impl(
    ProblemData<LhsRows, LhsCols, RhsCols> problem,
    const Element* __restrict__ lhs, const Element* __restrict__ rhs, Element* __restrict__ result);

}  // namespace Kernel

}  // namespace CudaMM

#ifdef __CUDACC__
#include "kernel.ipp"
#endif
