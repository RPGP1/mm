#pragma once

#include "kernel.hpp"

#include "error.hpp"
#include "gemm.hpp"


namespace CudaMM
{

namespace Kernel
{

// shared memory is allocated for each thread to do the sum of the products (StrideUnit) times
constexpr uint32_t StrideUnit{256};

// each thread can sum the products up to (MaxAccumulation) times
constexpr uint32_t MaxAccumulation{4096};

constexpr uint32_t BlockCols{32};
constexpr uint32_t BlockRows{32};

template <uint32_t Rows_, uint32_t Cols_>
Matrix2D<Rows_, Cols_>::Matrix2D(CudaMM::Matrix2D<Rows_, Cols_> const& mat)
    : pitch{mat.pitch}
{
}

template <uint32_t Rows_>
Matrix2D<Rows_, 0>::Matrix2D(CudaMM::Matrix2D<Rows_, 0> const& mat)
    : pitch{mat.cols}, cols{mat.cols}
{
}

template <uint32_t Cols_>
Matrix2D<0, Cols_>::Matrix2D(CudaMM::Matrix2D<0, Cols_> const& mat)
    : pitch{mat.pitch}, rows{mat.rows}
{
}


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
ProblemData<LhsRows, LhsCols, RhsCols>::ProblemData(CudaMM::ProblemData<LhsRows, LhsCols, RhsCols> const& data)
    : lhs{data.lhs}, rhs{data.rhs}, result{data.result}
{
}


template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void gemm(
    CudaMM::ProblemData<LhsRows, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    cudaStream_t stream = 0;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.lhs.data.get(), sizeof(Element) * problem.lhs.pitch,
        lhs, sizeof(Element) * problem.lhs.cols,
        sizeof(Element) * problem.lhs.cols, problem.lhs.rows,
        cudaMemcpyDefault,
        stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.rhs.data.get(), sizeof(Element) * problem.rhs.pitch,
        rhs, sizeof(Element) * problem.rhs.cols,
        sizeof(Element) * problem.rhs.cols, problem.rhs.rows,
        cudaMemcpyDefault,
        stream));

    auto kernelProblem = toKernel(problem);
    gemm_impl<<<{(problem.result.cols + BlockCols - 1) / BlockCols,
                    (problem.result.rows + BlockRows - 1) / BlockRows},
        {BlockCols, BlockRows}, 0, stream>>>(

        kernelProblem,
        problem.lhs.data.get(), problem.rhs.data.get(), problem.result.data.get());


    CUDA_CHECK(cudaMemcpy2DAsync(
        result, sizeof(Element) * problem.result.cols,
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        sizeof(Element) * problem.result.cols, problem.result.rows,
        cudaMemcpyDefault,
        stream));
}

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
__global__ void gemm_impl(
    ProblemData<LhsRows, LhsCols, RhsCols> problem,
    const Element* __restrict__ lhs, const Element* __restrict__ rhs, Element* __restrict__ result)
{
    auto const result_row = blockIdx.y * blockDim.y + threadIdx.y;
    auto const result_col = blockIdx.x * blockDim.x + threadIdx.x;

    auto const& lhs_row = result_row;
    auto const& rhs_col = result_col;

    if (result_row >= problem.result.rows || result_col >= problem.result.cols) {
        return;
    }

    Element result_cache = 0;

    for (uint32_t lhs_col{0}; lhs_col < problem.lhs.cols; lhs_col++) {
        auto const& rhs_row = lhs_col;

        result_cache += lhs[lhs_row * problem.lhs.pitch + lhs_col] * rhs[rhs_row * problem.rhs.pitch + rhs_col];
    }

    result[result_row * problem.result.pitch + result_col] = result_cache;
}

}  // namespace Kernel

}  // namespace CudaMM
