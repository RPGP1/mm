#pragma once

#include "kernel.hpp"

#include "error.hpp"
#include "gemm.hpp"


namespace CudaMM
{

namespace Kernel
{

namespace
{

__host__ __device__ constexpr uint32_t ceil_div(uint32_t devidend, uint32_t divider)
{
    return devidend / divider + (devidend % divider != 0);
}

__host__ __device__ constexpr uint32_t gcd(uint32_t num1, uint32_t num2)
{
    return num2 == 0 ? num1 : gcd(num2, num1 % num2);
}

}  // namespace


constexpr uint32_t WarpThreads = 32;

// each thread is assigned (ThreadRows x ThreadCols) elements in result matrix
constexpr uint32_t ThreadRows = 13, ThreadCols = 16;
// (ThreadSharedRows x ThreadSharedCols) of (ThreadRows x ThreadCols) are stored in shared memory
constexpr uint32_t ThreadSharedRows = 0, ThreadSharedCols = 0;

// shared memory is allocated for each thread to do the sum of the products (Stride) times
constexpr uint32_t Stride = 32;

// each block consists of (BlockRowThreads x BlockColThreads) threads
constexpr uint32_t BlockRowThreads = 16, BlockColThreads = 16,
                   BlockSize = BlockRowThreads * BlockColThreads;

constexpr uint32_t BlockRows = BlockRowThreads * ThreadRows, BlockCols = BlockColThreads * ThreadCols;

// each thread can sum the products up to (MaxAccumulation) times
constexpr uint32_t MaxAccumulation = 4096;

constexpr uint32_t MinBlocksPerMultiprocessor = 1;


constexpr uint32_t LhsSharedRows = BlockRows,
                   LhsSharedCols = Stride + (BlockColThreads % WarpThreads != 0);  // avoid bank conflicts
constexpr uint32_t RhsSharedRows = Stride,
                   RhsSharedCols = BlockCols;
constexpr uint32_t ResultSharedRows = BlockRowThreads * ThreadSharedRows,
                   ResultSharedCols = (ThreadSharedCols == 0 ? 0
                                                             : BlockColThreads
                                                                   + ceil_div(BlockColThreads * (ThreadSharedCols - 1), WarpThreads) * WarpThreads);  // avoid bank conflicts

constexpr uint32_t SharedSize = LhsSharedRows * LhsSharedCols + RhsSharedRows * RhsSharedCols + ResultSharedRows * ResultSharedCols;


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
    gemm_impl<<<{(problem.result.cols + BlockColThreads - 1) / BlockColThreads,
                    (problem.result.rows + BlockRowThreads - 1) / BlockRowThreads},
        {BlockColThreads, BlockRowThreads}, 0, stream>>>(

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
