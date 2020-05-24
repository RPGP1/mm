#include "kernel.hpp"

#include "gemm.hpp"
#include "size.hpp"


namespace CudaMM
{

namespace Kernel
{

Matrix2D<0, 0>::Matrix2D(CudaMM::Matrix2D<0, 0> const& mat)
    : pitch{mat.pitch}, rows{mat.rows}, cols{mat.cols}
{
}


template <>
__global__ void gemm_impl<LhsRows / 2, LhsCols, RhsCols>(
    ProblemData<LhsRows / 2, LhsCols, RhsCols> problem,
    const Element* __restrict__ lhs, const Element* __restrict__ rhs, Element* __restrict__ result)
{
    auto const& row_in_block = threadIdx.y;
    auto const& col_in_block = threadIdx.x;

    auto const result_row = blockIdx.y * blockDim.y + row_in_block;
    auto const result_col = blockIdx.x * blockDim.x + col_in_block;

    auto const& lhs_row = result_row;
    auto const& rhs_col = result_col;

    static_assert(problem.result.rows % BlockRows == 0);
    static_assert(problem.result.cols % BlockCols == 0);


    Element result_cache = 0;
    extern __shared__ Element shared_memory[];
    // Element lhs_cache[BlockRows][StrideUnit];
    // Element rhs_cache[StrideUnit][BlockCols];
    Element(*lhs_cache)[StrideUnit] = reinterpret_cast<Element(*)[StrideUnit]>(shared_memory);
    Element(*rhs_cache)[BlockCols] = reinterpret_cast<Element(*)[BlockCols]>(shared_memory + BlockRows * StrideUnit);

    static_assert(problem.lhs.cols % MaxAccumulation == 0);
    lhs += lhs_row * problem.lhs.pitch + blockIdx.z * MaxAccumulation;
    rhs += blockIdx.z * MaxAccumulation * problem.rhs.pitch + rhs_col;


    static_assert(MaxAccumulation % StrideUnit == 0);
#pragma unroll
    for (uint32_t lhs_stride_col{0}; lhs_stride_col < MaxAccumulation; lhs_stride_col += StrideUnit) {
        auto const& rhs_stride_row = lhs_stride_col;

#pragma unroll
        for (uint32_t col_in_stride{0}; col_in_stride < StrideUnit; col_in_stride += BlockCols) {
            auto col_in_cache = col_in_stride + col_in_block;
            lhs_cache[row_in_block][col_in_cache] = lhs[lhs_stride_col + col_in_cache];
        }

        static_assert(StrideUnit % BlockCols == 0);
        if (lhs_stride_col != 0) {
            __syncthreads();
        }
        static_assert(StrideUnit % BlockRows == 0);
#pragma unroll
        for (uint32_t i{0}; i < (StrideUnit / BlockRows); i++) {
            auto row_in_cache = i + row_in_block * (StrideUnit / BlockRows);
            rhs_cache[row_in_cache][col_in_block] = rhs[(rhs_stride_row + row_in_cache) * problem.rhs.pitch];
        }
        __syncthreads();

#pragma unroll
        for (uint32_t in_stride{0}; in_stride < StrideUnit; in_stride++) {
            result_cache += lhs_cache[row_in_block][in_stride] * rhs_cache[in_stride][col_in_block];
        }
    }

    atomicAdd(&result[result_row * problem.result.pitch + result_col], result_cache);
}


template <>
void gemm<2, 0, LhsRows / 2, LhsCols, RhsCols>(
    CudaMM::ProblemData<LhsRows / 2, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    CUDA_CHECK(cudaMemset2DAsync(
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        0,
        sizeof(Element) * problem.lhs.cols, problem.lhs.rows,
        0));

    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.lhs.data.get(), sizeof(Element) * problem.lhs.pitch,
        lhs, sizeof(Element) * problem.lhs.cols,
        sizeof(Element) * problem.lhs.cols, problem.lhs.rows,
        cudaMemcpyDefault,
        0));
    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.rhs.data.get(), sizeof(Element) * problem.rhs.pitch,
        rhs, sizeof(Element) * problem.rhs.cols,
        sizeof(Element) * problem.rhs.cols, problem.rhs.rows,
        cudaMemcpyDefault,
        0));

    static_assert(problem.lhs.cols % MaxAccumulation == 0);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_impl<4096, 8192, 8192>, cudaFuncAttributeMaxDynamicSharedMemorySize, 0x10000));

    gemm_impl<<<{problem.result.cols / BlockCols,
                    problem.result.rows / BlockRows,
                    problem.lhs.cols / MaxAccumulation},
        {BlockCols, BlockRows},
        sizeof(Element) * (size_t{BlockCols} + BlockRows) * StrideUnit>>>(

        toKernel(problem),
        problem.lhs.data.get(), problem.rhs.data.get(), problem.result.data.get());


    CUDA_CHECK(cudaMemcpy2DAsync(
        result, sizeof(Element) * problem.result.cols,
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        sizeof(Element) * problem.result.cols, problem.result.rows,
        cudaMemcpyDefault,
        0));
}

template <>
void gemm<2, 1, LhsRows / 2, LhsCols, RhsCols>(
    CudaMM::ProblemData<LhsRows / 2, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    gemm<2, 0, LhsRows / 2, LhsCols, RhsCols>(
        problem,
        lhs, rhs, result);
}

}  // namespace Kernel

}  // namespace CudaMM

template void CudaMM::gemm<LhsRows, LhsCols, RhsCols>(
    CudaMM::DeviceData<LhsRows, LhsCols, RhsCols>&,
    const Element* lhs, const Element* rhs, Element* result);

