#pragma once

#include "kernel.hpp"

#include "error.hpp"
#include "gemm.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>


namespace CudaMM
{

constexpr uint32_t BlockLength{8};  // means (BlockLength x BlockLength) block


namespace Kernel
{

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void gemm(
    ProblemData<LhsRows, LhsCols, RhsCols>& problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    cudaStream_t stream;

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


    // gemm_impl<<<{(problem.result.cols + BlockLength - 1) / BlockLength,
    //                   (problem.result.rows + BlockLength - 1) / BlockLength},
    //     {BlockLength, BlockLength}, 0, stream>>>(

    //     problem.lhs.data.get(), problem.rhs.data.get(), problem.result.data.get(),
    //     problem.lhs.rows, problem.lhs.cols, problem.rhs.cols,
    //     problem.lhs.pitch, problem.rhs.pitch, problem.result.pitch);


    CUDA_CHECK(cudaMemcpy2DAsync(
        result, sizeof(Element) * problem.result.cols,
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        sizeof(Element) * problem.result.cols, problem.result.rows,
        cudaMemcpyDefault,
        stream));
}

}  // namespace Kernel

}  // namespace CudaMM
