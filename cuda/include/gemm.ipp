#pragma once

#include "gemm.hpp"

#include "kernel.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <tuple>
#include <utility>


namespace CudaMM
{

namespace
{

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex DeviceCount, DeviceIndex Device>
auto allocateDeviceData_impl(
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols);

template <uint32_t Rows, uint32_t Cols>
void deviceAllocPitch(Matrix2D<Rows, Cols>& matrix);


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
void cudaGemm_helper(
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result,
    std::integer_sequence<DeviceIndex, Device...>);

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
cudaStream_t cudaGemm_impl(
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result,
    uint32_t& row_offset);

}  // namespace


template <uint32_t Rows, uint32_t Cols>
std::shared_ptr<Element[]> hostAlloc(uint32_t rows, uint32_t cols)
{
    Element* tmp;

    auto constexpr Size = sizeof(Element) * Rows * Cols;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&tmp),
        (Size != 0 ? Size : sizeof(Element) * rows * cols),
        cudaHostAllocDefault));

    constexpr auto deleter = [](Element ptr[]) {
        CUDA_CHECK(cudaFreeHost(ptr));
    };

    return {tmp, deleter};
}


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
DeviceData<LhsRows, LhsCols, RhsCols> allocateDeviceData(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols)
{
    return {allocateDeviceData_helper<LhsRows, LhsCols, RhsCols>(lhs_rows, lhs_cols, rhs_cols, std::make_integer_sequence<DeviceIndex, Devices>{})};
}

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void cudaGemm(
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result)
{
    cudaGemm_helper(data, lhs, rhs, result, std::make_integer_sequence<DeviceIndex, Devices>{});
}

namespace
{

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
auto allocateDeviceData_helper(
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    std::integer_sequence<DeviceIndex, Device...>)
{
    return std::make_tuple(allocateDeviceData_impl<LhsRows, LhsCols, RhsCols, sizeof...(Device), Device>(lhs_rows, lhs_cols, rhs_cols)...);
}

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex DeviceCount, DeviceIndex Device>
auto allocateDeviceData_impl(
    [[maybe_unused]] uint32_t lhs_rows, [[maybe_unused]] uint32_t lhs_cols, [[maybe_unused]] uint32_t rhs_cols)
{
    constexpr uint32_t AssignedRows{LhsRows / DeviceCount + (LhsRows % DeviceCount > static_cast<uint32_t>(Device))};

    CUDA_CHECK(cudaSetDevice(Device));

    ProblemData<AssignedRows, LhsCols, RhsCols> problem;

    if constexpr (AssignedRows == 0) {
        uint32_t assigned_rows{lhs_rows / DeviceCount + (lhs_rows % DeviceCount > static_cast<uint32_t>(Device))};
        problem.lhs.rows = assigned_rows;
        problem.result.rows = assigned_rows;
    }
    if constexpr (LhsCols == 0) {
        problem.lhs.cols = lhs_cols;
        problem.rhs.rows = lhs_cols;
    }
    if constexpr (RhsCols == 0) {
        problem.rhs.cols = rhs_cols;
        problem.result.cols = rhs_cols;
    }

    deviceAllocPitch(problem.lhs);
    deviceAllocPitch(problem.rhs);
    deviceAllocPitch(problem.result);

    return problem;
}

template <uint32_t Rows, uint32_t Cols>
void deviceAllocPitch(Matrix2D<Rows, Cols>& matrix)
{
    Element* tmp;

    CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&tmp), &matrix.pitch, sizeof(Element) * matrix.cols, matrix.rows));

    assert(matrix.pitch % sizeof(Element) == 0);
    matrix.pitch /= sizeof(Element);  // convert pitch in bytes to one in elems

    constexpr auto deleter = [](Element ptr[]) {
        CUDA_CHECK(cudaFree(ptr));
    };

    matrix.data = {tmp, deleter};
}


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
void cudaGemm_helper(
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result,
    std::integer_sequence<DeviceIndex, Device...>)
{
    using std::get;

    uint32_t row_offset{0};

    std::array<cudaStream_t, sizeof...(Device)> streams{cudaGemm_impl<sizeof...(Device), Device>(data, lhs, rhs, result, row_offset)...};
    (..., CUDA_CHECK(cudaStreamSynchronize(get<Device>(streams))));
}

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
cudaStream_t cudaGemm_impl(
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result,
    uint32_t& row_offset)
{
    using std::get;

    cudaStream_t stream;

    CUDA_CHECK(cudaSetDevice(Device));
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto& problem = get<Device>(data.data);

    size_t lhs_offset = size_t{row_offset} * problem.lhs.cols,
           result_offset = size_t{row_offset} * problem.result.cols;


    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.lhs.data.get(), sizeof(Element) * problem.lhs.pitch,
        lhs.get() + lhs_offset, sizeof(Element) * problem.lhs.cols,
        sizeof(Element) * problem.lhs.cols, problem.lhs.rows,
        cudaMemcpyDefault,
        stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        problem.rhs.data.get(), sizeof(Element) * problem.rhs.pitch,
        rhs.get(), sizeof(Element) * problem.rhs.cols,
        sizeof(Element) * problem.rhs.cols, problem.rhs.rows,
        cudaMemcpyDefault,
        stream));


    kernel(
        problem.lhs.data.get(), problem.rhs.data.get(), problem.result.data.get(),
        problem.lhs.rows, problem.lhs.cols, problem.rhs.cols,
        problem.lhs.pitch, problem.rhs.pitch, problem.result.pitch,
        {(problem.result.cols + BlockLength - 1) / BlockLength,
            (problem.result.rows + BlockLength - 1) / BlockLength},
        {BlockLength, BlockLength}, 0, stream);


    CUDA_CHECK(cudaMemcpy2DAsync(
        result.get() + result_offset, sizeof(Element) * problem.result.cols,
        problem.result.data.get(), sizeof(Element) * problem.result.pitch,
        sizeof(Element) * problem.result.cols, problem.result.rows,
        cudaMemcpyDefault,
        stream));

    row_offset += problem.lhs.rows;

    return stream;
}

}  // namespace

}  // namespace CudaMM
