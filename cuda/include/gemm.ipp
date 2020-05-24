#pragma once

#include "gemm.hpp"

#include "error.hpp"
#include "kernel.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <functional>
#include <thread>
#include <tuple>
#include <utility>


namespace CudaMM
{

namespace
{

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
auto allocateDeviceData_helper(
    std::integer_sequence<DeviceIndex, Device...>,
    uint32_t lhs_rows = LhsRows, uint32_t lhs_cols = LhsCols, uint32_t rhs_cols = RhsCols);

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
auto allocateDeviceData_impl(
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols);

template <uint32_t Rows, uint32_t Cols>
void deviceAllocPitch(Matrix2D<Rows, Cols>& matrix);


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
void gemm_helper(
    std::integer_sequence<DeviceIndex, Device...>,
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    const Element* lhs, const Element* rhs, Element* result);

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void kernel_gemm_wrapper(
    std::reference_wrapper<ProblemData<LhsRows, LhsCols, RhsCols>> problem,
    const Element* lhs, const Element* rhs, Element* result);

struct nop {
    template <class... T>
    nop(T&&...)
    {
    }
};

}  // namespace

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
struct DeviceData {
    decltype(allocateDeviceData_helper<LhsRows, LhsCols, RhsCols>(std::make_integer_sequence<DeviceIndex, Devices>{}))
        data;
};


template <uint32_t Rows, uint32_t Cols>
Matrix2D<Rows, Cols>::Matrix2D(uint32_t, uint32_t)
    : pitch{Cols}
{
}

template <uint32_t Rows>
Matrix2D<Rows, 0>::Matrix2D(uint32_t, uint32_t cols)
    : pitch{cols}, cols{cols}
{
}

template <uint32_t Cols>
Matrix2D<0, Cols>::Matrix2D(uint32_t rows, uint32_t)
    : pitch{Cols}, rows{rows}
{
}


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
ProblemData<LhsRows, LhsCols, RhsCols>::ProblemData(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols)
    : lhs{lhs_rows, lhs_cols}, rhs{lhs_cols, rhs_cols}, result{lhs_rows, rhs_cols}
{
}


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
DeviceData<LhsRows, LhsCols, RhsCols> allocateDeviceData(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols)
{
    return {allocateDeviceData_helper<LhsRows, LhsCols, RhsCols>(
        std::make_integer_sequence<DeviceIndex, Devices>{}, lhs_rows, lhs_cols, rhs_cols)};
}

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void gemm(
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    const Element* lhs, const Element* rhs, Element* result)
{
    gemm_helper(std::make_integer_sequence<DeviceIndex, Devices>{},
        data, lhs, rhs, result);
}


namespace
{

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
auto allocateDeviceData_helper(
    std::integer_sequence<DeviceIndex, Device...>,
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols)
{
    return std::make_tuple(
        allocateDeviceData_impl<sizeof...(Device), Device, LhsRows, LhsCols, RhsCols>(lhs_rows, lhs_cols, rhs_cols)...);
}

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
auto allocateDeviceData_impl(
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols)
{
    constexpr uint32_t AssignedRows{LhsRows / DeviceCount + (LhsRows % DeviceCount > static_cast<uint32_t>(Device))};

    CUDA_CHECK(cudaSetDevice(Device));

    ProblemData<AssignedRows, LhsCols, RhsCols> problem{
        lhs_rows / DeviceCount + (lhs_rows % DeviceCount > static_cast<uint32_t>(Device)), lhs_cols, rhs_cols};

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

    matrix.data = {tmp, [](Element ptr[]) {
                       CUDA_CHECK(cudaFree(ptr));
                   }};
}


template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
void gemm_helper(
    std::integer_sequence<DeviceIndex, Device...>,
    DeviceData<LhsRows, LhsCols, RhsCols>& data,
    const Element* lhs, const Element* rhs, Element* result)
{
    using std::get;

    std::array<size_t, sizeof...(Device) + 1> row_offsets{0};
    nop{(get<Device + 1>(row_offsets) = get<Device>(row_offsets) + get<Device>(data.data).lhs.rows)...};

    std::array<std::thread, sizeof...(Device)> thread_pool{
        std::thread{
            kernel_gemm_wrapper<sizeof...(Device), Device,
                decltype(get<Device>(data.data).lhs)::Rows,
                decltype(get<Device>(data.data).lhs)::Cols,
                decltype(get<Device>(data.data).rhs)::Cols>,
            std::ref(get<Device>(data.data)),
            lhs + get<Device>(row_offsets) * get<Device>(data.data).lhs.cols,
            rhs,
            result + get<Device>(row_offsets) * get<Device>(data.data).result.cols}...};

    nop{(get<Device>(thread_pool).join(), 0)...};

    nop{(CUDA_CHECK(cudaSetDevice(Device)), CUDA_CHECK(cudaDeviceSynchronize()), 0)...};
}

template <DeviceIndex DeviceCount, DeviceIndex Device, uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void kernel_gemm_wrapper(
    std::reference_wrapper<ProblemData<LhsRows, LhsCols, RhsCols>> problem,
    const Element* lhs, const Element* rhs, Element* result)
{
    CUDA_CHECK(cudaSetDevice(Device));
    Kernel::gemm<DeviceCount, Device>(problem.get(), lhs, rhs, result);
}

}  // namespace

}  // namespace CudaMM
