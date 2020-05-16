#pragma once

#include "definition.hpp"
#include "error.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>


namespace CudaMM
{

using DeviceIndex = int;
constexpr DeviceIndex Devices{2};

template <uint32_t Rows_, uint32_t Cols_>
struct Matrix2D {
    static constexpr auto Rows = Rows_, Cols = Cols_;

    std::shared_ptr<Element[]> data;
    static constexpr uint32_t rows = Rows_, cols = Cols_;
    size_t pitch = cols;
};

template <uint32_t Rows_>
struct Matrix2D<Rows_, 0> {
    static constexpr auto Rows = Rows_, Cols = 0;

    std::shared_ptr<Element[]> data;
    static constexpr uint32_t rows = Rows_;
    uint32_t cols{0};
    size_t pitch = cols;
};

template <uint32_t Cols_>
struct Matrix2D<0, Cols_> {
    static constexpr auto Rows = 0, Cols = Cols_;

    std::shared_ptr<Element[]> data;
    uint32_t rows{0};
    static constexpr uint32_t cols = Cols_;
    size_t pitch = cols;
};

template <>
struct Matrix2D<0, 0> {
    static constexpr auto Rows = 0, Cols = 0;

    std::shared_ptr<Element[]> data;
    uint32_t rows{0}, cols{0};
    size_t pitch = cols;
};

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
struct ProblemData {
    Matrix2D<LhsRows, LhsCols> lhs;
    Matrix2D<LhsCols, RhsCols> rhs;
    Matrix2D<LhsRows, RhsCols> result;
};


namespace
{

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols, DeviceIndex... Device>
auto allocateDeviceData_helper(
    uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    std::integer_sequence<DeviceIndex, Device...>);

}  // namespace

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
struct DeviceData {
    decltype(allocateDeviceData_helper<LhsRows, LhsCols, RhsCols>(LhsRows, LhsCols, RhsCols, std::make_integer_sequence<DeviceIndex, Devices>{}))
        data;
};


template <uint32_t Rows_ = 0, uint32_t Cols_ = 0>
std::shared_ptr<Element[]> hostAlloc(uint32_t rows = Rows_, uint32_t cols = Cols_);

template <uint32_t LhsRows = 0, uint32_t LhsCols = 0, uint32_t RhsCols = 0>
DeviceData<LhsRows, LhsCols, RhsCols> allocateDeviceData(
    uint32_t lhs_rows = LhsRows, uint32_t lhs_cols = LhsCols, uint32_t rhs_cols = RhsCols);

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void cudaGemm(
    DeviceData<LhsRows, LhsCols, RhsCols>&,
    std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result);

}  // namespace CudaMM

#include "gemm.ipp"
