#pragma once

#include "definition.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>


namespace CudaMM
{

template <uint32_t Rows_, uint32_t Cols_>
struct Matrix2D {
    static constexpr uint32_t Rows = Rows_, Cols = Cols_;

    std::shared_ptr<Element> data;
    size_t pitch;

    static constexpr uint32_t rows = Rows_, cols = Cols_;

    Matrix2D(uint32_t rows, uint32_t cols);
};

template <uint32_t Rows_>
struct Matrix2D<Rows_, 0> {
    static constexpr uint32_t Rows = Rows_, Cols = 0;

    std::shared_ptr<Element> data;
    size_t pitch;

    static constexpr uint32_t rows = Rows_;
    uint32_t cols;

    Matrix2D(uint32_t rows, uint32_t cols);
};

template <uint32_t Cols_>
struct Matrix2D<0, Cols_> {
    static constexpr uint32_t Rows = 0, Cols = Cols_;

    std::shared_ptr<Element> data;
    size_t pitch;

    uint32_t rows;
    static constexpr uint32_t cols = Cols_;

    Matrix2D(uint32_t rows, uint32_t cols);
};

template <>
struct Matrix2D<0, 0> {
    static constexpr uint32_t Rows = 0, Cols = 0;

    std::shared_ptr<Element> data;
    size_t pitch;

    uint32_t rows, cols;

    Matrix2D(uint32_t rows, uint32_t cols);
};

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
struct ProblemData {
    Matrix2D<LhsRows, LhsCols> lhs;
    Matrix2D<LhsCols, RhsCols> rhs;
    Matrix2D<LhsRows, RhsCols> result;

    ProblemData(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols);
};

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
struct DeviceData;


template <uint32_t LhsRows = 0, uint32_t LhsCols = 0, uint32_t RhsCols = 0>
DeviceData<LhsRows, LhsCols, RhsCols> allocateDeviceData(
    uint32_t lhs_rows = LhsRows, uint32_t lhs_cols = LhsCols, uint32_t rhs_cols = RhsCols);

template <uint32_t LhsRows, uint32_t LhsCols, uint32_t RhsCols>
void gemm(
    DeviceData<LhsRows, LhsCols, RhsCols>&,
    const Element* lhs, const Element* rhs, Element* result);

}  // namespace CudaMM
