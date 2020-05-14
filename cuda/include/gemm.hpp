#pragma once

#include "definition.hpp"
#include "error.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>


constexpr int Devices{2};

struct Matrix2D {
    std::shared_ptr<Element[]> data;
    uint32_t rows, cols;
    size_t pitch = cols;
};

struct ProblemData {
    Matrix2D lhs, rhs, result;
};

using DeviceData = std::array<ProblemData, Devices>;


std::shared_ptr<Element[]> hostAlloc(uint32_t rows, uint32_t cols);

DeviceData allocateDeviceData(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols);

void cudaGemm(DeviceData&, std::shared_ptr<Element[]> lhs, std::shared_ptr<Element[]> rhs, std::shared_ptr<Element[]> result);
