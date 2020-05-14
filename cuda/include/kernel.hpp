#pragma once

#include "definition.hpp"

#include <cstdint>


void cudaGemm(uint32_t lhs_rows, uint32_t lhs_cols, uint32_t rhs_cols,
    Element* lhs, Element* rhs, Element* result);
