#pragma once

#include <cstdint>


constexpr uint32_t LhsRows{8192}, LhsCols{8192}, RhsCols{8192};
constexpr auto RhsRows = LhsCols, ResultRows = LhsRows, ResultCols = RhsCols;
