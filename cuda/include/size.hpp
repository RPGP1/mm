#pragma once

#include <cstdint>


constexpr uint32_t LhsRows{105248}, LhsCols{49152}, RhsCols{52736};
constexpr auto RhsRows = LhsCols, ResultRows = LhsRows, ResultCols = RhsCols;
