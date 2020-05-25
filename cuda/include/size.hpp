#pragma once

#include <cstdint>


constexpr uint32_t LhsRows{32768}, LhsCols{32768}, RhsCols{32768};
constexpr auto RhsRows = LhsCols, ResultRows = LhsRows, ResultCols = RhsCols;
