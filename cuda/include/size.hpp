#pragma once

#include <cstdint>


constexpr uint32_t LhsRows{64064}, LhsCols{65536}, RhsCols{64000};
constexpr auto RhsRows = LhsCols, ResultRows = LhsRows, ResultCols = RhsCols;
