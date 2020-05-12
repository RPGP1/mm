#include "definition.hpp"

#include "mm/tester.hpp"

#include <cuda_runtime_api.h>
#include <cuda.h>


int main(int argc, char* argv[])
{
    MM::Tester<Element> tester(argc, argv);

    auto& problem = tester.problem();

    Element *lhs, rhs, result;
    cudaMalloc(reinterpret_cast<void**>(&lhs), size_t{problem.lhs_rows()} * problem.lhs_cols());
    cudaMalloc(reinterpret_cast<void**>(&rhs), size_t{problem.rhs_rows()} * problem.rhs_cols());
    cudaMalloc(reinterpret_cast<void**>(&result), size_t{problem.lhs_rows()} * problem.rhs_cols());
}
