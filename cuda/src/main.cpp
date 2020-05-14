#include "check_device_props.hpp"
#include "definition.hpp"
#include "error.hpp"
#include "gemm.hpp"

#include "mm/tester.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <memory>


int main(int argc, char* argv[])
try {
    checkDeviceProps();

    MM::Tester<Element> tester(argc, argv);


    auto& problem = tester.problem();
    std::cout << "# Problem: " << tester.problem_file_path() << std::endl
              << "# Size: lhs[ " << problem.lhs_rows() << " ][ " << problem.lhs_cols() << " ]"
              << " * rhs[ " << problem.rhs_rows() << " ][ " << problem.rhs_cols() << " ]" << std::endl;


    // memory allocation
    auto lhs = hostAlloc(problem.lhs_rows(), problem.lhs_cols()),
         rhs = hostAlloc(problem.rhs_rows(), problem.rhs_cols()),
         result = hostAlloc(problem.lhs_rows(), problem.rhs_cols());
    problem.get(lhs.get(), rhs.get(), problem.lhs_cols(), problem.rhs_cols());

    auto device_data = allocateDeviceData(problem.lhs_rows(), problem.lhs_cols(), problem.rhs_cols());


    // calculation with timer
    {
        auto timer = tester.timer();

        cudaGemm(device_data, lhs, rhs, result);
    }

    using millisec_double = std::chrono::duration<double, std::milli>;
    using std::chrono::duration_cast;
    std::cout << "# Elapsed [ms]: " << duration_cast<millisec_double>(tester.elapsed()).count() << std::endl;


    MM::Problem::Result<Element>::loose_standard_digits = 3;
    auto score = problem.score(result.get(), problem.rhs_cols(), [](uint32_t row, uint32_t col, Element calced, Element answer) {
        std::cerr << "### @( " << row << ", " << col << " ) "
                  << calced << " != Ans = " << answer << std::endl;
    });
    std::cout << "# Difference count (Strict): " << score.strict_violations << std::endl
              << "# Difference count (Loose): " << score.loose_violations << std::endl
              << "# Max Wrongness: " << score.max_difference << std::endl;

} catch (MM::TesterImpl::CommandLineHelp const& e) {
    std::cerr << e.what() << std::endl;
    return 0;
} catch (MM::TesterImpl::BadCommandLineArguments const& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
