#include "definition.hpp"
#include "kernel.hpp"

#include "mm/tester.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <iostream>
#include <memory>


int main(int argc, char* argv[])
try {
    MM::Tester<Element> tester(argc, argv);


    auto& problem = tester.problem();
    std::cout << "# Problem: " << tester.problem_file_path() << std::endl
              << "# Size: lhs[ " << problem.lhs_rows() << " ][ " << problem.lhs_cols() << " ]"
              << " * rhs[ " << problem.rhs_rows() << " ][ " << problem.rhs_cols() << " ]" << std::endl;


    // memory allocation
    std::shared_ptr<Element[]> host_lhs{new Element[size_t{problem.lhs_rows()} * problem.lhs_cols()]},
        host_rhs{new Element[size_t{problem.rhs_rows()} * problem.rhs_cols()]},
        host_result{new Element[size_t{problem.lhs_rows()} * problem.rhs_cols()]};

    std::shared_ptr<Element[]> lhs, rhs, result;
    {
        Element *tmp_lhs, *tmp_rhs, *tmp_result;
        cudaMalloc(reinterpret_cast<void**>(&tmp_lhs), sizeof(Element) * problem.lhs_rows() * problem.lhs_cols());
        cudaMalloc(reinterpret_cast<void**>(&tmp_rhs), sizeof(Element) * problem.rhs_rows() * problem.rhs_cols());
        cudaMalloc(reinterpret_cast<void**>(&tmp_result), sizeof(Element) * problem.lhs_rows() * problem.rhs_cols());

        auto deleter = [](Element ptr[]) {
            cudaFree(ptr);
        };
        lhs = {tmp_lhs, deleter};
        rhs = {tmp_rhs, deleter};
        result = {tmp_result, deleter};
    }


    // send data to device
    problem.get(&host_lhs[0], &host_rhs[0], problem.lhs_cols(), problem.rhs_cols());
    for (uint32_t row{0}; row < problem.lhs_rows(); row++) {
        for (uint32_t col{0}; col < problem.lhs_cols(); col++) {
            std::cout << "(" << row << ", " << col << ") = " << host_lhs[row * problem.lhs_cols() + col] << std::endl;
        }
    }
    cudaMemcpy(&lhs[0], &host_lhs[0], sizeof(Element) * problem.lhs_rows() * problem.lhs_cols(), cudaMemcpyHostToDevice);
    cudaMemcpy(&rhs[0], &host_rhs[0], sizeof(Element) * problem.rhs_rows() * problem.rhs_cols(), cudaMemcpyHostToDevice);


    // calculation with timer
    {
        auto timer = tester.timer();

        cudaGemm(problem.lhs_rows(), problem.lhs_cols(), problem.rhs_cols(),
            &lhs[0], &rhs[0], &result[0]);
    }

    using millisec_double = std::chrono::duration<double, std::milli>;
    using std::chrono::duration_cast;
    std::cout << "# Elapsed: " << duration_cast<millisec_double>(tester.elapsed()).count() << std::endl;


    // send data to host, score
    cudaMemcpy(&host_result[0], &result[0], sizeof(Element) * problem.lhs_rows() * problem.rhs_cols(), cudaMemcpyDeviceToHost);

    MM::Problem::Result<Element>::loose_standard_digits = 3;
    auto score = problem.score(&host_result[0], problem.rhs_cols(), [](uint32_t row, uint32_t col, Element calced, Element answer) {
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
