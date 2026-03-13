#ifndef REFERENCE_IMPL_MANAGER_HPP
#define REFERENCE_IMPL_MANAGER_HPP

#include <vector>
#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include <cuda_runtime.h>
#include "../traversers/cuda_utils.cuh"

#include "./run_params.hpp"
#include "./experiment_report.hpp"
#include "memory/standard_grid.hpp"

namespace cellato::run {

template <typename RunnerT, typename CellStateT>
class reference_impl_manager {
public:
    using runner_t = RunnerT;
    using cell_state_t = CellStateT;
    using standard_grid_t = cellato::memory::grids::standard::grid<cell_state_t>;
    
    constexpr static int margin = 1;
    
    reference_impl_manager() = default;
    
    experiment_report run_experiment(const run_params& params, const std::vector<cell_state_t>& initial_state) {
        experiment_report report;
        report.params = params;
        
        for (int i = 0; i < params.warmup_rounds + params.rounds; ++i) {
            auto [duration, checksum] = run_round(i, params, initial_state);
            
            if (i >= params.warmup_rounds) {
                report.execution_times_ms.push_back(duration);
                report.checksums.push_back(checksum);
            }
        }
        
        return report;
    }
    
private:
    /**
     * @brief The main method for a single run, now streamlined to handle only
     * common setup and result processing logic.
     */
    std::tuple<double, std::string> run_round(int round, const run_params& params, 
                                              const std::vector<cell_state_t>& initial_state) {
        // --- 1. Common Setup ---
        log_round_start(round, params);
        
        standard_grid_t grid(params.x_size, params.y_size);
        std::copy(initial_state.begin(), initial_state.end(), grid.data());

        runner_t runner;
        runner.init(grid.data(), params);
        
        // --- 2. Device-Specific Execution & Timing ---
        double duration_ms = 0.0;
        if (params.device == "CUDA") {
            duration_ms = time_cuda_run(runner, params);
        } else {
            duration_ms = time_cpu_run(runner, params);
        }
        
        // --- 3. Common Result Processing ---
        auto result = runner.fetch_result();
        standard_grid_t result_grid(grid.x_size_physical(), grid.y_size_physical());
        std::copy(result.begin(), result.end(), result_grid.data());
        
        std::string checksum = result_grid.get_checksum();

        return { duration_ms, checksum };
    }

    /**
     * @brief Handles CUDA-specific initialization, execution, and timing.
     * @return The execution time in milliseconds.
     */
    double time_cuda_run(runner_t& runner, const run_params& params) {
        runner.init_cuda(); // Perform CUDA-specific initialization

        cudaEvent_t start, stop;
        CUCH(cudaEventCreate(&start));
        CUCH(cudaEventCreate(&stop));

        CUCH(cudaEventRecord(start));
        runner.run_on_cuda(params.steps);
        CUCH(cudaEventRecord(stop));

        CUCH(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUCH(cudaEventElapsedTime(&milliseconds, start, stop));

        CUCH(cudaEventDestroy(start));
        CUCH(cudaEventDestroy(stop));

        CUCH(cudaDeviceSynchronize());

        return static_cast<double>(milliseconds);
    }

    /**
     * @brief Handles CPU-specific execution and timing.
     * @return The execution time in milliseconds.
     */
    double time_cpu_run(runner_t& runner, const run_params& params) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        runner.run(params.steps);
        const auto end_time = std::chrono::high_resolution_clock::now();
        
        const std::chrono::duration<double, std::milli> duration = end_time - start_time;
        return duration.count();
    }

    /**
     * @brief Helper to log the start of a round.
     */
    void log_round_start(int round, const run_params& params) {
        if (round < params.warmup_rounds) {
            std::cerr << "\nWarmup round: " << round << "\n";
        } else {
            std::cerr << "\nRound: " << round - params.warmup_rounds << "\n";
        }
    }
};

} // namespace cellato::run

#endif // REFERENCE_IMPL_MANAGER_HPP