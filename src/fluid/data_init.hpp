#ifndef FLUID_DATA_INIT_HPP
#define FLUID_DATA_INIT_HPP

#include <vector>
#include <random>

#include "experiments/run_params.hpp"

namespace fluid {

struct fluid_random_init {
    static std::vector<fluid_cell_state> init(cellato::run::run_params& params) {
        std::vector<fluid_cell_state> initial_state(params.x_size * params.y_size);
        
        // Probabilities for each cell state
        std::vector<std::tuple<fluid_cell_state, double>> probabilities = {
            { 0, 1.0 / 16.0 },
            { 1, 1.0 / 16.0 },
            { 2, 1.0 / 16.0 },
            { 3, 1.0 / 16.0 },
            { 4, 1.0 / 16.0 },
            { 5, 1.0 / 16.0 },
            { 6, 1.0 / 16.0 },
            { 7, 1.0 / 16.0 },
            { 8, 1.0 / 16.0 },
            { 9, 1.0 / 16.0 },
            {10, 1.0 / 16.0 },
            {11, 1.0 / 16.0 },
            {12, 1.0 / 16.0 },
            {13, 1.0 / 16.0 },
            {14, 1.0 / 16.0 },
            {15, 1.0 / 16.0 }
        };
        
        // Generate random grid using utility
        cellato::memory::grids::utils::generate_random_grid(
            initial_state,
            params.y_size, params.x_size,
            probabilities
        );
        
        return initial_state;
    }
};

}

#endif // FLUID_DATA_INIT_HPP
