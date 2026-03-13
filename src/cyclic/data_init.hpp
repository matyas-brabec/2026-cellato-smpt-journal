#ifndef CYCLIC_DATA_INIT_HPP
#define CYCLIC_DATA_INIT_HPP

#include <vector>
#include <random>

#include "experiments/run_params.hpp"

namespace cyclic {

struct cyclic_random_init {
    static std::vector<cyclic_cell_state> init(cellato::run::run_params& params) {
        std::vector<cyclic_cell_state> initial_state(params.x_size * params.y_size);
        
        // Probabilities for each cell state
        std::vector<std::tuple<cyclic_cell_state, double>> probabilities;

        for (int state = 0; state < STATES; ++state) {
            probabilities.emplace_back(state, 1.0 / STATES);
        }
        
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

#endif // CYCLIC_DATA_INIT_HPP
