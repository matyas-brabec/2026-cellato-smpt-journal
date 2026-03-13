#ifndef MAZE_DATA_INIT_HPP
#define MAZE_DATA_INIT_HPP

#include <vector>
#include <random>

#include "experiments/run_params.hpp"

namespace maze {

struct maze_random_init {
    static std::vector<maze_cell_state> init(cellato::run::run_params& params) {
        std::vector<maze_cell_state> initial_state(params.x_size * params.y_size);
        
        // Probabilities for each cell state
        std::vector<std::tuple<maze_cell_state, double>> probabilities = {
            {maze_cell_state::empty, 0.80},
            {maze_cell_state::wall, 0.20}
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

#endif // MAZE_DATA_INIT_HPP
