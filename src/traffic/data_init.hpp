#ifndef TRAFFIC_DATA_INIT_HPP
#define TRAFFIC_DATA_INIT_HPP

#include <vector>
#include <random>

#include "experiments/run_params.hpp"

namespace traffic {

struct traffic_random_init {
    static std::vector<traffic_cell_state> init(cellato::run::run_params& params) {
        std::vector<traffic_cell_state> initial_state(params.x_size * params.y_size);
        
        // Probabilities for each cell state
        std::vector<std::tuple<traffic_cell_state, double>> probabilities = {
            {traffic_cell_state::empty, 0.50},
            {traffic_cell_state::red_car, 0.25},
            {traffic_cell_state::blue_car, 0.25}
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

#endif // TRAFFIC_DATA_INIT_HPP
