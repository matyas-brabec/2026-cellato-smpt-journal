#ifndef BRIAN_INIT_HPP
#define BRIAN_INIT_HPP

#include <vector>
#include <tuple>
#include "memory/grid_utils.hpp"
#include "experiments/run_params.hpp"
#include "./algorithm.hpp"

namespace brian {

struct brian_random_init {
    static std::vector<brian_cell_state> init(cellato::run::run_params& params) {

        std::vector<brian_cell_state> initial_state(params.x_size * params.y_size);

        cellato::memory::grids::utils::generate_random_grid(
            initial_state,
            params.y_size, params.x_size,
            brian_cell_state::alive, 0.2,
            brian_cell_state::dead,
            params.seed
        );

        return initial_state;
    }
};

}

#endif // BRIAN_INIT_HPP