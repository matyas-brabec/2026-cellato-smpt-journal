#ifndef TRAFFIC_PRETTY_PRINT_HPP
#define TRAFFIC_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace traffic {

using print_config = cellato::memory::grids::standard::print_config<traffic_cell_state>;

struct traffic_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(traffic_cell_state::empty, "\033[90m.\033[0m")  // Dark grey for empty ground (almost invisible)
            .with(traffic_cell_state::red_car, "\033[1;31m>\033[0m") // Bright red for red cars
            .with(traffic_cell_state::blue_car, "\033[1;34mv\033[0m"); // Bright blue for blue cars
    }
};

}

#endif // TRAFFIC_PRETTY_PRINT_HPP
