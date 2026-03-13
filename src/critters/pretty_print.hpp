#ifndef CRITTERS_PRETTY_PRINT_HPP
#define CRITTERS_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace critters {

using print_config = cellato::memory::grids::standard::print_config<critters_cell_state>;

struct critters_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(critters_cell_state::dead, "\033[90m.\033[0m")  // Dark grey for dead cells
            .with(critters_cell_state::alive, "\033[1;32mO\033[0m"); // Bright green for alive cells
    }
};

}

#endif // CRITTERS_PRETTY_PRINT_HPP
