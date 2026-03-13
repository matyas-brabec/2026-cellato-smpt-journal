#ifndef BRIAN_PRETTY_PRINT_HPP
#define BRIAN_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace brian {

using print_config = cellato::memory::grids::standard::print_config<brian_cell_state>;

struct brian_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(brian_cell_state::dead, "\033[1;31m.\033[0m")
            .with(brian_cell_state::dying, "\033[1;33m*\033[0m")
            .with(brian_cell_state::alive, "\033[1;32m#\033[0m");
    }
};

} // namespace brian

#endif // BRIAN_PRETTY_PRINT_HPP