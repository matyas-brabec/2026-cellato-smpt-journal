#ifndef MAZE_PRETTY_PRINT_HPP
#define MAZE_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace maze {

using print_config = cellato::memory::grids::standard::print_config<maze_cell_state>;

struct maze_pretty_print {
    static print_config get_config() {
        return print_config()
            .with(maze_cell_state::empty, "\033[90m.\033[0m")  // Dark grey for empty ground (almost invisible)
            .with(maze_cell_state::wall, "\033[93m#\033[0m"); // Yellow for walls
    }
};

}

#endif // MAZE_PRETTY_PRINT_HPP
