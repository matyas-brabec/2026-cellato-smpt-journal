#ifndef MAZE_CONFIG_HPP
#define MAZE_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

#include <string>
namespace maze {

struct config {

    static constexpr char name[] = "maze";

    static constexpr double average_halo_radius = 1.0;

    using algorithm = maze_algorithm;
    
    using cell_state = maze_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::empty, cell_state::wall>;

    using pretty_print = maze_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = maze_random_init;
    };
};

}

#endif // MAZE_CONFIG_HPP
