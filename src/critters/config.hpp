#ifndef CRITTERS_CONFIG_HPP
#define CRITTERS_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace critters {

struct config {

    static constexpr char name[] = "critters";

    static constexpr double average_halo_radius = 1.0;

    using algorithm = critters_algorithm;
    
    using cell_state = critters_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::dead, cell_state::alive>;

    using pretty_print = critters_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = critters_random_init;
    };
};

}

#endif // CRITTERS_CONFIG_HPP
