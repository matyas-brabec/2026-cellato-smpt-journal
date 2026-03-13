#ifndef FLUID_CONFIG_HPP
#define FLUID_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace fluid {

struct config {

    static constexpr char name[] = "fluid";

    static constexpr double average_halo_radius = 1.0;

    using algorithm = fluid_algorithm;
    
    using cell_state = fluid_cell_state;
    using state_dictionary = cellato::memory::grids::int_based_state_dictionary<4>;

    using pretty_print = fluid_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = fluid_random_init;
    };
};

}

#endif // FLUID_CONFIG_HPP
