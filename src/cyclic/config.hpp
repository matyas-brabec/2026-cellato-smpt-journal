#ifndef CYCLIC_CONFIG_HPP
#define CYCLIC_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace cyclic {

struct config {

    static constexpr char name[] = "cyclic";

    static constexpr double average_halo_radius = 1.0;

    using algorithm = cyclic_algorithm;
    
    using cell_state = cyclic_cell_state;
    using state_dictionary = cellato::memory::grids::int_based_state_dictionary<BITS>;

    using pretty_print = cyclic_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = cyclic_random_init;
    };
};

}

#endif // CYCLIC_CONFIG_HPP
