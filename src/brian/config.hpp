#ifndef BRIAN_CONFIG_HPP
#define BRIAN_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace brian {

struct config {

    static constexpr char name[] = "brian";

    static constexpr double average_halo_radius = 1.0;

    using algorithm = brian_algorithm;

    using cell_state = brian_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::dead, cell_state::dying, cell_state::alive>;

    using pretty_print = brian_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = brian_random_init;
    };
};

} // namespace brian

#endif // BRIAN_CONFIG_HPP