#ifndef TRAFFIC_CONFIG_HPP
#define TRAFFIC_CONFIG_HPP

#include "./algorithm.hpp"
#include "./pretty_print.hpp"
#include "./data_init.hpp"
#include "./reference_implementation.hpp"
#include "memory/state_dictionary.hpp"

namespace traffic {

struct config {

    static constexpr char name[] = "traffic";

    static constexpr double average_halo_radius = 1.0;

    using algorithm = traffic_algorithm;
    
    using cell_state = traffic_cell_state;
    using state_dictionary = cellato::memory::grids::state_dictionary<
        cell_state::empty, cell_state::red_car, cell_state::blue_car>;

    using pretty_print = traffic_pretty_print;

    using reference_implementation = reference::runner;

    struct input {
        using random = traffic_random_init;
    };
};

}

#endif // TRAFFIC_CONFIG_HPP
