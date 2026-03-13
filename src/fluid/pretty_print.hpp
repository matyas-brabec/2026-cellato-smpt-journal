#ifndef FLUID_PRETTY_PRINT_HPP
#define FLUID_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"

namespace fluid {

using print_config = cellato::memory::grids::standard::print_config<fluid_cell_state>;

struct fluid_pretty_print {
    static print_config get_config() {
        return print_config()
            // no bits set
            .with(0, "\033[90m \033[0m")
            // one bit set
            .with(1, "\033[1;32m·\033[0m")
            .with(2, "\033[1;34m·\033[0m")
            .with(4, "\033[1;33m·\033[0m")
            .with(8, "\033[1;31m·\033[0m")
            // two bits set
            .with(3, "\033[1;36m○\033[0m")
            .with(5, "\033[1;35m○\033[0m")
            .with(9, "\033[1;31m○\033[0m")
            .with(6, "\033[1;37m○\033[0m")
            .with(10, "\033[1;33m○\033[0m")
            .with(12, "\033[1;30m○\033[0m")
            // three bits set
            .with(7, "\033[1;37m◉\033[0m")
            .with(11, "\033[1;33m◉\033[0m")
            .with(13, "\033[1;31m◉\033[0m")
            .with(14, "\033[1;34m◉\033[0m")
            // all bits set
            .with(15, "\033[1;30m█\033[0m");
    }
};

}

#endif // FLUID_PRETTY_PRINT_HPP
