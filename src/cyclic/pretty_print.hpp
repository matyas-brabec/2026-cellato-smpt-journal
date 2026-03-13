#ifndef CYCLIC_PRETTY_PRINT_HPP
#define CYCLIC_PRETTY_PRINT_HPP

#include "memory/standard_grid.hpp"
#include "./algorithm.hpp"
#include <cstring>
#include <string>
#include <vector>

namespace cyclic {

using print_config = cellato::memory::grids::standard::print_config<cyclic_cell_state>;

struct cyclic_pretty_print {
    static print_config get_config() {
        auto conf = print_config();

        auto bash_color_list = std::vector<std::string>{
            "\033[0;30m", // Black
            "\033[0;34m", // Blue
            "\033[0;32m", // Green
            "\033[0;36m", // Cyan
            "\033[0;31m", // Red
            "\033[0;35m", // Purple
            "\033[0;33m", // Yellow
            "\033[0;37m", // White
            "\033[1;30m", // Bright Black (Gray)
            "\033[1;34m", // Bright Blue
            "\033[1;32m", // Bright Green
            "\033[1;36m", // Bright Cyan
            "\033[1;31m", // Bright Red
            "\033[1;35m", // Bright Purple
            "\033[1;33m", // Bright Yellow
            "\033[1;37m", // Bright White
            "\033[0;90m", // Dark Gray
            "\033[0;94m", // Light Blue
            "\033[0;92m", // Light Green
            "\033[0;96m", // Light Cyan
            "\033[0;91m", // Light Red
            "\033[0;95m", // Light Purple
            "\033[0;93m", // Light Yellow
            "\033[0;97m", // Light White
            "\033[1;90m", // Very Dark Gray
            "\033[1;94m", // Very Light Blue
            "\033[1;92m", // Very Light Green
            "\033[1;96m", // Very Light Cyan
            "\033[1;91m", // Very Light Red
            "\033[1;95m", // Very Light Purple
            "\033[1;93m", // Very Light Yellow
            "\033[1;97m"  // Very Light White
        };

        auto symbols = "abcdefghijklmnopqrstuvwxyz012345.,;:!?'\"`~^*-+=<>[]{}()|\\/";

        for (std::size_t i = 0; i < STATES; ++i) {
            auto symbol = std::string(1, symbols[i % strlen(symbols)]);
            conf = conf.with(static_cast<cyclic_cell_state>(i), bash_color_list[i % bash_color_list.size()] + symbol + "\033[0m");
        }

        return conf;
    }
};

}

#endif // CYCLIC_PRETTY_PRINT_HPP
