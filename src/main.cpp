#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <string>

#include "experiments/run_params.hpp"
#include "experiments/test_suites.hpp"
#include "experiments/experiment_manager.hpp"
#include "experiments/reference_impl_manager.hpp"
#include "memory/grid_utils.hpp"

#include "game_of_life/algorithm.hpp"
#include "game_of_life/pretty_print.hpp"
#include "game_of_life/config.hpp"
#include "fire/config.hpp"
#include "excitable/config.hpp"
#include "wire/config.hpp"
#include "brian/config.hpp"
#include "maze/config.hpp"
#include "fluid/config.hpp"
#include "critters/config.hpp"
#include "cyclic/config.hpp"
#include "traffic/config.hpp"

#include "args-parser.hpp"


#define LOG std::cerr
#define REPORT std::cout

template <typename... all_test_suites>
struct switch_ {
    static void run(cellato::run::run_params& params) {
        // If reference implementation is requested, handle it separately
        if (params.reference_impl != "none") {
            bool ref_executed = run_reference_impl(params);
            if (!ref_executed) {
                std::cerr << "No suitable reference implementation found for the given parameters." << std::endl;
            }
            return;
        }
        
        bool any_executed = (call<all_test_suites>(params) || ...);
        if (!any_executed) {
            std::cerr << "No suitable test suite found for the given parameters." << std::endl;
        }
    }

private:
    static bool run_reference_impl(cellato::run::run_params& params) {

        if (params.reference_impl == "baseline") {
            if (params.automaton == "game-of-life") {
                return run_reference_for_automaton<game_of_life::config>(params);
            } else if (params.automaton == "fire" || params.automaton == "forest-fire") {
                return run_reference_for_automaton<fire::config>(params);
            } else if (params.automaton == "excitable") {
                return run_reference_for_automaton<excitable::config>(params);
            } else if (params.automaton == "wire") {
                return run_reference_for_automaton<wire::config>(params);
            } else if (params.automaton == "brian") {
                return run_reference_for_automaton<brian::config>(params);
            } else if (params.automaton == "maze") {
                return run_reference_for_automaton<maze::config>(params);
            } else if (params.automaton == "fluid") {
                return run_reference_for_automaton<fluid::config>(params);
            } else if (params.automaton == "critters") {
                return run_reference_for_automaton<critters::config>(params);
            } else if (params.automaton == "cyclic") {
                return run_reference_for_automaton<cyclic::config>(params);
            } else if (params.automaton == "traffic") {
                return run_reference_for_automaton<traffic::config>(params);
            }
        }

        return false;
    }

    template <typename automaton_config,
              typename runner_t = typename automaton_config::reference_implementation>
    static bool run_reference_for_automaton(cellato::run::run_params& params) {
        using cell_state_t = typename automaton_config::cell_state;
        
        // Generate initial state using the automaton's random initializer
        auto initial_state = automaton_config::input::random::init(params);
        
        // Run the reference implementation
        cellato::run::reference_impl_manager<runner_t, cell_state_t> manager;
        auto report = manager.run_experiment(params, initial_state);
        
        REPORT << report.csv_line() << std::endl;
        report.pretty_print(LOG);
        
        return true;
    }

    template <typename test_suite>
    static bool call(cellato::run::run_params& params) {
        if (!test_suite::is_for(params)) {
            return false;
        }

        using cellular_automaton = typename test_suite::automaton;

        auto initial_state = cellular_automaton::input::random::init(params);

        cellato::run::experiment_manager<test_suite> manager;
        manager.set_print_config(cellular_automaton::pretty_print::get_config());

        auto report = manager.run_experiment(
            params, initial_state
        );

        REPORT << report.csv_line() << std::endl;
        
        report.pretty_print(LOG);

        return true;
    }
};


template <typename test_suite>
void run(cellato::run::run_params& params) {

}

cellato::run::run_params get_params(int argc, char* argv[]) {
    input::parser parser {argc, argv};

    if (parser.exists("help")) {
        return cellato::run::run_params{.help = true};
    }

    if (parser.exists("print_csv_header")) {
        return cellato::run::run_params{.print_csv_header = true};
    }

    std::vector<std::string> required {
        "automaton",
        "device", "traverser", "evaluator", "layout",
        "x_size", "y_size", "steps",
    };

    std::vector<std::string> optional {
        "print", "word_size", "x_tile_size", "y_tile_size",
        "seed", "rounds", "warmup_rounds", "print_csv_header",
        "reference_impl", "cuda_block_size_x", "cuda_block_size_y",
        "temporal_steps", "temporal_tile_size_y"
    };

    if (parser.exists("evaluator")) {
        if (parser.get("evaluator") == "bit_planes" || parser.get("evaluator") == "bit_array") {
            required.push_back("word_size");
        }
    }

    if (parser.exists("traverser")) {
        if (parser.get("traverser") == "temporal") {
            required.push_back("temporal_steps");
            required.push_back("temporal_tile_size_y");
        }
    }

    if (parser.exists("reference_impl")) {
        for (const auto& no_longer_required : {
            "device", "traverser", "evaluator", "layout"
        }) {
            required.erase(
                std::remove(required.begin(), required.end(), no_longer_required),
                required.end()
            );
        }
    }

    for (const auto& opt : required) {
        if (!parser.exists(opt)) {
            std::cerr << "Missing required option: " << opt << std::endl;
            exit(1);
        }
    }

    cellato::run::run_params params {
        .automaton = parser.get("automaton"),

        .device = parser.get("device"),
        .traverser = parser.get("traverser"),
        .evaluator = parser.get("evaluator"),
        .layout = parser.get("layout"),

        .reference_impl = parser.exists("reference_impl") ? parser.get("reference_impl") : "none",

        .x_size = std::stoi(parser.get("x_size")),
        .y_size = std::stoi(parser.get("y_size")),
        .steps = std::stoi(parser.get("steps")),

        .word_size = parser.exists("word_size") ? std::stoi(parser.get("word_size")) : 0,
        
        .x_tile_size = parser.exists("x_tile_size") ? std::stoi(parser.get("x_tile_size")) : 0,
        .y_tile_size = parser.exists("y_tile_size") ? std::stoi(parser.get("y_tile_size")) : 0,

        .temporal_steps = parser.exists("temporal_steps") ? std::stoi(parser.get("temporal_steps")) : 0,
        .temporal_tile_size_y = parser.exists("temporal_tile_size_y") ? std::stoi(parser.get("temporal_tile_size_y")) : 0,
        
        .rounds = parser.exists("rounds") ? std::stoi(parser.get("rounds")) : 1,
        .warmup_rounds = parser.exists("warmup_rounds") ? std::stoi(parser.get("warmup_rounds")) : 0,

        .seed = parser.exists("seed") ? std::stoi(parser.get("seed")) : 42,

        .print = parser.exists("print"),
        .help = parser.exists("help"),
        .print_csv_header = parser.exists("print_csv_header"),

        .cuda_block_size_x = parser.exists("cuda_block_size_x") ? std::stoi(parser.get("cuda_block_size_x")) : 32,
        .cuda_block_size_y = parser.exists("cuda_block_size_y") ? std::stoi(parser.get("cuda_block_size_y")) : 8
    };

    return params;
}

void print_usage() {
    std::cout << "Usage: ./cellato [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --automaton <name>              Name of the cellular automaton\n";
    std::cout << "  --device <name>                 Device to run on (CPU, CUDA)\n";
    std::cout << "  --traverser <name>              Traverser type (simple, spacial_blocking)\n";
    std::cout << "  --evaluator <name>              Evaluator type (standard, bit_planes)\n";
    std::cout << "  --layout <name>                 Layout type (standard, bit_array, bit_planes)\n";
    std::cout << "  --reference_impl <name>         Reference implementation to use (baseline)\n";
    std::cout << "  --x_size <number>               X size of the grid\n";
    std::cout << "  --y_size <number>               Y size of the grid\n";
    std::cout << "  --x_tile_size <number>          X tile size for CUDA\n";
    std::cout << "  --y_tile_size <number>          Y tile size for CUDA\n";
    std::cout << "  --temporal_steps <number>       Temporal steps for CUDA (only for temporal_tiled_bit_planes)\n";
    std::cout << "  --temporal_tile_size_y <number> Temporal tile size Y for CUDA (only for temporal_tiled_bit_planes)\n";
    std::cout << "  --rounds <number>               Number of rounds to run\n";
    std::cout << "  --warmup_rounds <number>        Number of warmup rounds to run\n";
    std::cout << "  --steps <number>                Number of steps to run\n";
    std::cout << "  --word_size <number>            word_size for floating-point calculations (32, 64)\n";
    std::cout << "  --seed <number>                 Random seed for initialization\n";
    std::cout << "  --print                         Print the grid after each step\n";
    std::cout << "  --reference_impl                Use reference implementation for the automaton\n";
    std::cout << "  --cuda_block_size_x <number>    CUDA block size X (default: 16)\n";
    std::cout << "  --cuda_block_size_y <number>    CUDA block size Y (default: 16)\n";
    std::cout << "  --print_csv_header              Print CSV header\n";
    std::cout << "  --help                          Show this help message\n";
}


int main(int argc, char* argv[]) {
    
    auto params = get_params(argc, argv);

    if (params.help) {
        print_usage();
        return 0;
    }

    if (params.print_csv_header) {
        std::cout << cellato::run::experiment_report::csv_header() << std::endl;
        return 0;
    }

    if (params.print) {
        params.print_std();
    }

    namespace test = cellato::run::test_suites;

    using _game_of_life_ = game_of_life::config;
    using _fire_ = fire::config;
    using _wire_ = wire::config;
    using _excitable_ = excitable::config;
    using _brian_ = brian::config;
    using _maze_ = maze::config;
    using _fluid_ = fluid::config;
    using _critters_ = critters::config;
    using _cyclic_ = cyclic::config;
    using _traffic_ = traffic::config;

    #define cases_for(automaton) \
        test::on_cpu::standard<automaton>, \
        test::on_cpu::using_<std::uint32_t>::bit_array<automaton>, \
        test::on_cpu::using_<std::uint64_t>::bit_array<automaton>, \
        test::on_cpu::using_<std::uint32_t>::bit_planes<automaton>, \
        test::on_cpu::using_<std::uint64_t>::bit_planes<automaton>, \
        test::on_cpu::using_<std::uint32_t>::tiled_bit_planes<automaton>, \
        test::on_cpu::using_<std::uint64_t>::tiled_bit_planes<automaton>, \
        test::on_cuda::using_<std::uint32_t>::tiled_bit_planes<automaton>, \
        test::on_cuda::using_<std::uint64_t>::tiled_bit_planes<automaton>, \
        test::on_cuda::standard<automaton>, \
        test::on_cuda::standard<automaton>::with_spacial_blocking<1, 1>, \
        test::on_cuda::standard<automaton>::with_spacial_blocking<2, 1>, \
        test::on_cuda::standard<automaton>::with_spacial_blocking<4, 1>, \
        test::on_cuda::using_<std::uint32_t>::bit_array<automaton>, \
        test::on_cuda::using_<std::uint64_t>::bit_array<automaton>, \
        test::on_cuda::using_<std::uint32_t>::bit_planes<automaton>, \
        test::on_cuda::using_<std::uint64_t>::bit_planes<automaton>, \
        test::on_cuda::using_<std::uint32_t>::temporal_tiled_bit_planes<automaton>, \
        test::on_cuda::using_<std::uint64_t>::temporal_tiled_bit_planes<automaton>, \
        test::on_cuda::using_<std::uint32_t>::temporal_linear_bit_planes<automaton>, \
        test::on_cuda::using_<std::uint64_t>::temporal_linear_bit_planes<automaton>

    switch_<
        cases_for(_game_of_life_),
        cases_for(_fire_),
        cases_for(_wire_),
        cases_for(_excitable_),
        cases_for(_brian_),
        cases_for(_maze_),
        cases_for(_fluid_),
        cases_for(_critters_),
        cases_for(_cyclic_),
        cases_for(_traffic_)
    >::run(params);

    return 0;
}
