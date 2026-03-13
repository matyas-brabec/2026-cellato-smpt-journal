#ifndef CELLATO_TRAVERSERS_CUDA_LINEAR_TEMPORAL_HPP
#define CELLATO_TRAVERSERS_CUDA_LINEAR_TEMPORAL_HPP

#include <cstddef>
#include <iostream>
#include <utility>
#include <functional>
#include <cuda_runtime.h>
#include <memory>

#include "../../memory/interface.hpp"
#include "../../experiments/run_params.hpp"
#include "../traverser_utils.hpp"

namespace cellato::traversers::cuda::temporal {

using namespace cellato::traversers::utils;

enum class _run_mode {
    QUIET,
    VERBOSE,
};

template <
    typename evaluator_type,
    typename grid_type,
    double average_halo_radius>
class traverser {
    using evaluator_t = evaluator_type;
    using grid_t = grid_type;
    using cuda_grid_t = typename std::invoke_result<decltype(&grid_t::to_cuda), grid_t>::type;
    using cell_t = typename grid_t::store_type;

    constexpr static std::size_t word_tile_x = grid_t::x_word_tile_size;
    constexpr static std::size_t word_tile_y = grid_t::y_word_tile_size;

  public:
    static constexpr bool is_CUDA = true;

    traverser() : _final_grid(nullptr) {}

    void init(grid_t grid, 
              const cellato::run::run_params& params) {

        _block_size_x = params.cuda_block_size_x;
        _block_size_y = params.cuda_block_size_y;
        
        _temporal_tile_size_y = params.temporal_tile_size_y;
        _temporal_steps = params.temporal_steps;

        std::size_t needed_halo_cells = static_cast<std::size_t>(std::ceil(average_halo_radius * _temporal_steps));
        std::size_t x_halo_words = (needed_halo_cells + word_tile_x - 1) / word_tile_x;
        std::size_t y_halo_words = (needed_halo_cells + word_tile_y - 1) / word_tile_y;
    
        _effective_temporal_tile_size_x = _block_size_x - 2 * x_halo_words;
        _effective_temporal_tile_size_y = _temporal_tile_size_y - 2 * y_halo_words;

        _cells_per_thread = _temporal_tile_size_y / _block_size_y;

        if (grid.x_size_physical() % _effective_temporal_tile_size_x != 0 ||
            grid.y_size_physical() % _effective_temporal_tile_size_y != 0) {
            std::cerr << "Grid size must be divisible by effective temporal tile size. (effective temporal tile size: "
                      << _effective_temporal_tile_size_x << "x" << _effective_temporal_tile_size_y << ")\n";
            std::cerr << "Grid size: " << grid.x_size_physical() << "x" << grid.y_size_physical() << "\n";
            throw std::runtime_error("Invalid grid size for CUDA traverser.");
        }

        if (_effective_temporal_tile_size_y <= 0) {
            std::cerr << "Error: effective_y_tile_size must be greater than 0.\n";
            throw std::runtime_error("Invalid temporal tile size for temporal linear traverser.");
        }

        if (params.steps % _temporal_steps != 0) {
            std::cerr << "Total steps must be divisible by temporal steps.\n";
            throw std::runtime_error("Invalid steps for temporal linear traverser.");
        }

        if (_block_size_x != 32) {
            std::cerr << "Error: Using non-standard block size for CUDA (recommended: 32).\n";
            throw std::runtime_error("Invalid block size for CUDA traverser.");
        }

        if (_temporal_tile_size_y % _block_size_y != 0) {
            std::cerr << "Temporal tile size Y must be divisible by block size Y.\n";
            throw std::runtime_error("Invalid temporal tile size for temporal linear traverser.");
        }

        _input_grid = std::move(grid);
        
        _input_grid_cuda = _input_grid.to_cuda();
        _intermediate_grid_cuda = _input_grid.to_cuda();
        
        _final_grid = &_input_grid_cuda;
    }

    struct no_callback {};

    template <typename callback = no_callback>
    void run(int steps, callback&& callback_func = no_callback{}) {
        
        if constexpr (!std::is_same_v<callback, no_callback>) {
            _callback_func = std::make_unique<_lambda_wrapper<callback>>(std::forward<callback>(callback_func));
            run_kernel<_run_mode::VERBOSE>(steps);

        } else {
            run_kernel<_run_mode::QUIET>(steps);
        }
    }
    
    grid_t fetch_result();
    
private:
    template <_run_mode mode>
    void run_kernel(int steps);
    
    grid_t _input_grid;
    cuda_grid_t _input_grid_cuda;
    cuda_grid_t _intermediate_grid_cuda;
    cuda_grid_t* _final_grid;

    int _block_size_x = -1;
    int _block_size_y = -1;

    int _effective_temporal_tile_size_x = -1;
    int _effective_temporal_tile_size_y = -1;
    
    int _temporal_steps = -1;
    int _temporal_tile_size_y = -1;
    int _cells_per_thread = -1;

    struct _call_back_obj {
        virtual void call(int iteration, grid_t& grid) = 0;
    };

    template <typename func>
    struct _lambda_wrapper : public _call_back_obj {
        func f;
        _lambda_wrapper(func f) : f(f) {}
        void call(int iteration, grid_t& grid) override { f(iteration, grid); }
    };

    std::unique_ptr<_call_back_obj> _callback_func;

    void call_callback(int iteration, cuda_grid_t* grid) {
        if (_callback_func) {
            auto on_host_grid = grid->to_cpu();
            _callback_func->call(iteration, on_host_grid);
        }
    }
};

} // namespace cellato::traversers::cuda::temporal

#endif // CELLATO_TRAVERSERS_CUDA_LINEAR_TEMPORAL_HPP
