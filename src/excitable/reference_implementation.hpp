#ifndef GREENBERG_REFERENCE_IMPLEMENTATION_HPP
#define GREENBERG_REFERENCE_IMPLEMENTATION_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include "./algorithm.hpp"
#include "experiments/run_params.hpp"
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace excitable::reference {
using namespace ::reference::indexing;

struct runner {
    static constexpr std::size_t x_margin = indexer::x_margin;
    static constexpr std::size_t y_margin = indexer::y_margin;

    void init(const ghm_cell_state* grid,
              const cellato::run::run_params& params = cellato::run::run_params()) {
        _x_size = params.x_size;
        _y_size = params.y_size;
        _block_size_x = params.cuda_block_size_x;
        _block_size_y = params.cuda_block_size_y;
        _current_grid.resize(_x_size * _y_size);
        _next_grid.resize(_x_size * _y_size);  // Pre-allocate next_grid

        if (params.device == "CUDA") {
            if ((_x_size - 2 * x_margin) % _block_size_x != 0 || (_y_size - 2 * y_margin) % _block_size_y != 0) {
                std::cerr << "Grid size must be divisible by block size.\n";
                throw std::runtime_error("Invalid grid size for CUDA traverser.");
            }
        }

        // Copy input grid
        if (grid) {
            for (std::size_t i = 0; i < _x_size * _y_size; ++i) {
                _current_grid[i] = grid[i];
                _next_grid[i] = grid[i];
            }
        }
    }

    void init_cuda() {
        const size_t grid_size = _x_size * _y_size * sizeof(ghm_cell_state);
        
        // Allocate device memory
        CUCH(cudaMalloc(&d_current, grid_size));
        CUCH(cudaMalloc(&d_next, grid_size));
        
        // Copy data to device
        CUCH(cudaMemcpy(d_current, _current_grid.data(), grid_size, cudaMemcpyHostToDevice));
    }

    void run(int steps) {
        indexer idx(_x_size, _y_size);
        
        for (int step = 0; step < steps; ++step) {
            // Process each cell, accounting for margins
            for (std::size_t y = y_margin; y < _y_size - y_margin; ++y) {
                for (std::size_t x = x_margin; x < _x_size - x_margin; ++x) {
                    const int center_idx = idx.at(x, y);
                    ghm_cell_state current = _current_grid[center_idx];
                    ghm_cell_state next = current;
                    
                    if (current == ghm_cell_state::quiescent) {
                        // Quiescent cell becomes excited if it has at least one excited neighbor
                        auto excited_count =
                            (_current_grid[idx.at(x-1, y-1)] == ghm_cell_state::excited) + // Top-left
                            (_current_grid[idx.at(x  , y-1)] == ghm_cell_state::excited) + // Top
                            (_current_grid[idx.at(x+1, y-1)] == ghm_cell_state::excited) + // Top-right
                            (_current_grid[idx.at(x-1, y  )] == ghm_cell_state::excited) + // Left
                            (_current_grid[idx.at(x+1, y  )] == ghm_cell_state::excited) + // Right
                            (_current_grid[idx.at(x-1, y+1)] == ghm_cell_state::excited) + // Bottom-left
                            (_current_grid[idx.at(x  , y+1)] == ghm_cell_state::excited) + // Bottom
                            (_current_grid[idx.at(x+1, y+1)] == ghm_cell_state::excited);  // Bottom-right

                        if (excited_count > 0) {
                            next = ghm_cell_state::excited;
                        }
                        else {
                            next = ghm_cell_state::quiescent; // Remains quiescent if no excited neighbors
                        }
                    }
                    else if (current == ghm_cell_state::excited) {
                        // Excited cell becomes refractory_1
                        next = ghm_cell_state::refractory_1;
                    }
                    else if (current == ghm_cell_state::refractory_1) {
                        // Refractory cells progress through refractory states
                        next = ghm_cell_state::refractory_2;
                    }
                    else if (current == ghm_cell_state::refractory_2) {
                        next = ghm_cell_state::refractory_3;
                    }
                    else if (current == ghm_cell_state::refractory_3) {
                        next = ghm_cell_state::refractory_4;
                    }
                    else if (current == ghm_cell_state::refractory_4) {
                        next = ghm_cell_state::refractory_5;
                    }
                    else if (current == ghm_cell_state::refractory_5) {
                        next = ghm_cell_state::refractory_6;
                    }
                    else if (current == ghm_cell_state::refractory_6) {
                        // Last refractory state returns to quiescent
                        next = ghm_cell_state::quiescent;
                    }
                    
                    _next_grid[center_idx] = next;
                }
            }
            
            // Swap grids
            _current_grid.swap(_next_grid);
        }
    }

    void run_on_cuda(int steps) {
        if (!d_current || !d_next) {
            init_cuda();
        }
        run_kernel(steps);
    }

    std::vector<ghm_cell_state> fetch_result() {
        if (d_current) {
            // Copy result back from device to host
            const size_t grid_size = _x_size * _y_size * sizeof(ghm_cell_state);
            CUCH(cudaMemcpy(_current_grid.data(), d_current, grid_size, cudaMemcpyDeviceToHost));
            
            // Free CUDA memory
            CUCH(cudaFree(d_current));
            CUCH(cudaFree(d_next));
            d_current = nullptr;
            d_next = nullptr;
        }
        return _current_grid;
    }

    ~runner() {
        if (d_current) {
            cudaFree(d_current);
            d_current = nullptr;
        }
        if (d_next) {
            cudaFree(d_next);
            d_next = nullptr;
        }
    }

private:
    std::size_t _x_size, _y_size;
    int _block_size_x = 16;
    int _block_size_y = 16;
    std::vector<ghm_cell_state> _current_grid;
    std::vector<ghm_cell_state> _next_grid;
    
    // Device pointers
    ghm_cell_state* d_current = nullptr;
    ghm_cell_state* d_next = nullptr;

    void run_kernel(int steps);
};

} // namespace excitable::reference

#endif // GREENBERG_REFERENCE_IMPLEMENTATION_HPP