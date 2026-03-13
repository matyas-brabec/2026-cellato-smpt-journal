#ifndef FIRE_REFERENCE_IMPLEMENTATION_HPP
#define FIRE_REFERENCE_IMPLEMENTATION_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include "./algorithm.hpp"
#include "experiments/run_params.hpp"
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace fire::reference {
using namespace ::reference::indexing;

struct runner {
    static constexpr std::size_t x_margin = indexer::x_margin;
    static constexpr std::size_t y_margin = indexer::y_margin;

    void init(const fire_cell_state* grid,
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
        const size_t grid_size = _x_size * _y_size * sizeof(fire_cell_state);
        
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
                    // Forest fire rules
                    const int center_idx = idx.at(x, y);
                    fire_cell_state current = _current_grid[center_idx];
                    fire_cell_state next = current;
                    
                    if (current == fire_cell_state::empty) {
                        // Empty remains empty
                        next = fire_cell_state::empty;
                    } 
                    else if (current == fire_cell_state::tree) {
                        // Tree catches fire if any von Neumann neighbor is on fire
                        // Use toroidal indexing for the 4 von Neumann neighbors
                        next = fire_cell_state::tree;
                        if (_current_grid[idx.at(x, y-1)] == fire_cell_state::fire ||  // North
                            _current_grid[idx.at(x+1, y)] == fire_cell_state::fire ||  // East
                            _current_grid[idx.at(x, y+1)] == fire_cell_state::fire ||  // South
                            _current_grid[idx.at(x-1, y)] == fire_cell_state::fire) {  // West
                            next = fire_cell_state::fire;
                        }
                    }
                    else if (current == fire_cell_state::fire) {
                        // Fire becomes ash
                        next = fire_cell_state::ash;
                    }
                    else if (current == fire_cell_state::ash) {
                        // Check if ash has fire neighbors using toroidal indexing
                        bool has_fire_neighbor = 
                            _current_grid[idx.at(x, y-1)] == fire_cell_state::fire ||  // North
                            _current_grid[idx.at(x+1, y)] == fire_cell_state::fire ||  // East
                            _current_grid[idx.at(x, y+1)] == fire_cell_state::fire ||  // South
                            _current_grid[idx.at(x-1, y)] == fire_cell_state::fire;    // West
                        
                        // Ash cell with fire neighbors remains ash, others become empty
                        next = has_fire_neighbor ? fire_cell_state::ash : fire_cell_state::empty;
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

    std::vector<fire_cell_state> fetch_result() {
        if (d_current) {
            // Copy result back from device to host
            const size_t grid_size = _x_size * _y_size * sizeof(fire_cell_state);
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
    std::vector<fire_cell_state> _current_grid;
    std::vector<fire_cell_state> _next_grid;
    
    // Device pointers
    fire_cell_state* d_current = nullptr;
    fire_cell_state* d_next = nullptr;

    void run_kernel(int steps);
};

} // namespace fire::reference

#endif // FIRE_REFERENCE_IMPLEMENTATION_HPP