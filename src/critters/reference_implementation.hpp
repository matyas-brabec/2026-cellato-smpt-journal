#ifndef CRITTERS_REFERENCE_IMPLEMENTATION_HPP
#define CRITTERS_REFERENCE_IMPLEMENTATION_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include "./algorithm.hpp"
#include "experiments/run_params.hpp"
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace critters::reference {
using namespace ::reference::indexing;

struct runner {
    static constexpr std::size_t x_margin = indexer::x_margin;
    static constexpr std::size_t y_margin = indexer::y_margin;

    void init(const critters_cell_state* grid,
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
        const size_t grid_size = _x_size * _y_size * sizeof(critters_cell_state);
        
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
                    // Forest critters rules
                    const int center_idx = idx.at(x, y);
                    critters_cell_state current = _current_grid[center_idx];
                    
                    int x_parity = x % 2;
                    int y_parity = y % 2;
                    int step_parity = step % 2;
                    
                    int x_coords[2];
                    int y_coords[2];
                    
                    if (step_parity == 0) {
                        if (x_parity == 0) { x_coords[0] = 0; x_coords[1] = 1; } 
                        else { x_coords[0] = -1; x_coords[1] = 0; }
                        
                        if (y_parity == 0) { y_coords[0] = 0; y_coords[1] = 1; } 
                        else { y_coords[0] = -1; y_coords[1] = 0; }
                    } else { // step_parity == 1
                        if (x_parity == 0) { x_coords[0] = -1; x_coords[1] = 0; } 
                        else { x_coords[0] = 0; x_coords[1] = 1; }
                        
                        if (y_parity == 0) { y_coords[0] = -1; y_coords[1] = 0; } 
                        else { y_coords[0] = 0; y_coords[1] = 1; }
                    }
                    
                    // Count the total number of live cells in the 2x2 block
                    int live_cells_in_block = 0;
                    for (int dx_idx = 0; dx_idx < 2; dx_idx++) {
                        int dx = x_coords[dx_idx];
                        for (int dy_idx = 0; dy_idx < 2; dy_idx++) {
                            int dy = y_coords[dy_idx];
                            int neighbor_idx = idx.at(x + dx, y + dy);
                            live_cells_in_block += (_current_grid[neighbor_idx] == critters_cell_state::alive) ? 1 : 0;
                        }
                    }
                    
                    critters_cell_state next_state;

                    if (live_cells_in_block == 2) {
                        // Rule 1: No change. The block remains the same.
                        next_state = current;
                    } else if (live_cells_in_block == 3) {
                        // Rule 3: Flip and rotate 180 degrees.
                        // The new state is the FLIPPED state of the DIAGONAL neighbor.

                        int dx_opposite = x_coords[0] + x_coords[1];
                        int dy_opposite = y_coords[0] + y_coords[1];
                        
                        int opposite_idx = idx.at(x + dx_opposite, y + dy_opposite);
                        critters_cell_state opposite_state = _current_grid[opposite_idx];

                        next_state = (opposite_state == critters_cell_state::alive) 
                                    ? critters_cell_state::dead 
                                    : critters_cell_state::alive;
                    } else {
                        // Rule 2 (covers counts 0, 1, and 4): Flip the state in place.
                        next_state = (current == critters_cell_state::alive) 
                                    ? critters_cell_state::dead 
                                    : critters_cell_state::alive;
                    }

                    _next_grid[center_idx] = next_state;
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

    std::vector<critters_cell_state> fetch_result() {
        if (d_current) {
            // Copy result back from device to host
            const size_t grid_size = _x_size * _y_size * sizeof(critters_cell_state);
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
    std::vector<critters_cell_state> _current_grid;
    std::vector<critters_cell_state> _next_grid;
    
    // Device pointers
    critters_cell_state* d_current = nullptr;
    critters_cell_state* d_next = nullptr;

    void run_kernel(int steps);
};

} // namespace critters::reference

#endif // CRITTERS_REFERENCE_IMPLEMENTATION_HPP