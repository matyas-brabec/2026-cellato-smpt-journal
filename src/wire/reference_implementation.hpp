#ifndef WIRE_REFERENCE_IMPLEMENTATION_HPP
#define WIRE_REFERENCE_IMPLEMENTATION_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include "./algorithm.hpp"
#include "experiments/run_params.hpp"
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace wire::reference {
using namespace ::reference::indexing;

struct runner {
    static constexpr std::size_t x_margin = indexer::x_margin;
    static constexpr std::size_t y_margin = indexer::y_margin;
    
    void init(const wire_cell_state* grid, 
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
        const size_t grid_size = _x_size * _y_size * sizeof(wire_cell_state);
        
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
                    // WireWorld rules
                    const int center_idx = idx.at(x, y);
                    wire_cell_state current = _current_grid[center_idx];
                    wire_cell_state next = current;
                    
                    if (current == wire_cell_state::empty) {
                        // Empty remains empty
                        next = wire_cell_state::empty;
                    }
                    else if (current == wire_cell_state::electron_head) {
                        // Electron head becomes electron tail
                        next = wire_cell_state::electron_tail;
                    }
                    else if (current == wire_cell_state::electron_tail) {
                        // Electron tail becomes conductor
                        next = wire_cell_state::conductor;
                    }
                    else if (current == wire_cell_state::conductor) {
                        // Count electron heads in the Moore neighborhood using toroidal indexing
                        int electron_head_count = 
                            (_current_grid[idx.at(x-1, y-1)] == wire_cell_state::electron_head) + // Top-left
                            (_current_grid[idx.at(x  , y-1)] == wire_cell_state::electron_head) + // Top
                            (_current_grid[idx.at(x+1, y-1)] == wire_cell_state::electron_head) + // Top-right
                            (_current_grid[idx.at(x-1, y  )] == wire_cell_state::electron_head) + // Left
                            (_current_grid[idx.at(x+1, y  )] == wire_cell_state::electron_head) + // Right
                            (_current_grid[idx.at(x-1, y+1)] == wire_cell_state::electron_head) + // Bottom-left
                            (_current_grid[idx.at(x  , y+1)] == wire_cell_state::electron_head) + // Bottom
                            (_current_grid[idx.at(x+1, y+1)] == wire_cell_state::electron_head);  // Bottom-right
                        
                        // Conductor becomes electron head if exactly 1 or 2 neighboring cells are electron heads
                        if (electron_head_count == 1 || electron_head_count == 2) {
                            next = wire_cell_state::electron_head;
                        } else {
                            next = wire_cell_state::conductor;
                        }
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

    std::vector<wire_cell_state> fetch_result() {
        if (d_current) {
            // Copy result back from device to host
            const size_t grid_size = _x_size * _y_size * sizeof(wire_cell_state);
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
    std::vector<wire_cell_state> _current_grid;
    std::vector<wire_cell_state> _next_grid;
    
    // Device pointers
    wire_cell_state* d_current = nullptr;
    wire_cell_state* d_next = nullptr;

    void run_kernel(int steps);
};

} // namespace wire::reference

#endif // WIRE_REFERENCE_IMPLEMENTATION_HPP