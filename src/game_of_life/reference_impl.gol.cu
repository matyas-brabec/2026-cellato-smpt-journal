#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace game_of_life::reference {
using namespace ::reference::indexing;

// CUDA kernel for Game of Life (single step)
__global__ void gol_kernel(const gol_cell_state* current, gol_cell_state* next, 
                           int width, int height) {
    // Calculate thread indices, adjusting for margins
    int x = blockIdx.x * blockDim.x + threadIdx.x + indexer::x_margin;
    int y = blockIdx.y * blockDim.y + threadIdx.y + indexer::y_margin;

    indexer idx(width, height);
    const int center_idx = idx.at(x, y);
    
    // Count live neighbors using toroidal indexing
    int live_neighbors = 
        (current[idx.at(x - 1, y - 1)] == gol_cell_state::alive) + // Top-left
        (current[idx.at(x    , y - 1)] == gol_cell_state::alive) + // Top
        (current[idx.at(x + 1, y - 1)] == gol_cell_state::alive) + // Top-right
        (current[idx.at(x - 1, y    )] == gol_cell_state::alive) + // Left
        (current[idx.at(x + 1, y    )] == gol_cell_state::alive) + // Right
        (current[idx.at(x - 1, y + 1)] == gol_cell_state::alive) + // Bottom-left
        (current[idx.at(x    , y + 1)] == gol_cell_state::alive) + // Bottom
        (current[idx.at(x + 1, y + 1)] == gol_cell_state::alive);  // Bottom-right
    
    // Apply Game of Life rules
    gol_cell_state cell_state = current[center_idx];
    
    if (cell_state == gol_cell_state::alive) {
        // Live cell with fewer than 2 or more than 3 live neighbors dies
        if (live_neighbors < 2 || live_neighbors > 3) {
            next[center_idx] = gol_cell_state::dead;
        } else {
            // Live cell with 2 or 3 live neighbors stays alive
            next[center_idx] = gol_cell_state::alive;
        }
    } else {
        // Dead cell with exactly 3 live neighbors becomes alive
        if (live_neighbors == 3) {
            next[center_idx] = gol_cell_state::alive;
        } else {
            // Dead cell stays dead
            next[center_idx] = gol_cell_state::dead;
        }
    }
}

void runner::run_kernel(int steps) {
    if (!d_current || !d_next) {
        init_cuda();
    }
    
    // Set up grid and block dimensions, accounting for margins
    dim3 block_size(_block_size_x, _block_size_y);

    constexpr std::size_t x_margin = indexer::x_margin;
    constexpr std::size_t y_margin = indexer::y_margin;

    auto _x_size_threads = _x_size - 2 * x_margin; // Adjust for margins
    auto _y_size_threads = _y_size - 2 * y_margin; // Adjust for margins

    dim3 grid_dim((_x_size_threads + block_size.x - 1) / block_size.x, 
                 (_y_size_threads + block_size.y - 1) / block_size.y);
    
    // Run steps iterations
    for (int i = 0; i < steps; i++) {
        // Launch kernel for one step
        gol_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size);

        // Swap pointers for next iteration
        gol_cell_state* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
}

} // namespace game_of_life::reference
