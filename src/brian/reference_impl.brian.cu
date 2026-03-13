#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace brian::reference {
using namespace ::reference::indexing;

// CUDA kernel for Brian's Brain (single step)
__global__ void brian_kernel(const brian_cell_state* current, brian_cell_state* next, 
                             int width, int height) {
    // Calculate thread indices, adjusting for margins
    int x = blockIdx.x * blockDim.x + threadIdx.x + indexer::x_margin;
    int y = blockIdx.y * blockDim.y + threadIdx.y + indexer::y_margin;
        
    indexer idx(width, height);
    const int center_idx = idx.at(x, y);
    
    // Count alive neighbors using toroidal indexing (Moore neighborhood)
    int alive_neighbors = 
        (current[idx.at(x - 1, y - 1)] == brian_cell_state::alive) + // Top-left
        (current[idx.at(x    , y - 1)] == brian_cell_state::alive) + // Top
        (current[idx.at(x + 1, y - 1)] == brian_cell_state::alive) + // Top-right
        (current[idx.at(x - 1, y    )] == brian_cell_state::alive) + // Left
        (current[idx.at(x + 1, y    )] == brian_cell_state::alive) + // Right
        (current[idx.at(x - 1, y + 1)] == brian_cell_state::alive) + // Bottom-left
        (current[idx.at(x    , y + 1)] == brian_cell_state::alive) + // Bottom
        (current[idx.at(x + 1, y + 1)] == brian_cell_state::alive);  // Bottom-right
    
    // Apply Brian's Brain rules
    brian_cell_state cell_state = current[center_idx];
    
    if (cell_state == brian_cell_state::dead) {
        // Dead cell with exactly 2 alive neighbors becomes alive
        if (alive_neighbors == 2) {
            next[center_idx] = brian_cell_state::alive;
        } else {
            next[center_idx] = brian_cell_state::dead;
        }
    } else if (cell_state == brian_cell_state::alive) {
        // Alive cell always becomes dying
        next[center_idx] = brian_cell_state::dying;
    } else { // cell_state == brian_cell_state::dying
        // Dying cell always becomes dead
        next[center_idx] = brian_cell_state::dead;
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
        brian_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size);

        // Swap pointers for next iteration
        brian_cell_state* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
}

} // namespace game_of_life::reference
