#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace maze::reference {
using namespace ::reference::indexing;

__global__ void maze_kernel(const maze_cell_state* current, maze_cell_state* next, 
                            int width, int height) {
    // Calculate thread indices, adjusting for margins
    int x = blockIdx.x * blockDim.x + threadIdx.x + indexer::x_margin;
    int y = blockIdx.y * blockDim.y + threadIdx.y + indexer::y_margin;
    
    indexer idx(width, height);
    const int center_idx = idx.at(x, y);
    
    maze_cell_state cell_state = current[center_idx];
    
    // Count wall neighbors in Moore neighborhood (8 surrounding cells)
    int wall_count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue; // Skip the center cell
            if (current[idx.at(x + dx, y + dy)] == maze_cell_state::wall) {
                wall_count++;
            }
        }
    }
    
    // Apply maze algorithm rules
    maze_cell_state next_state;
    if (wall_count == 3) {
        next_state = maze_cell_state::wall;
    } else if (cell_state == maze_cell_state::wall && wall_count < 6) {
        next_state = maze_cell_state::wall;
    } else {
        next_state = maze_cell_state::empty;
    }

    next[center_idx] = next_state;
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
        maze_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size);
        
        // Swap pointers for next iteration
        maze_cell_state* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
}

} // namespace maze::reference
