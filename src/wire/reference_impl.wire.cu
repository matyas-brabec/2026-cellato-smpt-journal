#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace wire::reference {
using namespace ::reference::indexing;

// CUDA kernel for WireWorld (single step)
__global__ void wire_kernel(const wire_cell_state* current, wire_cell_state* next, 
                           int width, int height) {
    // Calculate thread indices, adjusting for margins
    int x = blockIdx.x * blockDim.x + threadIdx.x + indexer::x_margin;
    int y = blockIdx.y * blockDim.y + threadIdx.y + indexer::y_margin;
    
    indexer idx(width, height);
    const int center_idx = idx.at(x, y);
    
    wire_cell_state cell_state = current[center_idx];
    wire_cell_state next_state = cell_state;
    
    if (cell_state == wire_cell_state::empty) {
        // Empty remains empty
        next_state = wire_cell_state::empty;
    }
    else if (cell_state == wire_cell_state::electron_head) {
        // Electron head becomes electron tail
        next_state = wire_cell_state::electron_tail;
    }
    else if (cell_state == wire_cell_state::electron_tail) {
        // Electron tail becomes conductor
        next_state = wire_cell_state::conductor;
    }
    else if (cell_state == wire_cell_state::conductor) {
        // Count electron heads in the Moore neighborhood using toroidal indexing
        int electron_head_count = 
            (current[idx.at(x-1, y-1)] == wire_cell_state::electron_head) + // Top-left
            (current[idx.at(x  , y-1)] == wire_cell_state::electron_head) + // Top
            (current[idx.at(x+1, y-1)] == wire_cell_state::electron_head) + // Top-right
            (current[idx.at(x-1, y  )] == wire_cell_state::electron_head) + // Left
            (current[idx.at(x+1, y  )] == wire_cell_state::electron_head) + // Right
            (current[idx.at(x-1, y+1)] == wire_cell_state::electron_head) + // Bottom-left
            (current[idx.at(x  , y+1)] == wire_cell_state::electron_head) + // Bottom
            (current[idx.at(x+1, y+1)] == wire_cell_state::electron_head);  // Bottom-right
        
        // Conductor becomes electron head if exactly 1 or 2 neighboring cells are electron heads
        if (electron_head_count == 1 || electron_head_count == 2) {
            next_state = wire_cell_state::electron_head;
        } else {
            next_state = wire_cell_state::conductor;
        }
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
        wire_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size);
        
        // Swap pointers for next iteration
        wire_cell_state* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
}

} // namespace wire::reference
