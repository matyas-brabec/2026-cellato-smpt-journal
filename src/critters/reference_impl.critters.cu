#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace critters::reference {
using namespace ::reference::indexing;

// CUDA kernel for Forest critters (single step)
__global__ void critters_kernel(const critters_cell_state* current, critters_cell_state* next, 
                                int width, int height, int step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + indexer::x_margin;
    int y = blockIdx.y * blockDim.y + threadIdx.y + indexer::y_margin;
    
    indexer idx(width, height);
    const int center_idx = idx.at(x, y);
    
    critters_cell_state own_state = current[center_idx];

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
            live_cells_in_block += (current[neighbor_idx] == critters_cell_state::alive) ? 1 : 0;
        }
    }
    
    critters_cell_state next_state;

    if (live_cells_in_block == 3) {
        // Rule: Flip and rotate 180 degrees.
        // The new state is the FLIPPED state of the DIAGONAL neighbor.

        int dx_opposite, dy_opposite;

        if (step_parity == 0) {
            if (x_parity == 0) { dx_opposite = 1; } 
            else { dx_opposite = -1; }
            
            if (y_parity == 0) { dy_opposite = 1; } 
            else { dy_opposite = -1; }
        } else { // step_parity == 1
            if (x_parity == 0) { dx_opposite = -1; } 
            else { dx_opposite = 1; }
            
            if (y_parity == 0) { dy_opposite = -1; } 
            else { dy_opposite = 1; }
        }
        
        int opposite_idx = idx.at(x + dx_opposite, y + dy_opposite);
        critters_cell_state opposite_state = current[opposite_idx];

        next_state = (opposite_state == critters_cell_state::alive) 
                     ? critters_cell_state::dead 
                     : critters_cell_state::alive;

    } else if (live_cells_in_block == 2) {
        // Rule: No change. The block remains the same.
        next_state = own_state;
    } else {
        // Rule (covers counts 0, 1, and 4): Flip the state in place.
        next_state = (own_state == critters_cell_state::alive) 
                     ? critters_cell_state::dead 
                     : critters_cell_state::alive;
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
        critters_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size, i);
        
        // Swap pointers for next iteration
        critters_cell_state* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
}

} // namespace critters::reference
