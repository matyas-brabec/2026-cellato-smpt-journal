#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include <memory>
#include "traversers/cuda_utils.cuh"
#include "../_shared/indexing.hpp"

namespace fluid::reference {
using namespace ::reference::indexing;

namespace {

// CUDA kernel for Forest fluid (single step)
__global__ void fluid_kernel(const fluid_cell_state* current, fluid_cell_state* next, 
                            int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const indexer idx(width, height);
    const int center_idx = idx.at(x, y);

    const auto top_neighbor = current[idx.at(x, y-1)];
    const auto bottom_neighbor = current[idx.at(x, y+1)];
    const auto left_neighbor = current[idx.at(x-1, y)];
    const auto right_neighbor = current[idx.at(x+1, y)];

    constexpr fluid_cell_state TOP = 0b0001;
    constexpr fluid_cell_state BOTTOM = 0b0010;
    constexpr fluid_cell_state LEFT = 0b0100;
    constexpr fluid_cell_state RIGHT = 0b1000;

    const auto incoming_from_top = (top_neighbor & TOP);
    const auto incoming_from_bottom = (bottom_neighbor & BOTTOM);
    const auto incoming_from_left = (left_neighbor & LEFT);
    const auto incoming_from_right = (right_neighbor & RIGHT);

    const auto vertical_collision = (incoming_from_top != 0) & (incoming_from_bottom != 0);
    const auto horizontal_collision = (incoming_from_left != 0) & (incoming_from_right != 0);

    const auto combined_vertical_incoming = incoming_from_top | incoming_from_bottom;
    const auto combined_horizontal_incoming = incoming_from_left | incoming_from_right;

    fluid_cell_state result = 0;

    const auto just_vertical_collision = vertical_collision && (combined_horizontal_incoming == 0);
    const auto just_horizontal_collision = horizontal_collision && (combined_vertical_incoming == 0);

    if (just_vertical_collision) {
        result |= (LEFT | RIGHT); // horizontal outgoing
    } else {
        result |= combined_vertical_incoming; // pass vertical incoming
    }

    if (just_horizontal_collision) {
        result |= (TOP | BOTTOM); // vertical outgoing
    } else {
        result |= combined_horizontal_incoming; // pass horizontal incoming
    }

    next[center_idx] = result;
}

}

void runner::run_kernel(int steps) {
    if (!d_current || !d_next) {
        init_cuda();
    }
    
    // Set up grid and block dimensions, accounting for margins
    const dim3 block_size(_block_size_x, _block_size_y);

    // Toroidal wrapping - same width and height
    const auto _x_size_threads = _x_size;
    const auto _y_size_threads = _y_size;

    const dim3 grid_dim((_x_size_threads + block_size.x - 1) / block_size.x, 
                 (_y_size_threads + block_size.y - 1) / block_size.y);

    // Run steps iterations
    for (int i = 0; i < steps; i++) {
        // Launch kernel for one step
        fluid_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size);
        
        // Swap pointers for next iteration
        std::swap(d_current, d_next);

        CUCH(cudaGetLastError());
    }
}

} // namespace fluid::reference
