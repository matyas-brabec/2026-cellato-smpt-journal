#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

#include "./simple.hpp"
#include "../../memory/standard_grid.hpp"
#include "../../memory/interface.hpp"
#include "../../memory/idx_type.hpp"
#include "../../evaluators/standard.hpp"
#include "../../core/ast.hpp"
#include "../traverser_utils.hpp"
#include "../cuda_utils.cuh"

namespace cellato::traversers::cuda::simple {

using idx_type = cellato::memory::idx_type;

namespace {

template <typename evaluator_t, typename grid_data_t, typename output_data_t>
__global__ void process_grid_kernel_simple(
    grid_data_t input_data,
    output_data_t output_data,
    idx_type width,
    idx_type height,
    idx_type time_step
) {
    const idx_type x = blockIdx.x * blockDim.x + threadIdx.x;
    const idx_type y = blockIdx.y * blockDim.y + threadIdx.y;

    const cellato::memory::grids::point_in_grid state{
        .grid=input_data,
        .properties{.x_size = width, .y_size = height},
        .position{.x = x, .y = y},
        .time_step = time_step
    };

    save_to(output_data, state.idx(), evaluator_t::evaluate(state));
}

}

template <typename evaluator_type, typename grid_type>
template <_run_mode mode>
void traverser<evaluator_type, grid_type>::run_kernel(int steps) {

    auto current = &_input_grid_cuda;
    auto next = &_intermediate_grid_cuda;

    const idx_type width = current->x_size_physical();
    const idx_type height = current->y_size_physical();

    // Toroidal wrapping - same width and height
    const idx_type width_threads = width;
    const idx_type height_threads = height;

    const dim3 blockDim(_block_size_x, _block_size_y);
    const dim3 gridDim(
        (width_threads + blockDim.x - 1) / blockDim.x,
        (height_threads + blockDim.y - 1) / blockDim.y
    );

    if constexpr (mode == _run_mode::VERBOSE) {
        call_callback(0, current);
    }

    for (int step = 0; step < steps; ++step) {
        process_grid_kernel_simple<evaluator_t><<<gridDim, blockDim>>>(
            current->data(),
            next->data(),
            width,
            height,
            step
        );
        
        if constexpr (mode == _run_mode::VERBOSE) {
            call_callback(step + 1, next);
        }

        std::swap(current, next);

        CUCH(cudaGetLastError());
    }


    _final_grid = current;
}

template <typename evaluator_type, typename grid_type>
auto  traverser<evaluator_type, grid_type>::fetch_result() -> grid_t {
    grid_t cpu_grid = _final_grid->to_cpu();

    _input_grid_cuda.free_cuda_memory();
    _intermediate_grid_cuda.free_cuda_memory();

    return cpu_grid;
}

} // namespace cellato::traversers::cuda::simple

#define SIMPLE_CUDA_TRAVERSER_INSTANTIATIONS

#include "../../../src/game_of_life/cuda_instantiations.cuh"
#include "../../../src/fire/cuda_instantiations.cuh"
#include "../../../src/wire/cuda_instantiations.cuh"
#include "../../../src/excitable/cuda_instantiations.cuh"
#include "../../../src/brian/cuda_instantiations.cuh"
#include "../../../src/maze/cuda_instantiations.cuh"
#include "../../../src/fluid/cuda_instantiations.cuh"
#include "../../../src/critters/cuda_instantiations.cuh"
#include "../../../src/traffic/cuda_instantiations.cuh"
#include "../../../src/cyclic/cuda_instantiations.cuh"

#undef SIMPLE_CUDA_TRAVERSER_INSTANTIATIONS
