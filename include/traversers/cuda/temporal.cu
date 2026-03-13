#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <array>
#include <type_traits>
#include <cstddef> 

#include "./temporal.hpp"
#include "../../memory/standard_grid.hpp"
#include "../../memory/interface.hpp"
#include "../../memory/idx_type.hpp"
#include "../../evaluators/standard.hpp"
#include "../../core/ast.hpp"
#include "../traverser_utils.hpp"
#include "../cuda_utils.cuh"
#include "../../utils/static_dispatcher.hpp"
#include "../../traversers/temporal_utils.cuh"

namespace cellato::traversers::cuda::temporal {

using namespace cellato::traversers::temporal_utils;

using idx_type = cellato::memory::idx_type;

template <
    typename evaluator_t,

    idx_type temporal_steps, idx_type temporal_tile_size_y,
    idx_type word_tile_x, idx_type word_tile_y, double average_halo_radius,

    idx_type block_size_x, idx_type block_size_y,

    typename grid_data_t, typename output_data_t>

__global__ void process_grid_kernel_linear_temporal(
    grid_data_t input_data,
    output_data_t output_data,
    idx_type width,
    idx_type height,
    idx_type time_step
) {
    using grid_props = props<grid_data_t>;
    using store_t = typename grid_props::no_pointer_type;
    using prt_t = typename grid_props::ptr_type;

    constexpr idx_type cells_per_thread_y = temporal_tile_size_y / block_size_y;
    constexpr idx_type temporal_tile_size_x = block_size_x;

    constexpr idx_type needed_halo_cells = static_cast<idx_type>(std::ceil(average_halo_radius * temporal_steps * 0.999));
    constexpr idx_type x_halo_words = (needed_halo_cells + word_tile_x - 1) / word_tile_x;
    constexpr idx_type y_halo_words = (needed_halo_cells + word_tile_y - 1) / word_tile_y;

    constexpr idx_type effective_temporal_tile_size_x = temporal_tile_size_x - (2 * x_halo_words);
    constexpr idx_type effective_temporal_tile_size_y = temporal_tile_size_y - (2 * y_halo_words);

    idx_type global_x_non_wrapped = static_cast<idx_type>(blockIdx.x) * effective_temporal_tile_size_x + threadIdx.x - x_halo_words;
    idx_type global_y_start_non_wrapped = static_cast<idx_type>(blockIdx.y) * effective_temporal_tile_size_y + 
                                          static_cast<idx_type>(threadIdx.y) * cells_per_thread_y - y_halo_words;

    idx_type global_x_for_load = (global_x_non_wrapped + width) % width;
    idx_type global_x_for_save = global_x_non_wrapped;

    idx_type local_x = threadIdx.x;
    idx_type local_y_start = threadIdx.y * cells_per_thread_y;

    constexpr idx_type cells_in_temporal_blocks = temporal_tile_size_y * temporal_tile_size_x;

    extern __shared__ char s_buffer[];
    store_t* buffer_base = reinterpret_cast<store_t*>(s_buffer);

    store_t* buffer1 = buffer_base;
    store_t* buffer2 = buffer_base + grid_props::compute_buffer_size(cells_in_temporal_blocks);
    
    grid_data_t grid_buffer1 = grid_props::create_from_contiguous(buffer1, cells_in_temporal_blocks);
    grid_data_t grid_buffer2 = grid_props::create_from_contiguous(buffer2, cells_in_temporal_blocks);

    for (idx_type y_offset = 0; y_offset < cells_per_thread_y; ++y_offset) {
        idx_type global_y_for_load = (global_y_start_non_wrapped + y_offset + height) % height;

        grid_props::assign_to_from(
            grid_buffer1, temporal_tile_size_x,
            local_x, local_y_start + y_offset,

            input_data, width,
            global_x_for_load, global_y_for_load
        );
    }

    grid_data_t current = grid_buffer1;
    grid_data_t next = grid_buffer2;

    __syncthreads();

    for (int t = 0; t < temporal_steps; ++t) {
        for (idx_type y_offset = 0; y_offset < cells_per_thread_y; ++y_offset) {
            cellato::memory::grids::point_in_grid state(current);

            state.properties.x_size = temporal_tile_size_x;
            state.properties.y_size = temporal_tile_size_y;

            state.position.x = local_x;
            state.position.y = local_y_start + y_offset;

            state.time_step = time_step + t;

            auto result = evaluator_t::evaluate(state);
            save_to(next, state.idx(), result);
        }

        auto temp = current;
        current = next;
        next = temp;

        __syncthreads();
    }

    if (threadIdx.x < x_halo_words || threadIdx.x >= (temporal_tile_size_x - x_halo_words))
        return;

    for (idx_type y_offset = 0; y_offset < cells_per_thread_y; ++y_offset) {
        idx_type global_y_for_save = global_y_start_non_wrapped + y_offset;
        idx_type local_y = local_y_start + y_offset;

        if (local_y < y_halo_words || local_y >= (temporal_tile_size_y - y_halo_words))
            continue;

        grid_props::assign_to_from(
            output_data, width,
            global_x_for_save, global_y_for_save,

            current, temporal_tile_size_x,
            local_x, local_y
        );
    }
}

template <typename evaluator_type, typename grid_type, double average_halo_radius>
template <_run_mode mode>
void traverser<evaluator_type, grid_type, average_halo_radius>::run_kernel(int steps) {
    auto current = &_input_grid_cuda;
    auto next = &_intermediate_grid_cuda;

    idx_type width = current->x_size_physical();
    idx_type height = current->y_size_physical();

    idx_type width_threads = width;
    idx_type height_threads = height;

    dim3 blockDim(_block_size_x, _block_size_y);
    dim3 gridDim(
        width_threads / _effective_temporal_tile_size_x,
        height_threads / _effective_temporal_tile_size_y
    );

    if constexpr (mode == _run_mode::VERBOSE) {
        call_callback(0, current);
    }

    // Has to be fixed to 32 because of warp size
    using cuda_th_block_x_opts        = std::integer_sequence<idx_type, 32>;
    
    // Others can be adjusted:

    // For the full benchmarking
    #ifdef BENCHMARK_COMPILE
    using temporal_steps_opts        = std::integer_sequence<idx_type, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 20, 22, 24>;
    using temporal_block_size_Y_opts = std::integer_sequence<idx_type, 8, 16, 32, 64, 128>;
    using cuda_th_block_y_opts       = std::integer_sequence<idx_type, 2, 4, 8, 16>;
    #endif

    // Verification compilation (for <repo root>/src/_scripts/cluster_run/verify.py script)
    #ifdef VERIFICATION_COMPILE
    using temporal_steps_opts        = std::integer_sequence<idx_type, 4, 8, 12, 20>;
    using temporal_block_size_Y_opts = std::integer_sequence<idx_type, 8, 32>;
    using cuda_th_block_y_opts       = std::integer_sequence<idx_type, 2, 4>;
    #endif

    // Custom compilation -- FOR USER EDITS
    #ifndef BENCHMARK_COMPILE
    #ifndef VERIFICATION_COMPILE

    // possible values for --temporal_steps
    using temporal_steps_opts        = std::integer_sequence<idx_type, 4>;

    // possible values for --temporal_tile_size_y
    using temporal_block_size_Y_opts = std::integer_sequence<idx_type, 32>;
    
    // possible values for --cuda_block_size_y
    using cuda_th_block_y_opts       = std::integer_sequence<idx_type, 8>;
    
    #endif
    #endif

    cellato::generic_dispatcher::call<
        temporal_steps_opts,
        temporal_block_size_Y_opts, 
        
        cuda_th_block_x_opts,
        cuda_th_block_y_opts
    >(
        [&]<
            idx_type temporal_steps, idx_type temporal_tile_size_y,
            idx_type block_size_x, idx_type block_size_y
        >() {
            constexpr idx_type temporal_tile_size_x = block_size_x;
            constexpr idx_type required_buffers_bytes = 2 * temporal_tile_size_y * temporal_tile_size_x * grid_type::needed_bits * sizeof(typename grid_type::cell_t);

            constexpr idx_type needed_halo_cells = static_cast<idx_type>(std::ceil(average_halo_radius * temporal_steps * 0.999));
            constexpr int y_halo_words = (needed_halo_cells + word_tile_y - 1) / word_tile_y;
            constexpr int x_halo_words = (needed_halo_cells + word_tile_x - 1) / word_tile_x;
            constexpr int effective_y_tile_size = static_cast<int>(temporal_tile_size_y) - (2 * y_halo_words);
            constexpr int effective_x_tile_size = static_cast<int>(temporal_tile_size_x) - (2 * x_halo_words);

            if constexpr (required_buffers_bytes > max_shm_size) {
                throw std::runtime_error("Configuration exceeds maximum shared memory size. The temporal tile is too large.");
                
            } else if constexpr (temporal_tile_size_y < block_size_y) {
                throw std::runtime_error("Invalid configuration: block_size_y must be less than or equal to temporal_tile_size_y");

            } else if constexpr (effective_y_tile_size <= 0) {
                throw std::runtime_error("Invalid configuration: effective_y_tile_size must be greater than 0.");

            } else if constexpr (effective_x_tile_size <= 0) {
                throw std::runtime_error("Invalid configuration: effective_x_tile_size must be greater than 0.");

            } else {
                for (int step = 0; step < steps; step += temporal_steps) {
                    auto input_data = current->data();
                    auto output_data = next->data();

                    cudaFuncSetAttribute(
                        process_grid_kernel_linear_temporal<
                            evaluator_type,
                            temporal_steps, temporal_tile_size_y,
                            word_tile_x, word_tile_y, average_halo_radius,
                            block_size_x, block_size_y,
                            decltype(input_data), decltype(output_data)
                        >,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        cudaSharedmemCarveoutMaxShared
                    );

                    process_grid_kernel_linear_temporal<
                        evaluator_type,
                        temporal_steps, temporal_tile_size_y,
                        word_tile_x, word_tile_y, average_halo_radius,
                        block_size_x, block_size_y
                    ><<<gridDim, blockDim, required_buffers_bytes>>>(
                        input_data,
                        output_data,
                        width,
                        height,
                        step
                    );

                    if constexpr (mode == _run_mode::VERBOSE) {
                        call_callback(step + temporal_steps, next);
                    }

                    std::swap(current, next);
                    CUCH(cudaGetLastError());
                }
            }
        },
        (idx_type)_temporal_steps,
        (idx_type)_temporal_tile_size_y,
        (idx_type)_block_size_x,
        (idx_type)_block_size_y
    );
    
    _final_grid = current;
}

template <typename evaluator_type, typename grid_type, double average_halo_radius>
auto traverser<evaluator_type, grid_type, average_halo_radius>::fetch_result() -> grid_t {
    grid_t cpu_grid = _final_grid->to_cpu();

    _input_grid_cuda.free_cuda_memory();
    _intermediate_grid_cuda.free_cuda_memory();

    return cpu_grid;
}

} // namespace cellato::traversers::cuda::temporal

#define LINEAR_TEMPORAL_CUDA_TRAVERSER_INSTANTIATIONS
#define TILED_TEMPORAL_CUDA_TRAVERSER_INSTANTIATIONS

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

#undef LINEAR_TEMPORAL_CUDA_TRAVERSER_INSTANTIATIONS
#undef TILED_TEMPORAL_CUDA_TRAVERSER_INSTANTIATIONS
