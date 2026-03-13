#ifndef CELLATO_TRAVERSERS_UTILS_HPP
#define CELLATO_TRAVERSERS_UTILS_HPP

#include <cstddef>

#ifndef CUDA_CALLABLE
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#endif


namespace cellato::traversers::utils {

template <typename grid_data_t, typename idx_type, typename value_t>
CUDA_CALLABLE void save_to(grid_data_t grid, idx_type index, value_t new_value) {
    if constexpr (requires { new_value.save_to(grid, index); }) {
        new_value.save_to(grid, index);
    } else {
        grid[index] = new_value;
    }
}

}

#endif // CELLATO_TRAVERSERS_UTILS_HPP