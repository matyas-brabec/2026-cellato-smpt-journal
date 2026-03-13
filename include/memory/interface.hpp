#ifndef CELLATO_MEMORY_INTERFACE_HPP
#define CELLATO_MEMORY_INTERFACE_HPP

#include <cstddef>
#ifndef CUDA_CALLABLE
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#endif

#include "idx_type.hpp"

namespace cellato::memory::grids {

using idx_type = cellato::memory::idx_type;

enum class device {
    CPU,
    CUDA
};

struct properties {
    idx_type x_size;
    idx_type y_size;

    CUDA_CALLABLE idx_type idx(idx_type x, idx_type y) const {
        const auto x_real = (x + x_size) % x_size;
        const auto y_real = (y + y_size) % y_size;

        return y_real * x_size + x_real;
    }
};

struct point {
    idx_type x;
    idx_type y;
};

template <typename grid_data_type>
struct point_in_grid {
    using grid_t = grid_data_type;

    grid_data_type grid{};

    grids::properties properties{};

    grids::point position{};

    idx_type time_step = 0;

    CUDA_CALLABLE idx_type idx() const {
        return properties.idx(position.x, position.y);
    }

    CUDA_CALLABLE idx_type idx(idx_type x, idx_type y) const {
        return properties.idx(x, y);
    }
};

}

#endif // CELLATO_MEMORY_INTERFACE_HPP