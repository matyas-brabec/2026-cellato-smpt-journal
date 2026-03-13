#ifndef INDEXING_CUH
#define INDEXING_CUH

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include <cstddef>

namespace reference::indexing {

struct indexer {
    static constexpr int x_margin = 0;
    static constexpr int y_margin = 0;
    
    CUDA_CALLABLE indexer(int x_size, int y_size)
        : _x_size(x_size), _y_size(y_size) {};

    CUDA_CALLABLE int at(int x, int y) const {
        const auto x_real = (x + _x_size) % _x_size;
        const auto y_real = (y + _y_size) % _y_size;

        return y_real * _x_size + x_real;
    }

private:
    int _x_size, _y_size;
};

}


#endif // INDEXING_CUH