#ifndef TILED_BIT_PLANES_GRID_HPP
#define TILED_BIT_PLANES_GRID_HPP

#include <array>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#include "interface.hpp"
#include "standard_grid.hpp"
#include "grid_utils.hpp"

namespace cellato::memory::grids::tiled_bit_planes {

using namespace cellato::memory::grids::utils;
using namespace cellato::memory::grids;

template <typename store_word_type, int needed_bits>
struct cuda_params {
    std::array<store_word_type*, needed_bits> cuda_data;
    std::size_t x_size;
    std::size_t y_size;
};

template <typename store_word_type, typename states_dict_t, device device_type = device::CPU>
class grid {
    public:

    friend class cellato::memory::grids::tiled_bit_planes::grid<store_word_type, states_dict_t, device::CPU>;
    friend class cellato::memory::grids::tiled_bit_planes::grid<store_word_type, states_dict_t, device::CUDA>;

    constexpr static bool HAS_OWN_PRINT = false;

    constexpr static int needed_bits = states_dict_t::needed_bits;
    // TODO this shit should be dealt with
    static constexpr int word_store_bits = sizeof(store_word_type) * 8;

    static constexpr int x_word_tile_size = 8;
    static constexpr int y_word_tile_size = word_store_bits / x_word_tile_size;

    using storage_tuple_t = std::array<std::vector<store_word_type>, needed_bits>;
    using storage_tuple_of_pointers = std::array<store_word_type*, needed_bits>;
    using cuda_params_t = cuda_params<store_word_type, needed_bits>;

    using original_state_t = typename states_dict_t::state_t;

    using cell_t = store_word_type;
    using store_type = cell_t;

    grid() = default;

    grid(std::size_t y_size, std::size_t x_size)
        : _x_size(x_size / x_word_tile_size), _y_size(y_size / y_word_tile_size) {
        if constexpr (device_type == device::CPU) {
            _grid = storage_tuple_t{};

            for_each_bit([&]<std::size_t bit_idx>() {
                std::get<bit_idx>(_grid).resize(y_size_physical() * x_size_physical());
            });
        }
    }

    grid(std::size_t y_size, std::size_t x_size, const original_state_t* grid_input) requires (device_type == device::CPU) {
        initialize_grid(y_size, x_size, grid_input);        
    }

    // CUDA-specific constructor
    grid(cuda_params_t params) requires (device_type == device::CUDA)
        : _x_size(params.x_size), _y_size(params.y_size) {
        for_each_bit([&]<std::size_t bit_idx>() {
            _cuda_data[bit_idx] = params.cuda_data[bit_idx];
        });
    }

    grid(const cellato::memory::grids::standard::grid<original_state_t>& standard_grid) requires (device_type == device::CPU) {
        initialize_grid(
            standard_grid.y_size_physical(), standard_grid.x_size_physical(),
            standard_grid.data());
    }

    void free_cuda_memory() {
        if constexpr (device_type == device::CUDA) {
            for (int i = 0; i < needed_bits; i++) {
                if (_cuda_data[i]) {
                    cudaFree(_cuda_data[i]);
                    _cuda_data[i] = nullptr;
                }
            }
        }
    }

    original_state_t get_cell(std::size_t x, std::size_t y) const requires (device_type == device::CPU) {
        if (x >= x_size_original() || y >= y_size_original()) {
            throw std::out_of_range("Cell coordinates out of range");
        }

        auto word_x = x / x_word_tile_size;
        auto word_y = y / y_word_tile_size;

        auto word_idx = word_y * x_size_physical() + word_x;

        auto bit_x = x % x_word_tile_size;
        auto bit_y = y % y_word_tile_size;

        auto bit_linear_index = bit_y * x_word_tile_size + bit_x;

        int state_idx = 0;        

        for_each_bit([&]<std::size_t bit_idx>() {
            auto word = std::get<bit_idx>(_grid)[word_idx];
            auto bit = (word >> (bit_linear_index)) & 1;
            state_idx |= (bit << bit_idx);
        });

        return states_dict_t::index_to_state(state_idx);
    }

    std::vector<original_state_t> to_original_representation() const requires (device_type == device::CPU) {
        std::vector<original_state_t> result(x_size_original() * y_size_original());

        for (std::size_t y = 0; y < y_size_original(); ++y) {
            for (std::size_t x = 0; x < x_size_original(); ++x) {
                result[y * x_size_original() + x] = get_cell(x, y);
            }
        }

        return result;
    }

    cellato::memory::grids::standard::grid<original_state_t> to_standard() const requires (device_type == device::CPU) {
        auto grid_data = to_original_representation();
        return cellato::memory::grids::standard::grid<original_state_t>(
            std::move(grid_data), x_size_original(), y_size_original()); 
    }

    grid<store_word_type, states_dict_t, device::CUDA> to_cuda() const requires (device_type == device::CPU) {
        std::array<store_word_type*, needed_bits> device_data;
        size_t data_size = y_size_physical() * x_size_physical() * sizeof(store_word_type);
        
        // Allocate and copy each bit plane
        for_each_bit([&]<std::size_t bit_idx>() {
            cudaMalloc((void**)&device_data[bit_idx], data_size);
            cudaMemcpy(device_data[bit_idx], 
                       std::get<bit_idx>(_grid).data(), 
                       data_size, 
                       cudaMemcpyHostToDevice);
        });
        
        cuda_params_t params{
            device_data,
            _x_size,
            _y_size
        };
        
        return grid<store_word_type, states_dict_t, device::CUDA>(params);
    }
    
    grid<store_word_type, states_dict_t, device::CPU> to_cpu() const requires (device_type == device::CUDA) {
        // Create a CPU grid
        grid<store_word_type, states_dict_t, device::CPU> cpu_grid(y_size_original(), x_size_original());
        size_t data_size = y_size_physical() * x_size_physical() * sizeof(store_word_type);
        
        // Copy each bit plane back to host
        for_each_bit([&]<std::size_t bit_idx>() {
            cudaMemcpy(std::get<bit_idx>(cpu_grid._grid).data(), 
                       _cuda_data[bit_idx], 
                       data_size, 
                       cudaMemcpyDeviceToHost);
        });
        
        return cpu_grid;
    }

    std::size_t x_size_original() const {
        return x_size_physical() * x_word_tile_size;
    }

    std::size_t y_size_original() const {
        return y_size_physical() * y_word_tile_size;
    }

    std::size_t x_size_physical() const {
        return _x_size;
    }

    std::size_t y_size_physical() const {
        return _y_size;
    }

    storage_tuple_of_pointers data() {
        storage_tuple_of_pointers result;
        
        if constexpr (device_type == device::CUDA) {
            for_each_bit([&]<std::size_t bit_idx>() {
                std::get<bit_idx>(result) = _cuda_data[bit_idx];
            });
        } else {
            for_each_bit([&]<std::size_t bit_idx>() {
                std::get<bit_idx>(result) = std::get<bit_idx>(_grid).data();
            });
        }
        
        return result;
    }

    storage_tuple_of_pointers data() const {
        storage_tuple_of_pointers result;
        
        if constexpr (device_type == device::CUDA) {
            for_each_bit([&]<std::size_t bit_idx>() {
                std::get<bit_idx>(result) = _cuda_data[bit_idx];
            });
        } else {
            for_each_bit([&]<std::size_t bit_idx>() {
                std::get<bit_idx>(result) = const_cast<store_word_type*>(std::get<bit_idx>(_grid).data());
            });
        }
        
        return result;
    }

  private:
    storage_tuple_t _grid;  // Used for CPU storage
    std::array<store_word_type*, needed_bits> _cuda_data = {};  // Used for CUDA storage

    std::size_t _x_size, _y_size;

    template <typename Callback, std::size_t... Is>
    void for_each_bit_impl(Callback&& cb, std::index_sequence<Is...>) const {
        (cb.template operator()<Is>(), ...);
    }

    template <typename Callback>
    void for_each_bit(Callback&& cb) const {
        for_each_bit_impl(std::forward<Callback>(cb), std::make_index_sequence<needed_bits>{});
    }

    void initialize_grid(std::size_t y_size, std::size_t x_size, const original_state_t* grid_input) {
        _x_size = x_size / x_word_tile_size;
        _y_size = y_size / y_word_tile_size;

        if (x_size % x_word_tile_size != 0 || y_size % y_word_tile_size != 0) {
            throw std::invalid_argument(
                "`x_size` and `y_size` must be multiples of `x_word_tile_size` and `y_word_tile_size`, but was " +
                std::to_string(x_size) + " and " + std::to_string(y_size));
        }

        _grid = storage_tuple_t{};

        for_each_bit([&]<std::size_t bit_idx>() {
            std::get<bit_idx>(_grid).resize(y_size_physical() * x_size_physical());
        });

        for (std::size_t y_physical = 0; y_physical < y_size_physical(); ++y_physical) {
            for (std::size_t x_physical = 0; x_physical < x_size_physical(); ++x_physical) {

                auto word_idx = x_physical + y_physical * x_size_physical();
                
                for_each_bit([&]<std::size_t bit_idx>() {
                    store_type word = 0;

                    for (std::size_t x_bit = 0; x_bit < x_word_tile_size; ++x_bit) {
                        for (std::size_t y_bit = 0; y_bit < y_word_tile_size; ++y_bit) {

                            auto x_grid = x_physical * x_word_tile_size + x_bit;
                            auto y_grid = y_physical * y_word_tile_size + y_bit;

                            auto state = grid_input[y_grid * x_size_original() + x_grid];
                            auto state_idx = states_dict_t::state_to_index(state);

                            store_type state_bit = (state_idx & (1 << bit_idx)) != 0;

                            if (state_bit) {
                                word |= (static_cast<store_word_type>(1) << (x_bit + y_bit * x_word_tile_size));
                            }
                        }
                    }

                    std::get<bit_idx>(_grid)[word_idx] = word;                    
                });
            }
        }
    }
};

}

#endif // TILED_BIT_PLANES_GRID_HPP
