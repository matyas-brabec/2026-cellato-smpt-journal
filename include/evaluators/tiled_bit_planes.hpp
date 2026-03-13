#ifndef CELLATO_EVALUATORS_TILED_BIT_PLANES_HPP
#define CELLATO_EVALUATORS_TILED_BIT_PLANES_HPP

#include <array>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cstdint>
#include <algorithm>

#include "./bit_planes.hpp"

#include "../core/ast.hpp"
#include "../core/vector_int.hpp"
#include "../memory/bit_planes_grid.hpp"
#include "../memory/grid_utils.hpp"
#include "../memory/interface.hpp"
#include "../memory/idx_type.hpp"

// Use the same CUDA_CALLABLE definition as in standard evaluator
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellato::evaluators::tiled_bit_planes {

using idx_type = cellato::memory::idx_type;

using namespace cellato::ast;
using namespace cellato::core::bitwise;
using namespace cellato::memory::grids::utils;

template <typename cell_row_type, typename state_dictionary_type, template <typename, typename> class recursive_evaluator>
struct implementation_params {
    using cell_row_t = cell_row_type;
    using state_dict_t = state_dictionary_type;
    
    template <typename params, typename Expression>
    using evaluator_t = recursive_evaluator<params, Expression>;
};

template <typename params, typename Expression>
struct _evaluator_impl;

template <typename cell_row_type, typename state_dictionary_type, typename Expression>
using _simple_bit_planes_evaluator_implementation = cellato::evaluators::bit_planes::_evaluator_impl<
    implementation_params<cell_row_type, state_dictionary_type, _evaluator_impl>, Expression>;

template <typename cell_row_type, typename state_dictionary_type>
using grid_cell_data_type = std::array<cell_row_type*, state_dictionary_type::needed_bits>;

template <typename params>
using state_t = cellato::memory::grids::point_in_grid<
    grid_cell_data_type<typename params::cell_row_t, typename params::state_dict_t>>;

template <typename params, typename Expression>
struct _evaluator_impl {
    using eval_state_t = state_t<params>;

    CUDA_CALLABLE static auto evaluate(eval_state_t state) {
        // all but the 'neighbor_at' part is same as bit_planes

        return _simple_bit_planes_evaluator_implementation<typename params::cell_row_t, typename params::state_dict_t, Expression>::evaluate(state);
    }
};

template <typename cell_row_type,  typename state_dictionary_type, typename Expression>
using evaluator = _evaluator_impl<
    implementation_params<
        cell_row_type, state_dictionary_type, _evaluator_impl>,
    Expression>;


// Partial specialization for neighbor_at
template <typename params, idx_type x_offset, idx_type y_offset>
struct _evaluator_impl<params, neighbor_at<x_offset, y_offset>> {
    static constexpr int vector_width_bits = sizeof(typename params::cell_row_t) * 8;

    using eval_state_t = state_t<params>;
    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    CUDA_CALLABLE static auto evaluate(eval_state_t state) {
        auto center = get_center_offsetted(state);
        
        if constexpr (x_offset != 0) {
            auto horizontal_neighbor = get_horizontal_neighbor(state);
            center = center.template get_ored(horizontal_neighbor);
        }

        if constexpr (y_offset != 0) {
            auto vertical_neighbor = get_vertical_neighbor(state);
            center = center.template get_ored(vertical_neighbor);
        }

        if (x_offset != 0 && y_offset != 0) {
            auto diagonal_neighbor = get_diagonal_neighbor(state);
            center = center.template get_ored(diagonal_neighbor);
        }

        return center;
    }

  private:
    using vint = vector_int<cell_row_type, state_dictionary_type::needed_bits>;

    constexpr static int bits_per_cell = sizeof(cell_row_type) * 8;
    constexpr static int x_tile_size = 8;
    constexpr static int y_tile_size = bits_per_cell / x_tile_size;

    static constexpr auto TOP_LINE = static_cast<cell_row_type>(0b1111'1111);
    static constexpr auto BOTTOM_LINE = TOP_LINE << (bits_per_cell - x_tile_size);
    // TODO: RIGHT_BORDER and LEFT_BORDER should be generalized for other than offsets from [-1, 0, 1]
    static constexpr auto RIGHT_BORDER = static_cast<cell_row_type>(0x80'80'80'80'80'80'80'80LLU); 
    static constexpr auto LEFT_BORDER = static_cast<cell_row_type>(0x01'01'01'01'01'01'01'01LLU);

    CUDA_CALLABLE static vint get_center_offsetted(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto center = load_at(state, x, y);

        if constexpr (y_offset > 0) {
            center = center.template get_right_shifted_vector<y_offset * x_tile_size>();
        } else if constexpr (y_offset < 0) {
            center = center.template get_left_shifted_vector<-y_offset * x_tile_size>();
        }

        if constexpr (x_offset > 0) {
            center = center
                .template get_right_shifted_vector<x_offset>()
                .template get_ANDed_each_plane_with<~RIGHT_BORDER>();
        } else if constexpr (x_offset < 0) {
            center = center
                .template get_left_shifted_vector<-x_offset>()
                .template get_ANDed_each_plane_with<~LEFT_BORDER>(); 
        }

        return center;
    }

    CUDA_CALLABLE static vint get_horizontal_neighbor(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto neighbor_x = load_at(state, x + x_offset, y);

        constexpr int shift = x_tile_size - abs(x_offset);

        if constexpr (x_offset > 0) {
            neighbor_x = neighbor_x
                .template get_left_shifted_vector<shift>()
                .template get_ANDed_each_plane_with<RIGHT_BORDER>();
        } else if constexpr (x_offset < 0) {
            neighbor_x = neighbor_x
                .template get_right_shifted_vector<shift>()
                .template get_ANDed_each_plane_with<LEFT_BORDER>();
        }

        static_assert(x_offset != 0, "x_offset must be non-zero");

        if constexpr (y_offset > 0) {
            neighbor_x = neighbor_x.template get_right_shifted_vector<y_offset * x_tile_size>();
        } else if constexpr (y_offset < 0) {
            neighbor_x = neighbor_x.template get_left_shifted_vector<-y_offset * x_tile_size>();
        }

        return neighbor_x;
    }

    CUDA_CALLABLE static vint get_vertical_neighbor(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto neighbor_y = load_at(state, x, y + y_offset);

        constexpr int shift = (y_tile_size - abs(y_offset)) * x_tile_size;

        if constexpr (y_offset > 0) {
            neighbor_y = neighbor_y.template get_left_shifted_vector<shift>();
        } else if constexpr (y_offset < 0) {
            neighbor_y = neighbor_y.template get_right_shifted_vector<shift>();
        }
        
        static_assert(y_offset != 0, "y_offset must be non-zero");
        
        if constexpr (x_offset > 0) {
            neighbor_y = neighbor_y
                .template get_right_shifted_vector<x_offset>()
                .template get_ANDed_each_plane_with<~RIGHT_BORDER>();
        } else if constexpr (x_offset < 0) {
            neighbor_y = neighbor_y
                .template get_left_shifted_vector<-x_offset>()
                .template get_ANDed_each_plane_with<~LEFT_BORDER>();
        }

        return neighbor_y;
    }

    CUDA_CALLABLE static vint get_diagonal_neighbor(eval_state_t state) {
        // TODO: right now only works for offsets in {-1, 1}
        
        auto x = state.position.x;
        auto y = state.position.y;
        auto neighbor_xy = load_at(state, x + x_offset, y + y_offset);

        // Bottom-right
        if constexpr (x_offset > 0 && y_offset > 0) {
            neighbor_xy = neighbor_xy
                .template get_left_shifted_vector<bits_per_cell - 1>();
        
        // Top-right
        } else if constexpr (x_offset > 0 && y_offset < 0) {
            constexpr static auto corner_mask = TOP_LINE & RIGHT_BORDER;
            
            neighbor_xy = neighbor_xy
                .template get_right_shifted_vector<(y_tile_size - 2) * x_tile_size + 1>()
                .template get_ANDed_each_plane_with<corner_mask>();
            
        // Bottom-left
        } else if constexpr (x_offset < 0 && y_offset > 0) {
            constexpr static auto corner_mask = BOTTOM_LINE & LEFT_BORDER;
            
            neighbor_xy = neighbor_xy
                .template get_left_shifted_vector<(y_tile_size - 2) * x_tile_size + 1>()
                .template get_ANDed_each_plane_with<corner_mask>();

        // Top-left
        } else if constexpr (x_offset < 0 && y_offset < 0) {
           neighbor_xy = neighbor_xy
                .template get_right_shifted_vector<bits_per_cell - 1>();
        }

        return neighbor_xy;
    }

    CUDA_CALLABLE static vint load_at(eval_state_t state, std::size_t x, std::size_t y) {
        auto idx = state.properties.idx(x, y);
        return vector_int_factory::load_from<cell_row_type>(state.grid, idx);
    }

    constexpr static int abs(int v) {
        return v < 0 ? -v : v;
    }
};

template <typename params, typename cell_state_type, cell_state_type CellStateValue>
struct _evaluator_impl<
    params,
    count_neighbors<
        state_constant<CellStateValue>,
        margolus_alternating_neighborhood>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    using vint3 = vector_int<cell_row_type, 3>;

    constexpr static auto cell_state = state_dictionary_type::state_to_index(CellStateValue);

    CUDA_CALLABLE static vint3 evaluate(state_t<params> state) {
        auto parity = state.time_step % 2;

        if (parity == 0) {
            // return count_with_offset< 1>(state); // slower version
            return fast_even_parity(state);
        } else {
            return count_with_offset<-1>(state);
        }
    }

  private:
    constexpr static cell_row_type top_left_bit = static_cast<cell_row_type>(0x0055'0055'0055'0055LLU);
    constexpr static cell_row_type top_right_bit = top_left_bit << 1;
    constexpr static cell_row_type bottom_left_bit = top_left_bit << 8;
    constexpr static cell_row_type bottom_right_bit = bottom_left_bit << 1;

    template <int modifier>
    CUDA_CALLABLE static vint3 count_with_offset(state_t<params> state) {
        auto bottom_right = count_block< 0, 1, 0, 1, modifier>(state);
        auto bottom_left  = count_block<-1, 0, 0, 1, modifier>(state);
        auto top_right    = count_block< 0, 1,-1, 0, modifier>(state);
        auto top_left     = count_block<-1, 0,-1, 0, modifier>(state);

        auto result_for_top_left = bottom_right.mask_out_columns(top_left_bit);
        auto result_for_top_right = bottom_left.mask_out_columns(top_right_bit);
        auto result_for_bottom_left = top_right.mask_out_columns(bottom_left_bit);
        auto result_for_bottom_right = top_left.mask_out_columns(bottom_right_bit);

        return result_for_top_left
            .get_ored(result_for_top_right)
            .get_ored(result_for_bottom_left)
            .get_ored(result_for_bottom_right);
    }
    
    template <int x0, int x1, int y0, int y1, int modifier>
    CUDA_CALLABLE static vint3 count_block(state_t<params> state) {
        constexpr static int x0_mod = x0 * modifier;
        constexpr static int x1_mod = x1 * modifier;
        constexpr static int y0_mod = y0 * modifier;
        constexpr static int y1_mod = y1 * modifier;

        auto c00_c = evaluator_t<neighbor_at<x0_mod, y0_mod>>::evaluate(state).template equals_to<cell_state>();
        auto c01_c = evaluator_t<neighbor_at<x0_mod, y1_mod>>::evaluate(state).template equals_to<cell_state>();
        auto c10_c = evaluator_t<neighbor_at<x1_mod, y0_mod>>::evaluate(state).template equals_to<cell_state>();
        auto c11_c = evaluator_t<neighbor_at<x1_mod, y1_mod>>::evaluate(state).template equals_to<cell_state>();

        auto c00 = vector_int_factory::from_condition_result<cell_row_type>(c00_c);
        auto c01 = vector_int_factory::from_condition_result<cell_row_type>(c01_c);
        auto c10 = vector_int_factory::from_condition_result<cell_row_type>(c10_c);
        auto c11 = vector_int_factory::from_condition_result<cell_row_type>(c11_c);

        return c00.template to_vector_with_bits<2>()
            .get_added(c01)
            .get_added(c10).template to_vector_with_bits<3>()
            .get_added(c11);
    }

    CUDA_CALLABLE static vint3 fast_even_parity(state_t<params> state) {
        auto current_state_c = evaluator_t<neighbor_at< 0,  0>>::evaluate(state).template equals_to<cell_state>();
        auto current_state = vector_int_factory::from_condition_result<cell_row_type>(current_state_c);

        auto switched_pairs = current_state.get_with_switched_pairs_of_numbers();

        auto pairs_summed = current_state
            .template to_vector_with_bits<2>()
            .get_added(switched_pairs);

        auto switched_rows_of_blocks = pairs_summed.get_with_switched_rows_of_8();

        return pairs_summed
            .template to_vector_with_bits<3>()
            .get_added(switched_rows_of_blocks);
    }
};

template <typename params>
struct _evaluator_impl<
    params,
    margolus_180_neighbor> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto parity = state.time_step % 2;

        if (parity == 0) {
            // return get_with_offset< 1>(state); // slower version
            return fast_even_parity(state);
        } else {
            return get_with_offset<-1>(state);
        }
    }

  private:
    using cell_row_type = typename params::cell_row_t;
    
    constexpr static cell_row_type top_left_bit = static_cast<cell_row_type>(0x0055'0055'0055'0055LLU);
    constexpr static cell_row_type top_right_bit = top_left_bit << 1;
    constexpr static cell_row_type bottom_left_bit = top_left_bit << 8;
    constexpr static cell_row_type bottom_right_bit = bottom_left_bit << 1;

    template <int modifier>
    CUDA_CALLABLE static auto get_with_offset(state_t<params> state) {
        auto bottom_right = get_margolus_neighbor< 1, 1, modifier>(state);
        auto bottom_left  = get_margolus_neighbor<-1, 1, modifier>(state);
        auto top_right    = get_margolus_neighbor< 1,-1, modifier>(state);
        auto top_left     = get_margolus_neighbor<-1,-1, modifier>(state);

        auto result_for_top_left = bottom_right.mask_out_columns(top_left_bit);
        auto result_for_top_right = bottom_left.mask_out_columns(top_right_bit);
        auto result_for_bottom_left = top_right.mask_out_columns(bottom_left_bit);
        auto result_for_bottom_right = top_left.mask_out_columns(bottom_right_bit);

        return result_for_top_left
            .get_ored(result_for_top_right)
            .get_ored(result_for_bottom_left)
            .get_ored(result_for_bottom_right);
    }

    template <int x, int y, int modifier>
    CUDA_CALLABLE static auto get_margolus_neighbor(state_t<params> state)
    {
        return evaluator_t<neighbor_at<x * modifier, y * modifier>>::evaluate(state);
    }

    CUDA_CALLABLE static auto fast_even_parity(state_t<params> state) {
        auto current_state = evaluator_t<neighbor_at< 0,  0>>::evaluate(state);
        return current_state.get_with_switched_pairs_of_numbers().get_with_switched_rows_of_8();
    }
};


} // namespace cellato::evaluators::tiled_bit_planes

#endif // CELLATO_EVALUATORS_TILED_BIT_PLANES_HPP
