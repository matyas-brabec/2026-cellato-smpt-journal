#ifndef CELLATO_EVALUATORS_BIT_PLANES_HPP
#define CELLATO_EVALUATORS_BIT_PLANES_HPP

#include <array>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cstdint>
#include <algorithm>

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

namespace cellato::evaluators::bit_planes {

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
struct _evaluator_impl {};

template <typename cell_row_type,  typename state_dictionary_type, typename Expression>
using evaluator = _evaluator_impl<implementation_params<cell_row_type, state_dictionary_type, _evaluator_impl>, Expression>;

template <typename cell_row_type, typename state_dictionary_type>
using grid_cell_data_type = std::array<cell_row_type*, state_dictionary_type::needed_bits>;

template <typename params>
using state_t = cellato::memory::grids::point_in_grid<
    grid_cell_data_type<typename params::cell_row_t, typename params::state_dict_t>>;

template <typename params, auto Value>
struct _evaluator_impl<params, constant<Value>> {
    CUDA_CALLABLE static auto evaluate(state_t<params> /* state */) {
        return vector_int_factory::from_constant<typename params::cell_row_t, Value>();
    }
};

template <typename params, auto Value>
struct _evaluator_impl<params, state_constant<Value>> {
    using state_dict_type = typename params::state_dict_t;

    CUDA_CALLABLE static auto evaluate(state_t<params> /* state */) {
        constexpr auto index = state_dict_type::state_to_index(Value);
        return vector_int_factory::from_constant<typename params::cell_row_t, index>();
    }
};

template <typename params, typename Even, typename Odd>
struct _evaluator_impl<params, alternate_algorithms<Even, Odd>> {

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        if (state.time_step % 2 == 0) {
            return _evaluator_impl<params, Even>::evaluate(state);
        } else {
            return _evaluator_impl<params, Odd>::evaluate(state);
        }
    }
};

template <typename params, typename Condition, typename Then, typename Else>
struct _evaluator_impl<params, if_then_else<Condition, Then, Else>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto condition = evaluator_t<Condition>::evaluate(state);
        auto then_part = evaluator_t<Then>::evaluate(state);
        auto else_part = evaluator_t<Else>::evaluate(state);

        auto masked_then = then_part.mask_out_columns(condition);
        auto masked_else = else_part.mask_out_columns(~condition);

        if constexpr (requires { masked_then.get_ored(masked_else); }) {
            return masked_then.get_ored(masked_else);
        } else {
            return masked_then | masked_else;
        }
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, bit_and_<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        if constexpr (requires { left.get_anded(right); }) {
            return left.get_anded(right);
        } else {
            return left & right;
        }
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, bit_or_<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        if constexpr (requires { left.get_ored(right); }) {
            return left.get_ored(right);
        } else {
            return left | right;
        }
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, plus<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        if constexpr (requires { left.get_added(right); }) {
            return left.get_added(right);
        } else {
            return left + right;
        }
    }
};

// Only modulo by constant is supported for now (and it has to be a power of two)
template <typename params, typename Left, int ConstValue>
struct _evaluator_impl<params, modulo<Left, constant<ConstValue>>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    static_assert((ConstValue & (ConstValue - 1)) == 0, "Only modulo by power of two is supported");

    static constexpr int log2(int n) {
        return (n < 2) ? 0 : 1 + log2(n / 2);
    }

    static constexpr int number_of_bits = log2(ConstValue);

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        return left.template to_vector_with_bits<number_of_bits>();
    }
};

template <typename params, typename Value>
struct _evaluator_impl<params, not_<Value>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        typename params::cell_row_t value = evaluator_t<Value>::evaluate(state);
        return ~value;
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, and_<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        typename params::cell_row_t left = evaluator_t<Left>::evaluate(state);
        typename params::cell_row_t right = evaluator_t<Right>::evaluate(state);

        return left & right;
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, or_<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        typename params::cell_row_t left = evaluator_t<Left>::evaluate(state);
        typename params::cell_row_t right = evaluator_t<Right>::evaluate(state);

        return left | right;
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, equals<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        return left.equals_to(right);
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, greater_than<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        return left.greater_than(right);
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, less_than<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        return left.less_than(right);
    }
};

template <typename params, typename Left, typename Right>
struct _evaluator_impl<params, not_equals<Left, Right>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static bool evaluate(state_t<params> state) {
        auto left = evaluator_t<Left>::evaluate(state);
        auto right = evaluator_t<Right>::evaluate(state);

        return left.not_equal_to(right);
    }
};

template <typename params, typename Value, idx_type bit_idx>
struct _evaluator_impl<params, has_bit_set<Value, bit_idx>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto val = evaluator_t<Value>::evaluate(state);
        return val.template get_bit<bit_idx>();
    }
};


template <typename params, idx_type x_offset, idx_type y_offset>
struct _evaluator_impl<params, neighbor_at<x_offset, y_offset>> {
    static constexpr int vector_width_bits = sizeof(typename params::cell_row_t) * 8;

    using eval_state_t = state_t<params>;
    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    CUDA_CALLABLE static auto evaluate(eval_state_t state) {
        auto center = get_center_vector_int(state);

        if constexpr (x_offset == 0) {
            return center;
        }

        auto neighbor = get_neighbor_vector_int(state);

        auto shifted_center = shift_center(center);
        auto shifted_neighbor = shift_neighbor(neighbor);

        return shifted_center.get_ored(shifted_neighbor);
    }

  private:
    using vint = vector_int<cell_row_type, state_dictionary_type::needed_bits>;

    CUDA_CALLABLE static vint shift_center(vint center) {
        if constexpr (x_offset > 0) {
            return center.template get_right_shifted_vector<x_offset>();
        } else if constexpr (x_offset < 0) {
            return center.template get_left_shifted_vector<-x_offset>();
        } else {
            #ifndef __CUDA_ARCH__
            throw std::logic_error("Invalid x_offset value");
            #else
            // In CUDA device code, we can't throw exceptions
            // Just return the unshifted center as a fallback
            return center;
            #endif
        }
    }

    CUDA_CALLABLE static vint shift_neighbor(vint neighbor) {
        if constexpr (x_offset > 0) {
            return neighbor.template get_left_shifted_vector<vector_width_bits - x_offset>();
        } else if constexpr (x_offset < 0) {
            return neighbor.template get_right_shifted_vector<vector_width_bits + x_offset>();
        } else {
            #ifndef __CUDA_ARCH__
            throw std::logic_error("Invalid x_offset value");
            #else
            // In CUDA device code, we can't throw exceptions
            return neighbor;
            #endif
        }
    }

    CUDA_CALLABLE static vint get_center_vector_int(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto idx = state.properties.idx(x, y + y_offset);

        return vector_int_factory::load_from<cell_row_type>(state.grid, idx);
    }

    CUDA_CALLABLE static vint get_neighbor_vector_int(eval_state_t state) {
        auto x = state.position.x;
        auto y = state.position.y;
        auto idx = state.properties.idx(x + x_offset, y + y_offset);

        return vector_int_factory::load_from<cell_row_type>(state.grid, idx);
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

    constexpr static auto cell_state = state_dictionary_type::state_to_index(CellStateValue);
    
    CUDA_CALLABLE static vector_int<typename params::cell_row_t, 3> evaluate(state_t<params> state) {
        auto parity = state.time_step % 2;

        if (parity == 0) {
            return even_parity(state);
        } else {
            return odd_parity(state);
        }
    }

private:
    CUDA_CALLABLE static vector_int<cell_row_type, 3> even_parity(state_t<params> state) {
        auto current_state_c = evaluator_t<neighbor_at< 0,  0>>::evaluate(state).template equals_to<cell_state>();

        auto vertical_neighbor_c = (y_parity(state) == 0) ?
            evaluator_t<neighbor_at< 0,   1>>::evaluate(state).template equals_to<cell_state>() :
            evaluator_t<neighbor_at< 0,  -1>>::evaluate(state).template equals_to<cell_state>();

        auto current_state = vector_int_factory::from_condition_result<cell_row_type>(current_state_c);
        auto vertical_neighbor = vector_int_factory::from_condition_result<cell_row_type>(vertical_neighbor_c);

        auto columns_sum = current_state.template to_vector_with_bits<2>().get_added(vertical_neighbor);

        auto switched_pairs = columns_sum.get_with_switched_pairs_of_numbers();

        return columns_sum.template to_vector_with_bits<3>()
            .get_added(switched_pairs);
    }

    CUDA_CALLABLE static vector_int<cell_row_type, 3> odd_parity(state_t<params> state) {
        cell_row_type current_state_c = evaluator_t<neighbor_at<1,  0>>::evaluate(state).template equals_to<cell_state>();
        cell_row_type vertical_neighbor_c;

        if (y_parity(state) == 0) {
            vertical_neighbor_c = evaluator_t<neighbor_at<1, -1>>::evaluate(state).template equals_to<cell_state>();
        } else {
            vertical_neighbor_c = evaluator_t<neighbor_at<1,  1>>::evaluate(state).template equals_to<cell_state>();
        }

        auto current_state = vector_int_factory::from_condition_result<cell_row_type>(current_state_c);
        auto vertical_neighbor = vector_int_factory::from_condition_result<cell_row_type>(vertical_neighbor_c);

        auto columns_sum = current_state.template to_vector_with_bits<2>().get_added(vertical_neighbor);

        auto switched_pairs = columns_sum.get_with_switched_pairs_of_numbers();

        auto result = columns_sum.template to_vector_with_bits<3>()
            .get_added(switched_pairs).template get_left_shifted_vector<1>();

        int last_bit_result = solve_for_last_bit(state);
        result.set_at(0, last_bit_result);

        return result;
    }

    CUDA_CALLABLE static int solve_for_last_bit(state_t<params> state) {
        cell_row_type current_state_c = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).template equals_to<cell_state>();
        cell_row_type vertical_neighbor_c;


        if (y_parity(state) == 0) {
            vertical_neighbor_c = evaluator_t<neighbor_at<-1, -1>>::evaluate(state).template equals_to<cell_state>();
        } else {
            vertical_neighbor_c = evaluator_t<neighbor_at<-1,  1>>::evaluate(state).template equals_to<cell_state>();
        }

        constexpr int counts[4] = {0, 1, 1, 2};
        return counts[vertical_neighbor_c & 0b11] + counts[current_state_c & 0b11];
    }

    CUDA_CALLABLE static int y_parity(state_t<params> state) {
        return state.position.y % 2;
    }
};

template <typename params>
struct _evaluator_impl<params, margolus_180_neighbor> {
    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    using state_vint_type = vector_int<cell_row_type, state_dictionary_type::needed_bits>;
    
    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto parity = state.time_step % 2;

        if (parity == 0) {
            return even_parity(state);
        } else {
            return odd_parity(state);
        }
    }
  private:
    CUDA_CALLABLE static auto even_parity(state_t<params> state) {
        auto vertical_neighbor = (y_parity(state) == 0) ?
            evaluator_t<neighbor_at< 0,   1>>::evaluate(state) :
            evaluator_t<neighbor_at< 0,  -1>>::evaluate(state);

        return vertical_neighbor.get_with_switched_pairs_of_numbers();
    }

    CUDA_CALLABLE static auto odd_parity(state_t<params> state) {
        auto even_cells = (y_parity(state) == 0) ?
            evaluator_t<neighbor_at< 1, -1>>::evaluate(state) :
            evaluator_t<neighbor_at< 1,  1>>::evaluate(state);

        auto odd_cells = (y_parity(state) == 0) ?
            evaluator_t<neighbor_at<-1, -1>>::evaluate(state) :
            evaluator_t<neighbor_at<-1,  1>>::evaluate(state);

        constexpr cell_row_type alternating_mask_0_at_0 = static_cast<cell_row_type>(0xAAAAAAAAAAAAAAAALLU);
        constexpr cell_row_type alternating_mask_1_at_0 = static_cast<cell_row_type>(0x5555555555555555LLU);

        auto even_masked = even_cells.mask_out_columns(alternating_mask_0_at_0);
        auto odd_masked  = odd_cells.mask_out_columns(alternating_mask_1_at_0);

        return even_masked.get_ored(odd_masked);
    }

    CUDA_CALLABLE static int y_parity(state_t<params> state) {
        return state.position.y % 2;
    }
};

template <typename params, typename cell_state_type, cell_state_type CellStateValue>
struct _evaluator_impl<
    params,
    count_neighbors<
        state_constant<CellStateValue>,
        moore_8_neighbors>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    CUDA_CALLABLE static vector_int<typename params::cell_row_t, 4> evaluate(state_t<params> state) {
        constexpr auto cell_state = state_dictionary_type::state_to_index(CellStateValue);

        auto top_left_c     = evaluator_t<neighbor_at<-1, -1>>::evaluate(state).template equals_to<cell_state>();
        auto top_c          = evaluator_t<neighbor_at< 0, -1>>::evaluate(state).template equals_to<cell_state>();
        auto top_right_c    = evaluator_t<neighbor_at< 1, -1>>::evaluate(state).template equals_to<cell_state>();
        auto left_c         = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).template equals_to<cell_state>();
        auto right_c        = evaluator_t<neighbor_at< 1,  0>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_left_c  = evaluator_t<neighbor_at<-1,  1>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_c       = evaluator_t<neighbor_at< 0,  1>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_right_c = evaluator_t<neighbor_at< 1,  1>>::evaluate(state).template equals_to<cell_state>();

        auto top_left       = vector_int_factory::from_condition_result<cell_row_type>(top_left_c);
        auto top            = vector_int_factory::from_condition_result<cell_row_type>(top_c);
        auto top_right      = vector_int_factory::from_condition_result<cell_row_type>(top_right_c);
        auto left           = vector_int_factory::from_condition_result<cell_row_type>(left_c);
        auto right          = vector_int_factory::from_condition_result<cell_row_type>(right_c);
        auto bottom_left    = vector_int_factory::from_condition_result<cell_row_type>(bottom_left_c);
        auto bottom         = vector_int_factory::from_condition_result<cell_row_type>(bottom_c);
        auto bottom_right   = vector_int_factory::from_condition_result<cell_row_type>(bottom_right_c);

        return top_left.template to_vector_with_bits<2>()
            .get_added(top)
            .get_added(top_right).template to_vector_with_bits<3>()
            .get_added(left)
            .get_added(right)
            .get_added(bottom_left)
            .get_added(bottom).template to_vector_with_bits<4>()
            .get_added(bottom_right);
    }
};

template <typename params, typename CellStateValue>
struct _evaluator_impl<
    params,
    count_neighbors<
        CellStateValue,
        moore_8_neighbors>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    CUDA_CALLABLE static vector_int<typename params::cell_row_t, 4> evaluate(state_t<params> state) {
        auto cell_state = evaluator_t<CellStateValue>::evaluate(state);

        auto top_left_c     = evaluator_t<neighbor_at<-1, -1>>::evaluate(state).equals_to(cell_state);
        auto top_c          = evaluator_t<neighbor_at< 0, -1>>::evaluate(state).equals_to(cell_state);
        auto top_right_c    = evaluator_t<neighbor_at< 1, -1>>::evaluate(state).equals_to(cell_state);
        auto left_c         = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).equals_to(cell_state);
        auto right_c        = evaluator_t<neighbor_at< 1,  0>>::evaluate(state).equals_to(cell_state);
        auto bottom_left_c  = evaluator_t<neighbor_at<-1,  1>>::evaluate(state).equals_to(cell_state);
        auto bottom_c       = evaluator_t<neighbor_at< 0,  1>>::evaluate(state).equals_to(cell_state);
        auto bottom_right_c = evaluator_t<neighbor_at< 1,  1>>::evaluate(state).equals_to(cell_state);

        auto top_left       = vector_int_factory::from_condition_result<cell_row_type>(top_left_c);
        auto top            = vector_int_factory::from_condition_result<cell_row_type>(top_c);
        auto top_right      = vector_int_factory::from_condition_result<cell_row_type>(top_right_c);
        auto left           = vector_int_factory::from_condition_result<cell_row_type>(left_c);
        auto right          = vector_int_factory::from_condition_result<cell_row_type>(right_c);
        auto bottom_left    = vector_int_factory::from_condition_result<cell_row_type>(bottom_left_c);
        auto bottom         = vector_int_factory::from_condition_result<cell_row_type>(bottom_c);
        auto bottom_right   = vector_int_factory::from_condition_result<cell_row_type>(bottom_right_c);

        return top_left.template to_vector_with_bits<2>()
            .get_added(top)
            .get_added(top_right).template to_vector_with_bits<3>()
            .get_added(left)
            .get_added(right)
            .get_added(bottom_left)
            .get_added(bottom).template to_vector_with_bits<4>()
            .get_added(bottom_right);
    }
};

template <typename params, typename CellStateValue>
struct _evaluator_impl<
    params,
    greater_than<
        count_neighbors<
            CellStateValue,
            moore_8_neighbors>,
        constant<0>
    >> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    CUDA_CALLABLE static auto evaluate(state_t<params> state) {
        auto cell_state = evaluator_t<CellStateValue>::evaluate(state);

        auto top_left_c     = evaluator_t<neighbor_at<-1, -1>>::evaluate(state).equals_to(cell_state);
        auto top_c          = evaluator_t<neighbor_at< 0, -1>>::evaluate(state).equals_to(cell_state);
        auto top_right_c    = evaluator_t<neighbor_at< 1, -1>>::evaluate(state).equals_to(cell_state);
        auto left_c         = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).equals_to(cell_state);
        auto right_c        = evaluator_t<neighbor_at< 1,  0>>::evaluate(state).equals_to(cell_state);
        auto bottom_left_c  = evaluator_t<neighbor_at<-1,  1>>::evaluate(state).equals_to(cell_state);
        auto bottom_c       = evaluator_t<neighbor_at< 0,  1>>::evaluate(state).equals_to(cell_state);
        auto bottom_right_c = evaluator_t<neighbor_at< 1,  1>>::evaluate(state).equals_to(cell_state);

        return top_left_c | top_c | top_right_c | left_c | right_c | bottom_left_c | bottom_c | bottom_right_c;
    }
};

template <typename params, typename cell_state_type, cell_state_type CellStateValue>
struct _evaluator_impl<
    params,
    count_neighbors<
        state_constant<CellStateValue>,
        von_neumann_4_neighbors>> {

    template <typename E>
    using evaluator_t = typename params::template evaluator_t<params, E>;

    using cell_row_type = typename params::cell_row_t;
    using state_dictionary_type = typename params::state_dict_t;

    CUDA_CALLABLE static vector_int<typename params::cell_row_t, 3> evaluate(state_t<params> state) {
        constexpr auto cell_state = state_dictionary_type::state_to_index(CellStateValue);

        // Get the four neighbors (top, right, bottom, left)
        auto top_c          = evaluator_t<neighbor_at< 0, -1>>::evaluate(state).template equals_to<cell_state>();
        auto right_c        = evaluator_t<neighbor_at< 1,  0>>::evaluate(state).template equals_to<cell_state>();
        auto bottom_c       = evaluator_t<neighbor_at< 0,  1>>::evaluate(state).template equals_to<cell_state>();
        auto left_c         = evaluator_t<neighbor_at<-1,  0>>::evaluate(state).template equals_to<cell_state>();

        // Convert condition results to vector_int
        auto top            = vector_int_factory::from_condition_result<cell_row_type>(top_c);
        auto right          = vector_int_factory::from_condition_result<cell_row_type>(right_c);
        auto bottom         = vector_int_factory::from_condition_result<cell_row_type>(bottom_c);
        auto left           = vector_int_factory::from_condition_result<cell_row_type>(left_c);

        return top.template to_vector_with_bits<2>()
            .get_added(right)
            .get_added(bottom).template to_vector_with_bits<3>()
            .get_added(left);
    }
};

} // namespace cellato::evaluators::bit_planes

#endif // CELLATO_EVALUATORS_BIT_PLANES_HPP