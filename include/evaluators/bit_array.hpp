#ifndef CELLATO_BIT_ARRAY_EVALUATORS_HPP
#define CELLATO_BIT_ARRAY_EVALUATORS_HPP

#include "../core/ast.hpp"
#include "../memory/interface.hpp"
#include "../memory/idx_type.hpp"
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellato::evaluators::bit_array {

using idx_type = cellato::memory::idx_type;

using namespace cellato::ast;
using namespace cellato::memory;

template <typename grid_t>
using state_t = grids::point_in_grid<typename grid_t::cell_ptr_t>;

template <idx_type to>
struct static_for {
    template <typename Func>
    CUDA_CALLABLE static void apply(Func func) {
        if constexpr (to > 0) {
            func.template operator()<to - 1>();
            static_for<to - 1>::apply(func);
        }
    }
};

struct indexer {

    template <typename state_t>
    CUDA_CALLABLE static auto get_cell_at(state_t& state, idx_type x, idx_type y) {
        using grid_type = typename state_t::grid_t;
        constexpr static auto cells_per_word = grid_type::cells_per_word;
        idx_type x_size_original = state.properties.x_size * cells_per_word;

        idx_type x_wrapped = indexer::get_x(state, x);
        idx_type y_wrapped = indexer::get_y(state, y);

        return state.grid.get_individual_cell_at(y_wrapped * x_size_original + x_wrapped);
    }

    template <typename state_t>
    CUDA_CALLABLE static idx_type get_x(state_t& state, idx_type x) {
        using grid_type = typename state_t::grid_t;
        constexpr static auto cells_per_word = grid_type::cells_per_word;
        idx_type x_size_original = state.properties.x_size * cells_per_word;

        return (x + x_size_original) % x_size_original;
    }

    template <typename state_t>
    CUDA_CALLABLE static idx_type get_y(state_t& state, idx_type y) {
        idx_type y_size_original = state.properties.y_size;

        return (y + y_size_original) % y_size_original;
    }
};

// Implementation evaluator - processes a single subcell
template <typename grid_t, typename Expression, idx_type subcell_offset>
struct _impl_evaluator;

// Main evaluator - processes an entire word at once
template <typename bit_array_grid_t, typename Expression>
struct evaluator {
    using store_word_type = typename bit_array_grid_t::store_type;
    static constexpr idx_type cells_per_word = bit_array_grid_t::cells_per_word;

    CUDA_CALLABLE static store_word_type evaluate(state_t<bit_array_grid_t> state) {
        // Create a new word to store the results
        store_word_type result_word = 0;
        (void)state;
        
        // Iterate over each subcell and evaluate the expression
        static_for<cells_per_word>::apply([&]<idx_type subcell_idx>() {
            // Calculate and evaluate each subcell
            auto cell_result = _impl_evaluator<bit_array_grid_t, Expression, subcell_idx>::evaluate(state);
            // #ifndef __CUDACC__
            // std::cout << "Subcell " << subcell_idx << ": " << cell_result << std::endl;
            // #endif
            
            // Position this subcell in the result word
            result_word |= static_cast<store_word_type>(cell_result) << (subcell_idx * bit_array_grid_t::bits_per_cell);
        });

        // #ifndef __CUDACC__
        // exit(0);
        // #endif
        
        return result_word;
    }
};


using one_cell_int = std::int32_t;

// Implement specific expression evaluators below

// Constants
template <typename grid_t, auto Value, idx_type subcell_offset>
struct _impl_evaluator<grid_t, constant<Value>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> /* state */) {
        return Value;
    }
};

// State constants
template <typename grid_t, typename state_type, state_type Value, idx_type subcell_offset>
struct _impl_evaluator<grid_t, state_constant<Value>, subcell_offset> {
    using dictionary_t = typename grid_t::states_dict_t;

    CUDA_CALLABLE static one_cell_int evaluate(state_t<grid_t> /* state */) {
        constexpr one_cell_int index = static_cast<one_cell_int>(dictionary_t::state_to_index(Value));
        return index;
    }
};

// Conditional evaluation
template <typename grid_t, typename Condition, typename Then, typename Else, idx_type subcell_offset>
struct _impl_evaluator<grid_t, if_then_else<Condition, Then, Else>, subcell_offset> {

    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        if (_impl_evaluator<grid_t, Condition, subcell_offset>::evaluate(state)) {
            return _impl_evaluator<grid_t, Then, subcell_offset>::evaluate(state);
        } else {
            return _impl_evaluator<grid_t, Else, subcell_offset>::evaluate(state);
        }
    }
};

// Arithmetic operators
template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, bit_and_<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) &
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, plus<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) +
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, modulo<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) %
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, bit_or_<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) |
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};


// Logical operators

template <typename grid_t, typename Value, idx_type subcell_offset>
struct _impl_evaluator<grid_t, not_<Value>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return !_impl_evaluator<grid_t, Value, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, and_<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) && 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, or_<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) || 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

// Comparison operators
template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, equals<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) == 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, not_equals<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) != 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, greater_than<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) > 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Left, typename Right, idx_type subcell_offset>
struct _impl_evaluator<grid_t, less_than<Left, Right>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        return _impl_evaluator<grid_t, Left, subcell_offset>::evaluate(state) < 
               _impl_evaluator<grid_t, Right, subcell_offset>::evaluate(state);
    }
};

template <typename grid_t, typename Value, idx_type bit_idx, idx_type subcell_offset>
struct _impl_evaluator<grid_t, has_bit_set<Value, bit_idx>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        auto val = _impl_evaluator<grid_t, Value, subcell_offset>::evaluate(state);
        return (val & (1 << bit_idx)) != 0;
    }
};

// MISC

template <typename grid_t, idx_type subcell_offset, typename Even, typename Odd>
struct _impl_evaluator<grid_t, alternate_algorithms<Even, Odd>, subcell_offset> {

    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        if ((state.time_step % 2) == 0) {
            return _impl_evaluator<grid_t, Even, subcell_offset>::evaluate(state);
        } else {
            return _impl_evaluator<grid_t, Odd, subcell_offset>::evaluate(state);
        }
    }
};

// Neighborhood access
template <typename grid_t, idx_type x_offset, idx_type y_offset, idx_type subcell_offset>
struct _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset> {
    CUDA_CALLABLE static auto evaluate(state_t<grid_t> state) {
        constexpr auto cells_per_word = grid_t::cells_per_word;

        idx_type x = state.position.x * cells_per_word + x_offset + subcell_offset;
        idx_type y = state.position.y + y_offset;

        auto cell = indexer::get_cell_at(state, x, y);
        return static_cast<one_cell_int>(cell);
    }
};

// Neighbor counting
template <typename grid_t, typename CellStateValue, idx_type subcell_offset>
struct _impl_evaluator<grid_t, count_neighbors<CellStateValue, moore_8_neighbors>, subcell_offset> {

    template <idx_type x_offset, idx_type y_offset>
    using cell_at = _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset>;

    CUDA_CALLABLE static int evaluate(state_t<grid_t> state) {
        auto target_state = _impl_evaluator<grid_t, CellStateValue, subcell_offset>::evaluate(state);

        auto top_left_c     = cell_at<-1, -1>::evaluate(state) == target_state ? 1 : 0;
        auto top_c          = cell_at< 0, -1>::evaluate(state) == target_state ? 1 : 0;
        auto top_right_c    = cell_at< 1, -1>::evaluate(state) == target_state ? 1 : 0;
        auto left_c         = cell_at<-1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto right_c        = cell_at< 1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_left_c  = cell_at<-1,  1>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_c       = cell_at< 0,  1>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_right_c = cell_at< 1,  1>::evaluate(state) == target_state ? 1 : 0;

        return top_left_c + top_c + top_right_c +
               left_c + right_c +
               bottom_left_c + bottom_c + bottom_right_c;
    }
};

template <typename grid_t, typename CellStateValue, idx_type subcell_offset>
struct _impl_evaluator<grid_t, count_neighbors<CellStateValue, margolus_alternating_neighborhood>, subcell_offset> {

    CUDA_CALLABLE static int evaluate(state_t<grid_t> state) {
        auto target_value = _impl_evaluator<grid_t, CellStateValue, subcell_offset>::evaluate(state);

        constexpr auto cells_per_word = grid_t::cells_per_word;
        idx_type x_original = indexer::get_x(state, state.position.x * cells_per_word + subcell_offset);
        idx_type y_original = indexer::get_y(state, state.position.y);

        int parity = state.time_step % 2;
        int x_parity = x_original % 2;
        int y_parity = y_original % 2;

        int x_coords_0, x_coords_1, y_coords_0, y_coords_1;

        if (parity == 0) {
            if (x_parity == 0) {
                x_coords_0 = 0;
                x_coords_1 = 1;
            } else {
                x_coords_0 = -1;
                x_coords_1 = 0;
            }
            
            if (y_parity == 0) {
                y_coords_0 = 0;
                y_coords_1 = 1;
            } else {
                y_coords_0 = -1;
                y_coords_1 = 0;
            }
        } else {  // step_parity == 1
            if (x_parity == 0) {
                x_coords_0 = -1;
                x_coords_1 = 0;
            } else {
                x_coords_0 = 0;
                x_coords_1 = 1;
            }
            
            if (y_parity == 0) {
                y_coords_0 = -1;
                y_coords_1 = 0;
            } else {
                y_coords_0 = 0;
                y_coords_1 = 1;
            }
        }

        // Count cells with the target value in the Margolus neighborhood
        return (
            (get_cell_at(state, x_coords_0, y_coords_0) == target_value) +
            (get_cell_at(state, x_coords_0, y_coords_1) == target_value) +
            (get_cell_at(state, x_coords_1, y_coords_0) == target_value) +
            (get_cell_at(state, x_coords_1, y_coords_1) == target_value)
        );
    }

    
    template <idx_type x_offset, idx_type y_offset>
    using cell_at = _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset>;

    CUDA_CALLABLE static one_cell_int get_cell_at(state_t<grid_t> state, idx_type x_offset, idx_type y_offset) {
        if (x_offset == 0) {
            if (y_offset == 0) {
                return cell_at< 0,  0>::evaluate(state);
            } else if (y_offset == 1) {
                return cell_at< 0,  1>::evaluate(state);
            } else { // y_offset == -1
                return cell_at< 0, -1>::evaluate(state);
            }
        } else if (x_offset == 1) {
            if (y_offset == 0) {
                return cell_at< 1,  0>::evaluate(state);
            } else if (y_offset == 1) {
                return cell_at< 1,  1>::evaluate(state);
            } else { // y_offset == -1
                return cell_at< 1, -1>::evaluate(state);
            }
        } else { // x_offset == -1
            if (y_offset == 0) {
                return cell_at<-1,  0>::evaluate(state);
            } else if (y_offset == 1) {
                return cell_at<-1,  1>::evaluate(state);
            } else { // y_offset == -1
                return cell_at<-1, -1>::evaluate(state);
            }
        }
    }
};

template <typename grid_t, idx_type subcell_offset>
struct _impl_evaluator<grid_t, margolus_180_neighbor, subcell_offset> {

    CUDA_CALLABLE static one_cell_int evaluate(state_t<grid_t> state) {
        // Determine the absolute original coordinates of the specific sub-cell.
        constexpr auto cells_per_word = grid_t::cells_per_word;
        idx_type x_original = indexer::get_x(state, state.position.x * cells_per_word + subcell_offset);
        idx_type y_original = indexer::get_y(state, state.position.y);

        // Calculate parities to determine the 2x2 block for this sub-cell.
        int parity = state.time_step % 2;
        int x_parity = x_original % 2;
        int y_parity = y_original % 2;

        int x_coords_0, x_coords_1, y_coords_0, y_coords_1;

        if (parity == 0) {
            if (x_parity == 0) { x_coords_0 = 0; x_coords_1 = 1; } 
            else { x_coords_0 = -1; x_coords_1 = 0; }
            
            if (y_parity == 0) { y_coords_0 = 0; y_coords_1 = 1; } 
            else { y_coords_0 = -1; y_coords_1 = 0; }
        } else {  // parity == 1
            if (x_parity == 0) { x_coords_0 = -1; x_coords_1 = 0; } 
            else { x_coords_0 = 0; x_coords_1 = 1; }
            
            if (y_parity == 0) { y_coords_0 = -1; y_coords_1 = 0; } 
            else { y_coords_0 = 0; y_coords_1 = 1; }
        }

        // The offset to the diagonal neighbor is the sum of the coordinate pairs.
        int dx_opposite = x_coords_0 + x_coords_1;
        int dy_opposite = y_coords_0 + y_coords_1;

        // Fetch and return the state of the diagonally opposite cell using the helper.
        return get_cell_at(state, dx_opposite, dy_opposite);
    }

    // Helper to get the state of a cell at a relative offset.
    // This is identical to the one in your example to ensure consistent cell access.
    template <idx_type x_offset, idx_type y_offset>
    using cell_at = _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset>;

    CUDA_CALLABLE static one_cell_int get_cell_at(state_t<grid_t> state, idx_type x_offset, idx_type y_offset) {
        if (x_offset == 0) {
            if (y_offset == 0) {
                return cell_at< 0,  0>::evaluate(state);
            } else if (y_offset == 1) {
                return cell_at< 0,  1>::evaluate(state);
            } else { // y_offset == -1
                return cell_at< 0, -1>::evaluate(state);
            }
        } else if (x_offset == 1) {
            if (y_offset == 0) {
                return cell_at< 1,  0>::evaluate(state);
            } else if (y_offset == 1) {
                return cell_at< 1,  1>::evaluate(state);
            } else { // y_offset == -1
                return cell_at< 1, -1>::evaluate(state);
            }
        } else { // x_offset == -1
            if (y_offset == 0) {
                return cell_at<-1,  0>::evaluate(state);
            } else if (y_offset == 1) {
                return cell_at<-1,  1>::evaluate(state);
            } else { // y_offset == -1
                return cell_at<-1, -1>::evaluate(state);
            }
        }
    }
};

template <typename grid_t, typename CellStateValue, idx_type subcell_offset>
struct _impl_evaluator<grid_t, count_neighbors<CellStateValue, von_neumann_4_neighbors>, subcell_offset> {
    
    template <idx_type x_offset, idx_type y_offset>
    using cell_at = _impl_evaluator<grid_t, neighbor_at<x_offset, y_offset>, subcell_offset>;

    CUDA_CALLABLE static int evaluate(state_t<grid_t> state) {
        auto target_state = _impl_evaluator<grid_t, CellStateValue, subcell_offset>::evaluate(state);

        auto top_c          = cell_at< 0, -1>::evaluate(state) == target_state ? 1 : 0;
        auto left_c         = cell_at<-1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto right_c        = cell_at< 1,  0>::evaluate(state) == target_state ? 1 : 0;
        auto bottom_c       = cell_at< 0,  1>::evaluate(state) == target_state ? 1 : 0;

        return top_c + left_c + right_c + bottom_c;
    }
};

} // namespace cellato::evaluators::bit_array

#endif // CELLATO_BIT_ARRAY_EVALUATORS_HPP