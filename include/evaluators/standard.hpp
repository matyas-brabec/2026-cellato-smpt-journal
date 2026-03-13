#ifndef CELLATO_STANDARD_EVALUATORS_HPP
#define CELLATO_STANDARD_EVALUATORS_HPP

#include <cstddef>

#include <utility>

#include "../core/ast.hpp"
#include "../memory/interface.hpp"
#include "../memory/idx_type.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellato::evaluators::standard {

using idx_type = cellato::memory::idx_type;

using namespace cellato::ast;
using namespace cellato::memory;

template <typename cell_type, typename Expression, typename cell_ptr_type = cell_type*>
struct evaluator;

template <typename cell_type, typename cell_ptr_type = cell_type*>
using state_t = grids::point_in_grid<cell_ptr_type>;

template <typename cell_type, typename cell_ptr_type, auto Value>
struct evaluator<cell_type, constant<Value>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> /* state */) {
        return Value;
    }
};

template <typename cell_type, typename cell_ptr_type, auto Value>
struct evaluator<cell_type, state_constant<Value>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> /* state */) {
        return Value;
    }
};

template <typename cell_type, typename cell_ptr_type, typename Condition, typename Then, typename Else>
struct evaluator<cell_type, if_then_else<Condition, Then, Else>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return (evaluator<cell_type, Condition, cell_ptr_type>::evaluate(state))
            ? evaluator<cell_type, Then, cell_ptr_type>::evaluate(state)
            : evaluator<cell_type, Else, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, bit_and_<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) &
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, plus<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) +
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, modulo<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) %
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, bit_or_<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) |
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};


template <typename cell_type, typename cell_ptr_type, typename Value>
struct evaluator<cell_type, not_<Value>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return !evaluator<cell_type, Value, cell_ptr_type>::evaluate(state);
    }
};


template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, and_<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) && 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, or_<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) || 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, equals<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) == 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, greater_than<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) > 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, less_than<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) < 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, typename Left, typename Right>
struct evaluator<cell_type, not_equals<Left, Right>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return evaluator<cell_type, Left, cell_ptr_type>::evaluate(state) != 
               evaluator<cell_type, Right, cell_ptr_type>::evaluate(state);
    }
};

template <typename cell_type, typename cell_ptr_type, idx_type x_offset, idx_type y_offset>
struct evaluator<cell_type, neighbor_at<x_offset, y_offset>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        return state.grid[state.idx(
            state.position.x + x_offset,
            state.position.y + y_offset)];
    }
};

template <typename cell_type, typename cell_ptr_type, typename Value, int bit_idx>
struct evaluator<cell_type, has_bit_set<Value, bit_idx>, cell_ptr_type> {
    CUDA_CALLABLE static auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto val = evaluator<cell_type, Value, cell_ptr_type>::evaluate(state);
        return (val & ((decltype(val))1 << bit_idx)) != 0;
    }
};

template <typename cell_type, typename cell_ptr_type, typename Even, typename Odd>
struct evaluator<cell_type, alternate_algorithms<Even, Odd>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        if (state.time_step % 2 == 0) {
            return evaluator<cell_type, Even, cell_ptr_type>::evaluate(state);
        } else {
            return evaluator<cell_type, Odd, cell_ptr_type>::evaluate(state);
        }
    }
};

template <typename cell_type, typename cell_ptr_type, typename CellStateValue>
struct evaluator<cell_type, count_neighbors<CellStateValue, moore_8_neighbors>, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto target_value = evaluator<cell_type, CellStateValue, cell_ptr_type>::evaluate(state);

        return [target_value, state, x = state.position.x, y = state.position.y, x_size = state.properties.x_size]<std::size_t... I> (std::index_sequence<I...>) {
            constexpr int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
            constexpr int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

            return (... + (state.grid[state.idx(x + dx[I], y + dy[I])] == target_value));
        }(std::make_index_sequence<8>{});
    }
};

template <typename cell_type, typename cell_ptr_type, auto CellStateValue>
struct evaluator<cell_type, count_neighbors<
    state_constant<CellStateValue>, margolus_alternating_neighborhood>,
    cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {

        auto parity = state.time_step % 2;
        auto x_parity = (state.position.x) % 2;
        auto y_parity = (state.position.y) % 2;

        int x_coords[2], y_coords[2];

        if (parity == 0) {
            if (x_parity == 0) { x_coords[0] = 0; x_coords[1] = 1; } 
            else { x_coords[0] = -1; x_coords[1] = 0; }
            if (y_parity == 0) { y_coords[0] = 0; y_coords[1] = 1; }
            else { y_coords[0] = -1; y_coords[1] = 0; }
        } else {  // parity == 1
            if (x_parity == 0) { x_coords[0] = -1; x_coords[1] = 0; } 
            else { x_coords[0] = 0; x_coords[1] = 1; }
            if (y_parity == 0) { y_coords[0] = -1; y_coords[1] = 0; }
            else { y_coords[0] = 0; y_coords[1] = 1; }
        }

        return (
            (state.grid[state.idx(state.position.x + x_coords[0], state.position.y + y_coords[0])] == CellStateValue) +
            (state.grid[state.idx(state.position.x + x_coords[0], state.position.y + y_coords[1])] == CellStateValue) +
            (state.grid[state.idx(state.position.x + x_coords[1], state.position.y + y_coords[0])] == CellStateValue) +
            (state.grid[state.idx(state.position.x + x_coords[1], state.position.y + y_coords[1])] == CellStateValue)
        );
    }
};

template <typename cell_type, typename cell_ptr_type>
struct evaluator<cell_type, margolus_180_neighbor, cell_ptr_type> {
    static CUDA_CALLABLE auto evaluate(state_t<cell_type, cell_ptr_type> state) {
        // Determine the 2x2 block this cell belongs to.
        auto parity = state.time_step % 2;
        auto x_parity = state.position.x % 2;
        auto y_parity = state.position.y % 2;

        int dx_opposite, dy_opposite;

        if (parity == 0) {
            if (x_parity == 0) { dx_opposite = 1; } 
            else { dx_opposite = -1; }
            
            if (y_parity == 0) { dy_opposite = 1; } 
            else { dy_opposite = -1; }
        } else { // parity == 1
            if (x_parity == 0) { dx_opposite = -1; } 
            else { dx_opposite = 1; }
            
            if (y_parity == 0) { dy_opposite = -1; } 
            else { dy_opposite = 1; }
        }

        // Get and return the state of the diagonally opposite cell.
        return state.grid[state.idx(
            state.position.x + dx_opposite, 
            state.position.y + dy_opposite
        )];
    }
};

template <typename cell_type, typename cell_ptr_type, typename CellStateValue>
struct evaluator<cell_type, count_neighbors<CellStateValue, von_neumann_4_neighbors>, cell_ptr_type> {
    static CUDA_CALLABLE int evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto target_value = evaluator<cell_type, CellStateValue, cell_ptr_type>::evaluate(state);

        return [target_value, state, x = state.position.x, y = state.position.y, x_size = state.properties.x_size]<std::size_t... I> (std::index_sequence<I...>) {
            constexpr int dx[] = {0, 0, 1, -1};
            constexpr int dy[] = {1, -1, 0, 0};

            return (... + (state.grid[state.idx(x + dx[I], y + dy[I])] == target_value));
        }(std::make_index_sequence<4>{});
    }
};

// example of nested specialization for count_neighbors with moore_8_neighbors for better performance

template <typename cell_type, typename cell_ptr_type, typename CellStateValue>
struct evaluator<cell_type, 
        greater_than< 
            count_neighbors<CellStateValue, moore_8_neighbors>, 
            state_constant<0>
        >, cell_ptr_type> {
    static CUDA_CALLABLE int evaluate(state_t<cell_type, cell_ptr_type> state) {
        auto target_value = evaluator<cell_type, CellStateValue, cell_ptr_type>::evaluate(state);

        return
            state.grid[state.idx(state.position.x - 1, state.position.y - 1)] == target_value ||
            state.grid[state.idx(state.position.x - 1, state.position.y    )] == target_value ||
            state.grid[state.idx(state.position.x - 1, state.position.y + 1)] == target_value ||
            state.grid[state.idx(state.position.x    , state.position.y - 1)] == target_value ||
            state.grid[state.idx(state.position.x    , state.position.y + 1)] == target_value ||
            state.grid[state.idx(state.position.x + 1, state.position.y - 1)] == target_value ||
            state.grid[state.idx(state.position.x + 1, state.position.y    )] == target_value ||
            state.grid[state.idx(state.position.x + 1, state.position.y + 1)] == target_value;
    }
};

} // namespace cellato::evaluators::standard

#endif // CELLATO_STANDARD_EVALUATORS_HPP