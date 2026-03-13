#ifndef MAZE_ALGORITHM_HPP
#define MAZE_ALGORITHM_HPP

#include "core/ast.hpp"

namespace maze {
using namespace cellato::ast;

enum class maze_cell_state {
    empty,
    wall
};

// Define constants for cell states
using empty = state_constant<maze_cell_state::empty>;
using wall = state_constant<maze_cell_state::wall>;

using c_2 = constant<2>;
using c_3 = constant<3>;
using c_6 = constant<6>;

using cell_is_wall = p<current_state, equals, wall>;

using wall_count = count_neighbors<wall, moore_8_neighbors>;
using cell_has_3_wall_neighbors = p<wall_count, equals, c_3>;

using has_less_then_6_wall_neighbors = p<wall_count, less_than, c_6>;

using maze_algorithm = if_< cell_has_3_wall_neighbors >::then_<
        wall
    >::elif_< p<cell_is_wall, and_, has_less_then_6_wall_neighbors> >::then_<
        wall
    >::else_<
        empty
    >;

}

#endif // MAZE_ALGORITHM_HPP
