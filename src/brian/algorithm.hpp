#ifndef BRIAN_ALGORITHM_HPP
#define BRIAN_ALGORITHM_HPP

#include "core/ast.hpp"

namespace brian {
using namespace cellato::ast;

enum class brian_cell_state {
    dead,
    dying,
    alive
};

// Define constants for cell states
using dead = state_constant<brian_cell_state::dead>;
using dying = state_constant<brian_cell_state::dying>;
using alive = state_constant<brian_cell_state::alive>;

using cell_is_alive = p<current_state, equals, alive>;
using cell_is_dying = p<current_state, equals, dying>;
using cell_is_dead = p<current_state, equals, dead>;

using c_2 = constant<2>;

// Count neighbors in Moore neighborhood
using alive_count = count_neighbors<alive, moore_8_neighbors>;

using has_two_alive_neighbors = p<alive_count, equals, c_2>;

using brian_algorithm = 
    if_< cell_is_dead >::then_<
        if_< has_two_alive_neighbors >::then_<
            alive
        >::else_<
            dead
        >
    >::
    elif_< cell_is_alive >::then_<
        dying
    >::
    else_< // cell_is_dying
        dead
    >;

// using brian_algorithm = neighbor_at<1, 1>;

}

#endif // GAME_OF_LIFE_ALGORITHM_HPP