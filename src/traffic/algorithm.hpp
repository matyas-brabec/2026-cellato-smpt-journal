#ifndef TRAFFIC_ALGORITHM_HPP
#define TRAFFIC_ALGORITHM_HPP

#include "core/ast.hpp"

namespace traffic {
using namespace cellato::ast;

enum class traffic_cell_state {
    empty, red_car, blue_car
};

using empty = state_constant<traffic_cell_state::empty>;

template <typename movable_car, typename stationary_car, typename incoming_neighbor, typename outgoing_neighbor>
struct one_direction {

    using is_movable_car = p<current_state, equals, movable_car>;
    using is_empty = p<current_state, equals, empty>;

    using outgoing_is_empty = p<outgoing_neighbor, equals, empty>;
    using incoming_is_movable_car = p<incoming_neighbor, equals, movable_car>;

    using move_to_outgoing_if_possible =
        if_< outgoing_is_empty >::template then_<
            empty
        >::template else_<
            movable_car
        >;

    using move_incoming_neighbor_if_possible =
        if_< incoming_is_movable_car >::template then_<
            movable_car
        >::template else_<
            empty
        >;

    using algorithm = 
        if_< is_movable_car >::template then_<
            move_to_outgoing_if_possible
        
        >::template elif_< is_empty >::template then_<
            move_incoming_neighbor_if_possible

        >::template else_< // is_stationary_car
            stationary_car // do not move
        >;
};

using red_car = state_constant<traffic_cell_state::red_car>;
using blue_car = state_constant<traffic_cell_state::blue_car>;

using up = neighbor_at<0, -1>;
using down = neighbor_at<0, 1>;
using left = neighbor_at<-1, 0>;
using right = neighbor_at<1, 0>;

using traffic_algorithm = alternate_algorithms<
    one_direction<red_car, blue_car, left, right>::algorithm,
    one_direction<blue_car, red_car, up, down>::algorithm
>;

}

#endif // TRAFFIC_ALGORITHM_HPP
