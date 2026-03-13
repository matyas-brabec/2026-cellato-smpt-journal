#ifndef FLUID_ALGORITHM_HPP
#define FLUID_ALGORITHM_HPP

#include "core/ast.hpp"

namespace fluid {
using namespace cellato::ast;

using fluid_cell_state = int;

constexpr int TOP_bit = 0, TOP = 1 << TOP_bit;
constexpr int BOTTOM_bit = 1, BOTTOM = 1 << BOTTOM_bit;
constexpr int LEFT_bit = 2, LEFT = 1 << LEFT_bit;
constexpr int RIGHT_bit = 3, RIGHT = 1 << RIGHT_bit;

using incoming_from_top = has_bit_set<neighbor_at<0, -1>, TOP_bit>;
using incoming_from_bottom = has_bit_set<neighbor_at<0, 1>, BOTTOM_bit>;
using incoming_from_left = has_bit_set<neighbor_at<-1, 0>, LEFT_bit>;
using incoming_from_right = has_bit_set<neighbor_at<1, 0>, RIGHT_bit>;

using vertical_collision = p<incoming_from_top, bit_and_, incoming_from_bottom>;
using horizontal_collision = p<incoming_from_left, bit_and_, incoming_from_right>;

using combined_vertical_incoming = p<
        p<neighbor_at<0, -1>, bit_and_, constant<TOP>>,
        bit_or_,
        p<neighbor_at<0, 1>, bit_and_, constant<BOTTOM>>
    >;

using combined_horizontal_incoming = p<
        p<neighbor_at<-1, 0>, bit_and_, constant<LEFT>>,
        bit_or_,
        p<neighbor_at<1, 0>, bit_and_, constant<RIGHT>>
    >;

using just_vertical_collision = p<
        vertical_collision,
        bit_and_,
        p<combined_horizontal_incoming, equals, constant<0>>
    >;

using just_horizontal_collision = p<
        horizontal_collision,
        bit_and_,
        p<combined_vertical_incoming, equals, constant<0>>
    >;

using vertical_result = if_< just_vertical_collision >::then_<
        state_constant<LEFT | RIGHT>
    >::else_<
        combined_vertical_incoming
    >;

using horizontal_result = if_< just_horizontal_collision >::then_<
        state_constant<TOP | BOTTOM>
    >::else_<
        combined_horizontal_incoming
    >;

using fluid_algorithm = p< vertical_result, bit_or_, horizontal_result >;

} // namespace fluid

#endif // FLUID_ALGORITHM_HPP
