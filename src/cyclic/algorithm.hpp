#ifndef CYCLIC_ALGORITHM_HPP
#define CYCLIC_ALGORITHM_HPP

#include "core/ast.hpp"

namespace cyclic {
using namespace cellato::ast;

using cyclic_cell_state = int;

constexpr int BITS = 5;
constexpr int STATES = 1 << BITS;

using one_bigger_then_current_absolute = p<current_state, plus, constant<1>>;
using one_bigger_then_current = p<one_bigger_then_current_absolute, modulo, constant<STATES>>;

using one_bigger_count = count_neighbors<one_bigger_then_current, moore_8_neighbors>;

using cyclic_algorithm = if_< p<one_bigger_count, greater_than, constant<0>> >::then_<
        one_bigger_then_current
    >::else_<
        current_state
    >;

}

#endif // CYCLIC_ALGORITHM_HPP
