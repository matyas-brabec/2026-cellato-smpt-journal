#ifndef CELLATO_UTILS_STATIC_DISPATCHER_HPP
#define CELLATO_UTILS_STATIC_DISPATCHER_HPP

#include <stdexcept>
#include <tuple>
#include <utility>
#include <type_traits>

namespace cellato::generic_dispatcher {
namespace detail {

// Helper to unpack the values from an integer_sequence into a generic lambda
template <typename T>
struct sequence_unpacker;

template <typename T, T... Values>
struct sequence_unpacker<std::integer_sequence<T, Values...>> {
    template <typename F>
    static void execute(F&& f) {
        f.template operator()<Values...>();
    }
};

// --- RECURSIVE DISPATCH IMPLEMENTATION ---

// Base case: All levels resolved
template <std::size_t Level, typename... IntegerSequences, typename Callable, typename... RuntimeArgs>
    requires(Level == sizeof...(IntegerSequences))
void dispatch_impl(
    const std::tuple<IntegerSequences...>&,
    Callable&& func,
    const std::tuple<RuntimeArgs...>&,
    auto... resolved_args
) {
    func.template operator()<resolved_args.value...>();
}

// Recursive step: Resolve one level
template <std::size_t Level = 0, typename... IntegerSequences, typename Callable, typename... RuntimeArgs, typename... ResolvedArgs>
    requires(Level < sizeof...(IntegerSequences))
void dispatch_impl(
    const std::tuple<IntegerSequences...>& seq_tuple,
    Callable&& func,
    const std::tuple<RuntimeArgs...>& runtime_args_tuple,
    // --- THIS IS THE FIX ---
    // The identifier 'resolved_args' is added back, with [[maybe_unused]] to silence the warning.
    [[maybe_unused]] ResolvedArgs... resolved_args
) {
    using CleanTupleType = std::remove_cvref_t<decltype(seq_tuple)>;
    using CurrentSequence = std::tuple_element_t<Level, CleanTupleType>;

    const auto current_runtime_val = std::get<Level>(runtime_args_tuple);
    using value_type = typename CurrentSequence::value_type;

    sequence_unpacker<CurrentSequence>::execute(
        [&]<auto... Options>() {
            bool match_found = false;

            ( (static_cast<value_type>(current_runtime_val) == static_cast<value_type>(Options) && (
                dispatch_impl<Level + 1>(
                    seq_tuple,
                    std::forward<Callable>(func),
                    runtime_args_tuple,
                    resolved_args...,
                    std::integral_constant<decltype(Options), Options>{}
                ),
                match_found = true
            )) || ... );

            if (!match_found) {
                throw std::runtime_error("A runtime parameter did not match any of its provided compile-time options.");
            }
        }
    );
}

} // namespace detail


template<typename... IntegerSequences, typename Callable, typename... Args>
void call(Callable&& func, Args&&... args) {
    static_assert(
        sizeof...(IntegerSequences) == sizeof...(Args),
        "The number of integer sequences must match the number of runtime arguments."
    );

    detail::dispatch_impl(
        std::tuple<IntegerSequences...>{},
        std::forward<Callable>(func),
        std::make_tuple(std::forward<Args>(args)...)
    );
}

} // namespace cellato::generic_dispatcher

#endif // CELLATO_UTILS_STATIC_DISPATCHER_HPP