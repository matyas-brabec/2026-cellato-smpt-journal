#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <array>
#include <type_traits>
#include <cstddef> 

namespace cellato::traversers::temporal_utils {

#if __CUDA_ARCH__ > 900
    // Blackwell Architecture (e.g., B100, B200) - CC 9.1+
    // Note: Assuming "B40" refers to the Blackwell family.
    // Each SM has 128 KB of combined L1/Shared Memory.
    // Max configurable shared memory is typically L1/SHM size minus a few KB for L1.
    constexpr std::size_t max_shm_size = 124 * 1024; // 124 KB

#elif __CUDA_ARCH__ == 900
    // Hopper Architecture (e.g., H100) - CC 9.0
    // Each SM has 256 KB of combined L1/Shared Memory.
    constexpr std::size_t max_shm_size = 228 * 1024; // 228 KB

#elif __CUDA_ARCH__ >= 800
    // Ampere Architecture (e.g., A100) - CC 8.0 to 8.9
    // Each SM has 192 KB of combined L1/Shared Memory.
    constexpr std::size_t max_shm_size = 164 * 1024; // 164 KB

#else
    // Fallback for older or unsupported architectures (e.g., Turing max is 64 KB)
    // A compile-time error might be more appropriate depending on your needs.
    // #error "Unsupported CUDA architecture."
    constexpr std::size_t max_shm_size = 48 * 1024; // 48 KB (a safe default)

#endif

template <typename TArray>
struct props {};

template <typename TPointer, std::size_t Size>
struct props<std::array<TPointer, Size>> {
    static constexpr std::size_t size = Size;
    using ptr_type = TPointer;
    using no_pointer_type = typename std::remove_pointer<TPointer>::type;
    using grid_type = std::array<TPointer, Size>;

    static constexpr std::size_t compute_buffer_size(std::size_t total_elements) {
        constexpr std::size_t bits_count = size;
        return (total_elements * bits_count);
    }

    static __device__ __host__ grid_type create_from_contiguous(TPointer data, std::size_t total_elements) {
        grid_type arr;
        for_each_bit([&]<std::size_t bit_idx>() {
            std::get<bit_idx>(arr) = data + bit_idx * total_elements;
        });
        return arr;
    }

    static __device__ __host__ void assign_to_from(
        grid_type to, std::size_t to_x_size,
        std::size_t to_x, std::size_t to_y,

        grid_type from, std::size_t from_x_size,
        std::size_t from_x, std::size_t from_y
    ) {

        for_each_bit([&]<std::size_t bit_idx>() {
            auto from_ptr = std::get<bit_idx>(from);
            auto to_ptr = std::get<bit_idx>(to);

            to_ptr[to_y * to_x_size + to_x] = from_ptr[from_y * from_x_size + from_x];
        });
    } 
  private:
    template <typename Callback, std::size_t... Is>
    static __device__ void for_each_bit_impl(Callback&& cb, std::index_sequence<Is...>) {
        (cb.template operator()<Is>(), ...);
    }

    template <typename Callback>
    static __device__ void for_each_bit(Callback&& cb) {
        for_each_bit_impl(std::forward<Callback>(cb), std::make_index_sequence<Size>{});
    }
};

} // namespace cellato::traversers::temporal_utils