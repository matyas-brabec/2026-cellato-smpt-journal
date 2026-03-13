#ifndef CELLATO_TESTS_BIT_PLANES_GRID_HPP
#define CELLATO_TESTS_BIT_PLANES_GRID_HPP

#include "manager.hpp"
#include "../memory/bit_planes_grid.hpp"
#include "../memory/tiled_bit_planes_grid.hpp"
#include "../memory/grid_utils.hpp"
#include "../memory/state_dictionary.hpp"
#include "../memory/interface.hpp"
#include <cstdint>
#include <random>
#include <ctime>

namespace cellato::tests {
// Create a nested namespace for bit_planes tests to avoid conflicts
namespace bit_planes {

// Define an enum for testing
enum class TestCellState {
    DEAD,
    ALIVE,
    DYING
};


template <typename tested_grid>
struct grid_params {};

template <typename store_word_type, typename states_dict_t, cellato::memory::grids::device device_type>
struct grid_params<cellato::memory::grids::tiled_bit_planes::grid<store_word_type, states_dict_t, device_type>> {
    using grid_type = cellato::memory::grids::tiled_bit_planes::grid<store_word_type, states_dict_t, device_type>;

    static constexpr auto x_bit_width = 8;
    static constexpr auto y_bit_height = (sizeof(typename grid_type::store_type) * 8) / x_bit_width;

};

template <typename store_word_type, typename states_dict_t, cellato::memory::grids::device device_type>
struct grid_params<cellato::memory::grids::bit_planes::grid<store_word_type, states_dict_t, device_type>> {
    using grid_type = cellato::memory::grids::bit_planes::grid<store_word_type, states_dict_t, device_type>;

    static constexpr auto x_bit_width = (sizeof(typename grid_type::store_type) * 8);
    static constexpr auto y_bit_height = 1;
};

// Stream operator for TestCellState to help with test output
inline std::string to_string(const TestCellState& state) {
    switch (state) {
        case TestCellState::DEAD: return "DEAD";
        case TestCellState::ALIVE: return "ALIVE";
        case TestCellState::DYING: return "DYING";
        default: return "UNKNOWN";
    }
}

// Define the state dictionary for testing
using TestStateDictionary = cellato::memory::grids::state_dictionary<
    TestCellState::DEAD, 
    TestCellState::ALIVE, 
    TestCellState::DYING
>;

} // namespace bit_planes

template <template <typename, typename> class tested_grid_template, typename store_type, char const* test_tag>
class bit_planes_grid_test_suite : public test_suite {
public:

    using tested_grid = tested_grid_template<store_type, bit_planes::TestStateDictionary>;

    std::string name() const override {
        return std::string(test_tag);
    }

    test_result run() override {
        test_result result;
        test_case tc(result, true);

        // Run all bit_planes_grid tests
        test_state_dictionary_basics(tc);
        test_state_dictionary_conversion(tc);
        test_bit_grid_sizes(tc);
        test_bit_grid_pattern(tc);
        test_bit_grid_pattern_bigger_grid(tc);
        test_bit_grid_complex(tc);
        test_get_cell(tc);
        test_to_original_representation_with_get_cell(tc);
        test_bit_grid_small_random(tc);  // Smaller scale for unit tests
        
        return result;
    }

private:

    const static auto bits = sizeof(store_type) * 8;

    // Test state_dictionary basic properties
    void test_state_dictionary_basics(test_case& tc) {
        std::cout << BLUE << "\n--- Testing state_dictionary basics ---" << RESET << std::endl;
        
        tc.assert_equal(3, bit_planes::TestStateDictionary::number_of_values, "Dictionary should have 3 values");
        tc.assert_equal(2, bit_planes::TestStateDictionary::needed_bits, "Should need 2 bits to represent 3 states");
    }

    // Test state_dictionary conversion functions
    void test_state_dictionary_conversion(test_case& tc) {
        std::cout << BLUE << "\n--- Testing state_dictionary conversion ---" << RESET << std::endl;
        
        tc.assert_equal(0, bit_planes::TestStateDictionary::state_to_index(bit_planes::TestCellState::DEAD), "DEAD should map to index 0");
        tc.assert_equal(1, bit_planes::TestStateDictionary::state_to_index(bit_planes::TestCellState::ALIVE), "ALIVE should map to index 1");
        tc.assert_equal(2, bit_planes::TestStateDictionary::state_to_index(bit_planes::TestCellState::DYING), "DYING should map to index 2");
        
        tc.assert_true(bit_planes::TestCellState::DEAD == bit_planes::TestStateDictionary::index_to_state(0), "Index 0 should map to DEAD");
        tc.assert_true(bit_planes::TestCellState::ALIVE == bit_planes::TestStateDictionary::index_to_state(1), "Index 1 should map to ALIVE");
        tc.assert_true(bit_planes::TestCellState::DYING == bit_planes::TestStateDictionary::index_to_state(2), "Index 2 should map to DYING");
        
        bool exception_thrown = false;
        try {
            bit_planes::TestStateDictionary::state_to_index(static_cast<bit_planes::TestCellState>(99));
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for invalid state");
        
        exception_thrown = false;
        try {
            bit_planes::TestStateDictionary::index_to_state(99);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for invalid index");
    }

    // Test bit_planes_grid construction and size methods
    void test_bit_grid_sizes(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid sizes (" << test_tag << ") ---" << RESET << std::endl;

        // Create a 2x3 grid (2 rows, 3 word columns)
        const size_t height_words = 2;
        const size_t width_words = 3;
        const size_t width = bit_planes::grid_params<tested_grid>::x_bit_width * width_words;
        const size_t height = bit_planes::grid_params<tested_grid>::y_bit_height * height_words;

        // Initialize grid with DEAD cells
        std::vector<bit_planes::TestCellState> input_grid(height * width, bit_planes::TestCellState::DEAD);
        
        tested_grid grid(height, width, input_grid.data());

        tc.assert_equal(width, grid.x_size_original(), "Grid width should match original width");
        tc.assert_equal(height, grid.y_size_original(), "Grid height should match original height");
        tc.assert_equal(width_words, grid.x_size_physical(), "Physical width should be original width / bits_per_word");
        tc.assert_equal(height_words, grid.y_size_physical(), "Physical height should match original height");
    }

    // Test bit_planes_grid storage and retrieval of patterns
    void test_bit_grid_pattern(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid pattern storage and retrieval (" << test_tag << ") ---" << RESET << std::endl;

        if (bits != 8) {
            std::cout << YELLOW << "SKIPPING: test is designed for 8 bits" << RESET << std::endl;
            return;
        }

        // Create a 1x1 grid (1 row, 1 word)
        // For store_type, this stores 8 cells in a row
        std::vector<bit_planes::TestCellState> input_grid = {
            bit_planes::TestCellState::DEAD, bit_planes::TestCellState::ALIVE, bit_planes::TestCellState::DYING, bit_planes::TestCellState::DEAD,
            bit_planes::TestCellState::ALIVE, bit_planes::TestCellState::DEAD, bit_planes::TestCellState::ALIVE, bit_planes::TestCellState::DYING
        };

        tested_grid grid(1, 8, input_grid.data());
        
        // Reconstruct and verify
        auto result = grid.to_original_representation();
        
        tc.assert_equal(size_t{8}, result.size(), "Result should have 8 cells");
        
        // Check each cell matches what we put in
        for (size_t i = 0; i < 8; ++i) {
            tc.assert_true(input_grid[i] == result[i], "Cell " + std::to_string(i) + " should match input");
        }
    }

        // Test bit_planes_grid storage and retrieval of patterns
    void test_bit_grid_pattern_bigger_grid(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid pattern storage and retrieval (bigger) (" << test_tag << ") ---" << RESET << std::endl;

        constexpr auto d = bit_planes::TestCellState::DEAD;
        constexpr auto a = bit_planes::TestCellState::ALIVE;
        constexpr auto x = bit_planes::TestCellState::DYING;

        constexpr size_t x_size = 32;
        constexpr size_t y_size = 16;

        // Create a 3x24 grid (3 rows, 24 word columns)
        std::vector<bit_planes::TestCellState> input_grid = {
            // row 0
            d, a, d, x, a, x, x, a, 
            d, a, d, d, d, a, d, a, 
            d, a, x, a, a, x, d, d, 
            a, d, a, d, x, d, x, d, 
            // row 1
            x, x, x, a, d, d, a, a, 
            a, a, d, a, a, d, d, a, 
            a, x, a, d, a, a, d, d, 
            x, a, d, d, d, a, x, d, 
            // row 2
            x, d, d, x, a, x, a, x, 
            a, x, x, d, a, d, a, x, 
            x, d, a, d, a, d, x, x, 
            d, x, a, x, a, a, x, x, 
            // row 3
            x, a, a, a, x, d, x, d, 
            d, d, a, x, a, d, a, d, 
            x, d, x, x, d, x, a, d, 
            a, d, a, x, a, a, a, a, 
            // row 4
            x, d, x, x, d, x, d, x, 
            a, d, d, x, d, d, d, a, 
            x, a, d, d, x, a, a, a, 
            a, a, a, a, x, a, a, d, 
            // row 5
            d, d, d, x, x, x, x, a, 
            d, d, a, d, d, a, a, d, 
            a, x, a, a, a, x, a, d, 
            a, a, a, a, d, d, a, x, 
            // row 6
            x, a, x, a, x, d, a, x, 
            a, a, d, a, d, d, d, d, 
            x, a, a, x, a, x, x, d, 
            d, x, d, a, x, d, d, a, 
            // row 7
            d, a, d, x, x, d, a, d, 
            x, x, a, d, a, x, d, a, 
            a, x, x, d, d, x, d, x, 
            a, a, x, x, a, a, a, a, 
            // row 8
            d, d, d, x, d, d, d, a, 
            a, x, x, a, a, a, d, a, 
            a, a, x, x, d, a, d, a, 
            a, d, a, d, x, d, a, d, 
            // row 9
            x, a, d, d, d, x, a, x, 
            d, d, a, x, a, x, x, x, 
            x, x, a, a, d, d, d, x, 
            d, x, x, a, x, d, d, d, 
            // row 10
            a, x, a, a, x, x, d, d, 
            x, a, x, d, x, d, a, d, 
            x, a, d, x, x, a, a, x, 
            x, x, d, d, a, d, a, d, 
            // row 11
            x, d, d, d, d, x, a, x, 
            d, a, a, d, a, a, d, a, 
            a, a, x, a, d, d, a, x, 
            x, d, d, x, a, x, a, a, 
            // row 12
            x, x, a, d, x, x, x, d, 
            a, d, d, d, x, d, d, d, 
            a, d, d, x, d, x, a, a, 
            a, d, d, a, x, d, x, a, 
            // row 13
            d, a, a, x, d, x, a, x, 
            x, x, x, d, a, x, a, x, 
            d, x, x, a, a, x, x, d, 
            x, d, x, x, a, a, x, a, 
            // row 14
            a, a, d, x, a, d, a, d, 
            d, d, x, x, a, a, a, x, 
            x, d, x, a, a, d, a, a, 
            d, x, d, x, d, d, x, a, 
            // row 15
            x, d, x, x, x, x, d, a, 
            x, x, d, x, x, d, d, x, 
            a, x, a, a, a, a, d, a, 
            d, a, a, a, x, x, a, x, 
        };

        tested_grid grid(y_size, x_size, input_grid.data());

        // Reconstruct and verify
        auto result = grid.to_original_representation();
        
        tc.assert_equal(
            size_t{x_size * y_size}, result.size(), "Result should have 72 cells");

        // Check each cell matches what we put in
        for (size_t i = 0; i < x_size * y_size; ++i) {
            tc.silent_if_passed().assert_true(input_grid[i] == result[i], "Cell " + std::to_string(i) + " should match input");
        }
    }

    // Test bit_planes_grid with larger grid and complex pattern
    void test_bit_grid_complex(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid with complex pattern (" << test_tag << ") ---" << RESET << std::endl;

        if (bits != 8) {
            std::cout << YELLOW << "SKIPPING: test is designed for 8 bits" << RESET << std::endl;
            return;
        }

        // Create a 2x2 grid (2 rows, 2 word columns) for 2x16 cells with store_type
        const size_t height = 2;
        const size_t width_words = 2;
        const size_t word_bits = sizeof(store_type) * 8;
        const size_t width = width_words * word_bits;
        std::vector<bit_planes::TestCellState> input_grid(height * width, bit_planes::TestCellState::DEAD);
        
        // Set specific cells to create a pattern
        // Row 0, positions 0, 3, 7 are ALIVE
        // Row 1, positions 1, 4, 9 are DYING
        input_grid[0] = bit_planes::TestCellState::ALIVE;
        input_grid[3] = bit_planes::TestCellState::ALIVE;
        input_grid[7] = bit_planes::TestCellState::ALIVE;
        input_grid[width + 1] = bit_planes::TestCellState::DYING;
        input_grid[width + 4] = bit_planes::TestCellState::DYING;
        input_grid[width + 9] = bit_planes::TestCellState::DYING;
        
        tested_grid grid(height, width, input_grid.data());
        
        // Verify the reconstruction
        auto result = grid.to_original_representation();
        
        tc.assert_true(bit_planes::TestCellState::ALIVE == result[0], "Cell (0,0) should be ALIVE");
        tc.assert_true(bit_planes::TestCellState::ALIVE == result[3], "Cell (0,3) should be ALIVE");
        tc.assert_true(bit_planes::TestCellState::ALIVE == result[7], "Cell (0,7) should be ALIVE");
        tc.assert_true(bit_planes::TestCellState::DYING == result[width + 1], "Cell (1,1) should be DYING");
        tc.assert_true(bit_planes::TestCellState::DYING == result[width + 4], "Cell (1,4) should be DYING");
        tc.assert_true(bit_planes::TestCellState::DYING == result[width + 9], "Cell (1,9) should be DYING");
    }

    // Test get_cell function
    void test_get_cell(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid get_cell function (" << test_tag << ") ---" << RESET << std::endl;

        if (bits != 8) {
            std::cout << YELLOW << "SKIPPING: test is designed for 8 bits" << RESET << std::endl;
            return;
        }

        // Create a 2x2 grid (2 rows, 2 word columns) with a specific pattern
        const size_t height = 2;
        const size_t width_words = 2;
        const size_t word_bits = sizeof(store_type) * 8;
        const size_t width = width_words * word_bits;
        std::vector<bit_planes::TestCellState> input_grid(height * width, bit_planes::TestCellState::DEAD);
        
        // Set specific cells based on their (x, y) coordinates
        // Row 0
        input_grid[0] = bit_planes::TestCellState::ALIVE;                 // (0,0)
        input_grid[3] = bit_planes::TestCellState::DYING;                 // (3,0)
        
        // Row 1 - offset by width
        input_grid[width + 1] = bit_planes::TestCellState::ALIVE;         // (1,1)
        input_grid[width + 7] = bit_planes::TestCellState::DYING;         // (7,1)
        input_grid[width + 9] = bit_planes::TestCellState::ALIVE;         // (9,1)
        
        tested_grid grid(height, width, input_grid.data());
        
        // Test specific cell retrievals
        tc.assert_true(bit_planes::TestCellState::ALIVE == grid.get_cell(0, 0), "Cell (0,0) should be ALIVE");
        tc.assert_true(bit_planes::TestCellState::DEAD == grid.get_cell(1, 0), "Cell (1,0) should be DEAD");
        tc.assert_true(bit_planes::TestCellState::DYING == grid.get_cell(3, 0), "Cell (3,0) should be DYING");
        tc.assert_true(bit_planes::TestCellState::ALIVE == grid.get_cell(1, 1), "Cell (1,1) should be ALIVE");
        tc.assert_true(bit_planes::TestCellState::DYING == grid.get_cell(7, 1), "Cell (7,1) should be DYING");
        tc.assert_true(bit_planes::TestCellState::ALIVE == grid.get_cell(9, 1), "Cell (9,1) should be ALIVE");
        
        // Test bounds checking
        bool exception_thrown = false;
        try {
            grid.get_cell(width, 0);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for out of bounds x coordinate");
        
        exception_thrown = false;
        try {
            grid.get_cell(0, height);
        } catch (const std::out_of_range&) {
            exception_thrown = true;
        }
        tc.assert_true(exception_thrown, "Should throw exception for out of bounds y coordinate");
    }

    // Test that to_original_representation uses get_cell correctly
    void test_to_original_representation_with_get_cell(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid to_original_representation with get_cell (" << test_tag << ") ---" << RESET << std::endl;

        // Create a small grid
        const size_t height = 16;
        const size_t width_words = 2;
        const size_t word_bits = sizeof(store_type) * 8;
        const size_t width = width_words * word_bits;
        const size_t total_cells = height * width;
        
        // Generate patterned cell states
        std::vector<bit_planes::TestCellState> input_grid(total_cells);
        for (size_t i = 0; i < input_grid.size(); ++i) {
            switch (i % 3) {
                case 0: input_grid[i] = bit_planes::TestCellState::DEAD; break;
                case 1: input_grid[i] = bit_planes::TestCellState::ALIVE; break;
                case 2: input_grid[i] = bit_planes::TestCellState::DYING; break;
            }
        }
        
        tested_grid grid(height, width, input_grid.data());
        
        // Compare direct get_cell with to_original_representation results
        auto result = grid.to_original_representation();
        
        auto all_correct = true;

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                bit_planes::TestCellState from_get_cell = grid.get_cell(x, y);
                bit_planes::TestCellState from_representation = result[idx];
                
                tc.silent_if_passed().assert_true(from_get_cell == from_representation, 
                    "get_cell and to_original_representation should return the same value at (" + 
                    std::to_string(x) + "," + std::to_string(y) + ")");

                all_correct = all_correct && (from_get_cell == from_representation);
            }
        }

        tc.assert_true(all_correct, "All cells should match between get_cell and to_original_representation");
    }

    // Test bit_planes_grid with a small random grid (for unit tests)
    void test_bit_grid_small_random(test_case& tc) {
        std::cout << BLUE << "\n--- Testing bit_planes_grid with small random pattern (" << test_tag << ") ---" << RESET << std::endl;

        // Create a smaller random grid for unit tests
        const size_t height = 8 * 4;
        const size_t width_words = 4;
        const size_t word_bits = sizeof(store_type) * 8;
        const size_t width = width_words * word_bits;
        const size_t total_cells = height * width;
        
        std::cout << "  Generating random grid with " << total_cells << " cells..." << std::endl;
        std::vector<bit_planes::TestCellState> input_grid(total_cells);
        
        // Initialize with random values
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 2);
        
        for (size_t i = 0; i < input_grid.size(); ++i) {
            int random_value = dist(rng);
            switch (random_value) {
                case 0: input_grid[i] = bit_planes::TestCellState::DEAD; break;
                case 1: input_grid[i] = bit_planes::TestCellState::ALIVE; break;
                case 2: input_grid[i] = bit_planes::TestCellState::DYING; break;
            }
        }
        
        std::cout << "  Creating bit planes grid..." << std::endl;
        tested_grid grid(height, width, input_grid.data());

        std::cout << "  Converting back to original representation..." << std::endl;
        auto result = grid.to_original_representation();
        
        std::cout << "  Verifying results..." << std::endl;
        tc.assert_equal(input_grid.size(), result.size(), "Result size should match input size");
        
        auto all_correct = true;
        
        for (size_t i = 0; i < input_grid.size(); ++i) {
            if (input_grid[i] != result[i]) {
                all_correct = false;
                tc.silent_if_passed().assert_true(false, "Cell at index " + std::to_string(i) + " should match original");
            }
        }
        
        tc.assert_true(all_correct, "All checked cells should match original");
    }
};

template <typename store_type, typename state_dict>
using linear_bit_planes = cellato::memory::grids::bit_planes::grid<store_type, state_dict>;

template <typename store_type, typename state_dict>
using tiled_bit_planes = cellato::memory::grids::tiled_bit_planes::grid<store_type, state_dict>;

// Helper function to register the suite with the manager
inline void register_bit_planes_grid_tests() {

    static constexpr char linear_planes_tag[] = "Linear Bit Planes";
    static bit_planes_grid_test_suite<linear_bit_planes, uint8_t, linear_planes_tag> linear_planes_suite;
    test_manager::instance().register_suite(&linear_planes_suite);

    // TODO finish tiled planes tests
    static constexpr char tiled_planes_tag[] = "Tiled Bit Planes";
    static bit_planes_grid_test_suite<tiled_bit_planes, uint64_t, tiled_planes_tag> tiled_planes_suite;
    test_manager::instance().register_suite(&tiled_planes_suite);
}

} // namespace cellato::tests

#endif // CELLATO_TESTS_BIT_PLANES_GRID_HPP
