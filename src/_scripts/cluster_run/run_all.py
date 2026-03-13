import math
import os
import sys
import subprocess
import time

EXE_PATH = "bin/cellato"

ROUNDS = 5         # Number of measurement rounds
WARMUP = 1         # Number of warmup rounds

SEED = 42
DEVICE = "CUDA"

group_number = int(sys.argv[1] if len(sys.argv) > 1 else "-1")

# GRID_SIZES = [x ** 2 for x in [4096, 8192, 16384, 32768]]
# STEPS =                       [1024,  256,   128,    64]
GRID_SIZES = [x ** 2 for x in [16384]]
STEPS =                       [128]
Y_BLOCK_SIZES = [2, 4, 8, 16, 32]
TEMPORAL_TILE_SIZES_Y = [8, 16, 32, 64, 128] # 256 is too large even for a single bit automaton using 32-bit word_size
WORD_SIZES = [32, 64]
TEMPORAL_STEPS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 20, 22, 24]


ALL_AUTOMATA = [
    "critters",
    "traffic",
    "fluid",
    "game-of-life",
    "cyclic",

    "brian",
    "maze",
    "forest-fire",
    "wire",
    "excitable",
]

if group_number == -1:
    print ("Running all automata", file=sys.stderr)
    AUTOMATA_TO_TEST = ALL_AUTOMATA
else:
    print (f"Running group number: {group_number}", file=sys.stderr)
    AUTOMATA_TO_TEST = ALL_AUTOMATA[(group_number * 5): (group_number * 5 + 5)]

print (f"Automata to test: {AUTOMATA_TO_TEST}", file=sys.stderr)

AUTOMATA_bits = {
    "game-of-life": 1,
    "brian": 2,
    "maze": 1,
    "critters": 1,
    "forest-fire": 2,
    "wire": 2,
    "traffic": 2,
    "excitable": 3,
    "fluid": 4,
    "cyclic": 5,
}

average_halo_radii = {
    "game-of-life": 1.0,
    "brian": 1.0,
    "maze": 1.0,
    "critters": 1.0,
    "forest-fire": 1.0,
    "wire": 1.0,
    "traffic": 1.0,
    "excitable": 1.0,
    "fluid": 1.0,
    "cyclic": 1.0,
}

# max sizes for H100 GPU
biggest_temporal_tile_size_for_automata = {
    32: {
        "game-of-life": 128,
        "brian": 64,
        "maze": 128,
        "critters": 128,
        "forest-fire": 64,
        "wire": 64,
        "traffic": 64,
        "excitable": 64,
        "fluid": 32,
        "cyclic": 32,
    },
    64: {
        "game-of-life": 64,
        "brian": 32,
        "maze": 64,
        "critters": 64,
        "forest-fire": 32,
        "wire": 32,
        "traffic": 32,
        "excitable": 32,
        "fluid": 16,
        "cyclic": 16,
    }
}

terminal_supports_colors = sys.stdout.isatty() and os.name != 'nt' and 'NO_COLOR' not in os.environ

# ANSI color codes for terminal output
GREY_COLOR = "\033[90m" if terminal_supports_colors else ""
RESET_COLOR = "\033[0m" if terminal_supports_colors else ""
RED_COLOR = "\033[91m" if terminal_supports_colors else ""
BLUE_COLOR = "\033[94m" if terminal_supports_colors else ""
GREEN_COLOR = "\033[92m" if terminal_supports_colors else ""
CYAN_COLOR = "\033[96m" if terminal_supports_colors else ""
YELLOW_COLOR = "\033[93m" if terminal_supports_colors else ""

def get_effective_xy_block_size(
    temporal_tile_size_x, temporal_tile_size_y,
    temporal_steps, average_halo_radius,
    x_word_tile_size, y_word_tile_size):
    
    needed_halo_cells = math.ceil(temporal_steps * average_halo_radius * 0.999)
    
    x_halo_words = (needed_halo_cells + x_word_tile_size - 1) // x_word_tile_size
    y_halo_words = (needed_halo_cells + y_word_tile_size - 1) // y_word_tile_size
    
    effective_temporal_tile_size_x = temporal_tile_size_x - (2 * x_halo_words)
    effective_temporal_tile_size_y = temporal_tile_size_y - (2 * y_halo_words)

    return (effective_temporal_tile_size_x, effective_temporal_tile_size_y)
class TestCase:
    def __init__(self):
        self.automaton = None
        self.device = None

        self.elem_count = None
        self.steps = None

        self.cuda_block_size_y = None
        
        self.word_size = None
        
        self.temporal_steps = None
        self.temporal_tile_size_y = None


        self.rounds = ROUNDS
        self.warmup_rounds = WARMUP


    def clone(self):
        tc = TestCase()
        for attr, value in self.__dict__.items():
            setattr(tc, attr, value)
        return tc



    def with_attr(self, attr, value):
        setattr(self, attr, value)
        return self

    def with_automaton(self, automaton):
        return self.with_attr("automaton", automaton)

    def with_elem_count(self, elem_count):
        return self.with_attr("elem_count", elem_count)

    def with_steps(self, steps):
        return self.with_attr("steps", steps)

    def with_cuda_block_size_y(self, block_size_y):
        return self.with_attr("cuda_block_size_y", block_size_y)

    def with_word_size(self, word_size):
        return self.with_attr("word_size", word_size)
    
    def with_temporal_steps(self, temporal_steps):
        return self.with_attr("temporal_steps", temporal_steps)
    
    def with_temporal_tile_size_y(self, temporal_tile_size_y):
        return self.with_attr("temporal_tile_size_y", temporal_tile_size_y)

    def with_device(self, device):
        return self.with_attr("device", device)

class Dims:
    def with_elem_count(self, elem_count):
        self.elem_count = elem_count
        return self

    def x_divisible_by(self, divisor):
        self.x_divisor = divisor
        return self
    
    def y_divisible_by(self, divisor):
        self.y_divisor = divisor
        return self

    def get_xy(self):
        root = int(self.elem_count**0.5)
        x = root - (root % self.x_divisor)
        y = self.elem_count // x
        y = y - (y % self.y_divisor)
        return (x, y)

def lcm(a, b):
    return (a * b) // math.gcd(a, b)

class ReferenceImplementation:
    @staticmethod
    def params(tc: TestCase):
        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(32).y_divisible_by(tc.cuda_block_size_y).get_xy()

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --reference_impl baseline --x_size {x} --y_size {y} --steps {tc.steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y}"
    
class StandardImplementation:
    @staticmethod
    def params(tc: TestCase):
        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(32).y_divisible_by(tc.cuda_block_size_y).get_xy()

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --traverser simple --evaluator standard --layout standard --x_size {x} --y_size {y} --steps {tc.steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y}"


class BitArrayImplementation:
    @staticmethod
    def params(tc: TestCase):
        x_block = 32
        y_block = tc.cuda_block_size_y

        cells_per_word = tc.word_size // AUTOMATA_bits[tc.automaton]

        x_divisor = x_block * cells_per_word

        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(x_divisor).y_divisible_by(y_block).get_xy()

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --traverser simple --evaluator bit_array --layout bit_array --word_size {tc.word_size} --x_size {x} --y_size {y} --steps {tc.steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y}"
    

class BitPlanesImplementation:
    @staticmethod
    def params(tc: TestCase):
        x_block = 32
        y_block = tc.cuda_block_size_y

        cells_per_word = tc.word_size
        x_divisor = x_block * cells_per_word 

        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(x_divisor).y_divisible_by(y_block).get_xy()

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --traverser simple --evaluator bit_planes --layout bit_planes --word_size {tc.word_size} --x_size {x} --y_size {y} --steps {tc.steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y}"
    

class TiledBitPlanesImplementation:
    @staticmethod
    def params(tc: TestCase):
        x_block = 32
        y_block = tc.cuda_block_size_y

        cells_per_word_in_x = 8
        cells_per_word_in_y = tc.word_size // cells_per_word_in_x

        x_divisor = x_block * cells_per_word_in_x
        y_divisor = y_block * cells_per_word_in_y

        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(x_divisor).y_divisible_by(y_divisor).get_xy()

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --traverser simple --layout tiled_bit_planes --evaluator tiled_bit_planes --word_size {tc.word_size} --x_size {x} --y_size {y} --steps {tc.steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y}"

class TemporalLinearImplementation:
    @staticmethod
    def params(tc: TestCase):
        effective_temporal_tile_size_x, effective_temporal_tile_size_y \
            = TemporalLinearImplementation.get_effective_xy_block_size(tc)

        x_word_tile_size = tc.word_size
        y_word_tile_size = 1

        x_divisor = effective_temporal_tile_size_x * x_word_tile_size
        y_divisor = effective_temporal_tile_size_y * y_word_tile_size

        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(x_divisor).y_divisible_by(y_divisor).get_xy()

        divisible_steps = tc.steps - (tc.steps % tc.temporal_steps)

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --traverser temporal --layout bit_planes --evaluator bit_planes --word_size {tc.word_size} --x_size {x} --y_size {y} --steps {divisible_steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y} --temporal_tile_size_y {tc.temporal_tile_size_y} --temporal_steps {tc.temporal_steps}"

    @staticmethod
    def get_effective_xy_block_size(tc: TestCase):
        return get_effective_xy_block_size(
            temporal_tile_size_x = 32,
            temporal_tile_size_y = tc.temporal_tile_size_y,
            temporal_steps = tc.temporal_steps,
            average_halo_radius = average_halo_radii[tc.automaton],
            x_word_tile_size = tc.word_size,
            y_word_tile_size = 1
        )

class TemporalTiledBitPlanesImplementation:
    @staticmethod
    def params(tc: TestCase):
        effective_temporal_tile_size_x, effective_temporal_tile_size_y \
            = TemporalTiledBitPlanesImplementation.get_effective_xy_block_size(tc)

        x_word_tile_size = 8
        y_word_tile_size = tc.word_size // x_word_tile_size

        x_divisor = effective_temporal_tile_size_x * x_word_tile_size
        y_divisor = effective_temporal_tile_size_y * y_word_tile_size

        x, y = Dims().with_elem_count(tc.elem_count).x_divisible_by(x_divisor).y_divisible_by(y_divisor).get_xy()

        divisible_steps = tc.steps - (tc.steps % tc.temporal_steps)

        return f"--automaton {tc.automaton} --seed {SEED} --device {tc.device} --traverser temporal --layout tiled_bit_planes --evaluator tiled_bit_planes --word_size {tc.word_size} --x_size {x} --y_size {y} --steps {divisible_steps} --rounds {tc.rounds} --warmup_rounds {tc.warmup_rounds} --cuda_block_size_y {tc.cuda_block_size_y} --temporal_tile_size_y {tc.temporal_tile_size_y} --temporal_steps {tc.temporal_steps}"

    @staticmethod
    def get_effective_xy_block_size(tc: TestCase):
        return get_effective_xy_block_size(
            temporal_tile_size_x = 32,
            temporal_tile_size_y = tc.temporal_tile_size_y,
            temporal_steps = tc.temporal_steps,
            average_halo_radius = average_halo_radii[tc.automaton],
            x_word_tile_size = 8,
            y_word_tile_size = tc.word_size // 8
        )

class ParamsGenerator:

    def generate(self):
        return self._generate_with_automata()

    def _generate_with_automata(self):
        test_case = TestCase().with_device(DEVICE)
        all = []

        for automaton in AUTOMATA_TO_TEST:
            tc = test_case.with_automaton(automaton)
            all.extend(self._generate_with_size_and_steps(tc))

        return all

    def _generate_with_size_and_steps(self, tc: TestCase):
        all = []

        for (size, steps) in zip(GRID_SIZES, STEPS):
            passed_tc = tc.with_elem_count(size).with_steps(steps)
            all.extend(self._generate_with_y_block_size(passed_tc))

        return all

    def _generate_with_y_block_size(self, tc: TestCase):
        all = []

        for y_block in Y_BLOCK_SIZES:
            passed_tc = tc.with_cuda_block_size_y(y_block)

            all.append(ReferenceImplementation.params(passed_tc))
            all.append(StandardImplementation.params(passed_tc))
            all.extend(self._generate_with_word_size(passed_tc))

        return all


    def _generate_with_word_size(self, tc: TestCase):
        all = []

        for word_size in WORD_SIZES:
            passed_tc = tc.with_word_size(word_size)

            all.append(BitArrayImplementation.params(passed_tc))
            all.append(BitPlanesImplementation.params(passed_tc))
            all.append(TiledBitPlanesImplementation.params(passed_tc))
            all.extend(self._generate_with_temporal_params(passed_tc))

        return all

    def _generate_with_temporal_params(self, tc: TestCase):
        all = []
        
        for temporal_steps in TEMPORAL_STEPS:
            passed_tc = tc.with_temporal_steps(temporal_steps)

            all.extend(self._generate_with_temporal_tile_size_y(passed_tc))

        return all

    def _generate_with_temporal_tile_size_y(self, tc: TestCase):
        all = []

        temporal_tile_size_y = biggest_temporal_tile_size_for_automata[tc.word_size][tc.automaton]

        passed_tc = tc.with_temporal_tile_size_y(temporal_tile_size_y)
        
        if (temporal_tile_size_y < tc.cuda_block_size_y):
            return []  # temporal tile size Y must be at least as large as block size Y

        if (temporal_tile_size_y % tc.cuda_block_size_y != 0):
            return []  # temporal tile size Y must be divisible by block size Y

        _, effective_y_for_linear = TemporalLinearImplementation.get_effective_xy_block_size(passed_tc)
        _, effective_y_for_tiled = TemporalTiledBitPlanesImplementation.get_effective_xy_block_size(passed_tc)

        if (effective_y_for_linear > 0):
            all.append(TemporalLinearImplementation.params(passed_tc))

        if (effective_y_for_tiled > 0):
            passed_tc = tc.with_temporal_tile_size_y(temporal_tile_size_y)
            all.append(TemporalTiledBitPlanesImplementation.params(passed_tc))

        return all

class Executable:
    def __init__(self):
        # Set up paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.join(script_dir, "..", "..", "..")
        self.path = os.path.join(project_dir, EXE_PATH)
        
        # Check if executable exists
        if not os.path.exists(self.path):
            print(f"Error: Executable not found at {self.path}", file=sys.stderr)
            print("Try running 'make' first", file=sys.stderr)
            sys.exit(1)
        
    def run(self, args, throw_on_error=False):
        """Run the executable with the given arguments and return the output."""
        cmd = [self.path] + args.split()
        
        try:
            # print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Extract the CSV line (last non-empty line of stdout)
            csv_lines = [line for line in process.stdout.strip().split('\n') if line.strip()]
            if csv_lines:
                return csv_lines[-1]  # Return the last line which is the CSV data
            else:
                if throw_on_error:
                    raise RuntimeError(f"No output from command: {' '.join(cmd)}")

                print(f"No output from command: {' '.join(cmd)}", file=sys.stderr)
                return None
        except subprocess.CalledProcessError as e:
            if throw_on_error:
                raise RuntimeError(f"Error running command: {' '.join(cmd)}\nSTDERR: {e.stderr}") from e
                
            print(f"Error running command: {' '.join(cmd)}", file=sys.stderr)
            print(f"STDERR: {e.stderr}", file=sys.stderr)
            return None
    
    def get_csv_header(self):
        """Get the CSV header from the executable."""
        try:
            result = subprocess.run(
                [self.path, "--print_csv_header"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error getting CSV header: {e}", file=sys.stderr)
            print(f"STDERR: {e.stderr}", file=sys.stderr)
            sys.exit(1)

### MAIN SCRIPT ###

def main():
    start_time = time.time()
    
    # Initialize the executable
    executable = Executable()
    
    # Get and print CSV header
    header = executable.get_csv_header()
    print(header)  # Print CSV header to stdout
    
    # Generate test cases
    generator = ParamsGenerator()
    test_cases = generator.generate()

    # test_cases = [tc for tc in test_cases if 'temporal' not in tc]  # TEMPORAL TESTS ARE DISABLED FOR NOW
    # test_cases = [tc for tc in test_cases if 'bit_array' not in tc]  # ONLY 32-BIT TESTS FOR NOW
    # test_cases = [tc for tc in test_cases if 'bit_planes' not in tc]  # ONLY 32-BIT TESTS FOR NOW
    # # for t in test_cases:
    # #     print(t, file=sys.stderr)

    # # exit(0)

    empirical_time_per_case = 4 * 13.0 / 11.2
    secs_per_case = empirical_time_per_case * (ROUNDS + WARMUP)  # Rough estimate of seconds per test case
    count = len(test_cases)
    total_time_hours = (secs_per_case * count) / 3600
 
    print(f"{CYAN_COLOR}Generated {count} test cases{RESET_COLOR}", file=sys.stderr)
    print(f"{GREEN_COLOR}Estimated total time: {total_time_hours:.2f} hours{RESET_COLOR}", file=sys.stderr)
    print(f"{YELLOW_COLOR}Starting benchmark run at {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET_COLOR}", file=sys.stderr)
    
    # Run all test cases
    for i, test_case in enumerate(test_cases):
        print(f"{GREY_COLOR}Running test case {i+1}/{count}: {test_case}{RESET_COLOR}", file=sys.stderr)
        csv_line = executable.run(test_case)
        if csv_line:
            print(csv_line)  # Print CSV data to stdout
            sys.stdout.flush()  # Ensure output is flushed immediately
        else:
            print(f"{RED_COLOR}Failed to get results for: {test_case}{RESET_COLOR}", file=sys.stderr)
    
    # Calculate and print elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_hours = elapsed_time / 3600
    elapsed_minutes = (elapsed_time % 3600) / 60
    elapsed_seconds = elapsed_time % 60
    
    print(f"{GREEN_COLOR}Total benchmark time: {int(elapsed_hours)}h {int(elapsed_minutes)}m {int(elapsed_seconds)}s{RESET_COLOR}", file=sys.stderr)
    print(f"{YELLOW_COLOR}Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET_COLOR}", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
