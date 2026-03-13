# Improving Cellular Automata Performance with Bit-Planes Encoding and Bitwise Vectorization 🔍

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE) [![doi](https://img.shields.io/badge/DOI-TODO-blue)](https://doi.org/TODO)

This repository accompanies the paper titled "Improving Cellular Automata Performance with Bit-Planes Encoding and Bitwise Vectorization":

```bibtex
@article{
  TODO
}
```

## 🚀 Overview

Cellular automata (CA) are discrete computational models widely used to simulate complex systems through simple, localized rules. While CA are primarily valued for their ability to visualize complex phenomena, certain simulations, such as traffic models or electrical circuits, demand high-performance processing to be practical. Being a special case of stencil computations, CA are well-suited for data parallel models and can benefit from both vectorization and GPU offloading. However, the performance optimization of CA has not been thoroughly explored, leaving many open questions. In this paper, we propose a bit-plane data encoding of CA cell states that enables efficient bitwise vectorization and works well with well-known optimizations like temporal blocking. We implemented 10 different CA using this technique to demonstrate the versatility of our approach and performed extensive evaluation. Furthermore, our implementation took advantage of Cellato abstraction, which allows simple CA rule definitions using C++ templates while abstracting away the complex implementation details. Our approach offers speedup of two orders of magnitude (comparing baseline and optimized CUDA implementations) whilst maintaining code simplicity and ease of use for the end users.

---

## 📂 Repository Structure

```text
.
├── LICENSE
├── README.md
├── include/             ← Cellato library
├── src/
│   ├── game_of_life/    ← Game of Life example
│   ├── fire/            ← Forest Fire example
│   ├── wire/            ← WireWorld example
│   ├── excitable/       ← Greenberg–Hastings example
│   ├── .../             ← ... remaining 6 automata
│   └── _scripts/        ← Benchmark & plotting scripts
└── results/             ← Benchmark outputs (CSV, PNG, PDF)
```

### 🔍 Implemented Automata

| Automaton              | Key in paper | Description                                             | Implementation Directory                 |
| ---------------------- | ------------ | ------------------------------------------------------- | ---------------------------------------- |
| **Game of Life**       | `GoL`        | Conway’s binary grid (Moore neighborhood)               | [`src/game_of_life`](./src/game_of_life) |
| **Forest Fire**        | `fire`       | Spread of forest fire simulation (von Neumann)          | [`src/fire`](./src/fire)                 |
| **WireWorld**          | `wire`       | Digital circuit simulator (4 states)                    | [`src/wire`](./src/wire)                 |
| **Greenberg–Hastings** | `excitable`  | excitable medium with refractory states                 | [`src/excitable`](./src/excitable)       |
| **Maze**               | `maze`       | Maze generating CA                                      | [`src/maze`](./src/maze)                 |
| **Brian's Brain**      | `brian`      | Game of life cousin with 3 states                       | [`src/brian`](./src/brian)               |
| **Cyclic**             | `cyclic`     | Modeling of excitable medium (32 states)                | [`src/cyclic`](./src/cyclic)             |
| **Fluid Simulation**   | `fluid`      | The Hardy–Pomeau–Pazzis (HPP) model                     | [`src/fluid`](./src/fluid)               |
| **Critters**           | `critters`   | Reversible automaton with a Margolus block neighborhood | [`src/critters`](./src/critters)         |
| **Traffic**            | `traffic`    | A traffic simulation using 2 different cars             | [`src/traffic`](./src/traffic)           |

## 🛠️ Core Cellato Components

All core headers live in [`include/`](./include/). Key components:

| Component| Header    |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AST nodes**  | [`include/core/ast.hpp`](./include/core/ast.hpp)    |
| **Evaluators** | [`include/evaluators/standard.hpp`](./include/evaluators/standard.hpp) • [`bit_array.hpp`](./include/evaluators/bit_array.hpp) • [`bit_planes.hpp`](./include/evaluators/bit_planes.hpp) • [`tiled_bit_planes.hpp`](./include/evaluators/tiled_bit_planes.hpp)|
| **Memory layouts**   | [`include/memory/standard_grid.hpp`](./include/memory/standard_grid.hpp) • [`bit_array_grid.hpp`](include/memory/bit_array_grid.hpp) • [`bit_planes_grid.hpp`](./include/memory/bit_planes_grid.hpp) • [`tiled_bit_planes_grid.hpp`](./include/memory/tiled_bit_planes_grid.hpp)|
| **Traversers (iteration)** | CPU: [`traversers/cpu/simple.hpp`](./traversers/cpu/simple.hpp)<br>CUDA: `traversers/cuda/simple.{hpp,cu}` [.hpp](./include/traversers/cuda/simple.hpp) [.cu](./include/traversers/cuda/simple.cu), `…/temporal.{hpp,cu}` [.hpp](./include/traversers/cuda/temporal.hpp) [.cu](./include/traversers/cuda/temporal.cu) |

## 📖 Tutorial

### 🔧 Prerequisites

* **Compiler:** GCC 15.2.0
* **CUDA:** NVCC V13.0.88

### Compilation

```bash
# Clone & enter
$> git clone https://github.com/matyas-brabec/2026-cellato-smpt-journal
$> cd 2026-cellato-smpt-journal

# Build Cellato and the `baseline` reference implementation
$> (cd src && make)

# Run the CLI test harness
$> ./bin/cellato <options>
```

### ▶️ Running Examples

```bash
# Game of Life on CPU, standard layout
$> ./bin/cellato \
  --automaton game-of-life \
  --device CPU \
  --traverser simple \
  --evaluator standard \
  --layout standard \
  --x_size 256 --y_size 256 \
  --steps 100

# Game of Life on CUDA with bit-planes
$> ./bin/cellato \
  --automaton game-of-life \
  --device CUDA \
  --traverser simple \
  --evaluator bit_planes \
  --layout bit_planes \
  --word_size 32 \
  --x_size 4096 --y_size 4096 \
  --steps 1000 \
  --cuda_block_size_x 32 --cuda_block_size_y 8
```

---

## ⚙️ CLI Options Explained

```bash
Usage: ./cellato [options]
Options:
  --automaton <name>           Name of the automaton to run (game-of-life, forest-fire, wire, excitable, ...)
  --device <CPU|CUDA>          Execution device
  --traverser <name>           Traversal strategy (simple, temporal)
  --evaluator <name>           Evaluator type (standard, bit_array, bit_planes, tiled_bit_planes)
  --layout <name>              Memory layout (standard, bit_array, bit_planes, tiled_bit_planes)
  --reference_impl <name>      Run reference implementation (baseline)
  --x_size <N>                 Grid width
  --y_size <N>                 Grid height
  --rounds <N>                 Number of benchmarking rounds
  --warmup_rounds <N>          Number of warmup rounds
  --steps <N>                  Number of CA time steps
  --word_size <32|64>          Word word_size used by the `bit array`, `bit planes` and `tiled_bit_planes`
  --seed <N>                   RNG seed for initialization
  --print                      Print grid state after each step
  --print_csv_header           Emit CSV header line
  --cuda_block_size_x <N>      CUDA block X dimension (default: 32)
  --cuda_block_size_y <N>      CUDA block Y dimension (default: 8)
  --temporal_steps             Number of temporal steps for the temporal implementation
  --temporal_tile_size_y       Height of temporal block (Width is fixed to 32)
  --help                       Show this help message
```

### ✨ Supported Evaluator / Layout / Traverser Combinations

Note that a specific implementation is uniquely identified by the triplet of options `--traverser`, `--evaluator`, and `--layout`. However, not all combinations are allowed. For CUDA (`--device CUDA`) and `32`/`64`-bit word_size (`--word_size (32|64)`), we have implemented the following.

| Implementation | Traverser | Evaluator | Layout | Notes |
| --- | --- | --- | --- | --- |
| Baseline | | | | As baseline does not use the Cellato framework, it has none of the mentioned options set. Instead, it is invoked with the option `--reference_impl baseline`. |
| Cellato Standard | `simple` | `standard` | `standard` | The implementation uses a standard encoding to represent states of the automaton (`enum type`) and the CUDA kernel evaluates one cell per CUDA thread. |
| Bit Packed Representation | `simple` | `bit_array` | `bit_array` | This implementation uses the well-known "bit packing" technique; i.e., many states are packed into one machine word. The number that fits depends both on the `--automaton` in question and the `--word_size` of the machine word used. |
| Linear Bit Planes | `simple` | `bit_planes` | `bit_planes` | The integer representation of the cell state is split into independent "bit planes". These cells are then processed in a vectorized manner (for details, see the affiliated paper). |
| Tiled Bit Planes | `simple` | `tiled_bit_planes` | `tiled_bit_planes` | Similar to the Linear Bit Planes, but one machine word does not encode a row of consecutive cells - as in the previous case - but a small 8x8 (or 8x4) tile. |
| Temporal Bit Planes (Linear) | `temporal` | `bit_planes` | `bit_planes` | Implementation of temporal blocking. The number of steps is set using `--temporal_steps` option, and the size of the temporal block is set using `--temporal_tile_size_y` - the `x` dimension of the temporal block is fixed to `32`. |
| Temporal Bit Planes (Tiled) | `temporal` | `tiled_bit_planes` | `tiled_bit_planes` | Same as the above but uses the 8x8 (or 8x4) machine word tiles. |

#### ⚠️🚫 Limitations

| Implementation | Limitation Description |
| --- | --- |
| Baseline | The grid X (`--x_size`) and Y (`--y_size`) dimensions **must** be divisible by the CUDA thread block `X` (`--cuda_block_size_x`) and `Y` (`--cuda_block_size_y`), respectively. |
| Cellato standard | Same limitations as for *Baseline* |
| Bit Packed Representation | Defining `k` as the number of cells in a machine word (`k = word_size // bits_per_automaton_state`).<br> The grid size `X` (`--x_size`) **must** be divisible by the thread block size `X` (`--cuda_block_size_x`) times `k`. <br> The grid size `Y` (`--y_size`) **must** be divisible by the thread block size `Y` (`--cuda_block_size_y`). |
| Linear Bit Planes | The grid size `X` (`--x_size`) **must** be divisible by the `--word_size` times cuda thread block block size `X` (`--cuda_block_size_x`). <br> The grid size `Y` (`--y_size`) **must** be divisible by the thread block size `Y` (`--cuda_block_size_y`). |
| Tiled Bit Planes | The grid size `X` (`--x_size`) **must** be divisible by `8` times cuda thread block block size `X` (`--cuda_block_size_x`). <br> The grid size `Y` (`--y_size`) **must** be divisible by the thread block size `Y` (`--cuda_block_size_y`) times `4` or `8` for `--word_size 32` and `--word_size 64` respectively. |
| Temporal Bit Planes (Linear) | Defining the `effective_temporal_size_Y` as `--temporal_tile_size_y - 2 * --temporal_steps`. <br> The grid size `X` (`--x_size`) **must** be divisible by `30` (which is warp size `32` minus the halo of `2`) times `--word_size`. <br> The grid size `Y` (`--y_size`) **must** be divisible by `effective_temporal_size_Y`. <br> The `--temporal_tile_size_y` **must** be divisible by the thread block size `Y` (`--cuda_block_size_y`). <br> The total simulation steps (`--steps`) must be divisible by the `--temporal_steps`. <br> The thread block size X `--cuda_block_size_x` **must** be 32. |
| Temporal Bit Planes (Tiled) | Defining the `effective_temporal_size_X` as `32 - 2 * word_halo` where `word_halo_x` is `ceil(8 / --time_steps)`. <br> Defining the `effective_temporal_size_Y` as `--temporal_tile_size_y - 2 * word_halo` where `word_halo_x` is `ceil(8 / --time_steps)` for `--word_size 64` and `ceil(4 / --time_steps)` for `--word_size 32`.  <br> The grid size `X` (`--x_size`) **must** be divisible by `effective_temporal_size_X` <br> The grid size `Y` (`--y_size`) **must** be divisible by `effective_temporal_size_Y` <br> The `--temporal_tile_size_y` **must** be divisible by the thread block size `Y` (`--cuda_block_size_y`). <br> The total simulation steps (`--steps`) must be divisible by the `--temporal_steps` <br> The thread block size X `--cuda_block_size_x` **must** be 32. |

⚠️ **IMPORTANT NOTE**: The temporal blocking traverser requires the `--temporal_steps`, `--temporal_tile_size_y`, and `--cuda_block_size_y` parameters to be known at **compile-time**. Specific values for **verification** and **benchmarking** are already configured. You can use them by running `make COMPILE_TO_VERIFY` or `make COMPILE_TO_BENCHMARK`. If you need to use custom values, edit the [`temporal.cu`](./include/traversers/cuda/temporal.cu) kernel file. The section that needs modification is clearly marked within the source code.

#### 🧩 Examples for each combination

```bash
# ▶️ Standard layout + evaluator
$> ./bin/cellato \
  --evaluator standard \
  --layout standard \
  --traverser simple \
  [other options…]

# ▶️ Bit-array layout + evaluator (32-bit)
$> ./bin/cellato \
  --evaluator bit_array \
  --layout bit_array \
  --word_size 32 \
  --traverser simple \
  [other options…]

# ▶️ Bit-planes layout + evaluator (32-bit)
$> ./bin/cellato \
  --evaluator bit_planes \
  --layout bit_planes \
  --word_size 32 \
  --traverser simple \
  [other options…]

# ▶️ Tiled bit-planes layout + evaluator (32-bit)
$> ./bin/cellato \
  --evaluator tiled_bit_planes \
  --layout tiled_bit_planes \
  --word_size 32 \
  --traverser simple \
  [other options…]

# ▶️ Temporal bit-planes layout + evaluator (32-bit)
$> ./bin/cellato \
  --evaluator bit_planes \
  --layout bit_planes \
  --word_size 32 \
  --traverser temporal \
  [other options…]

# ▶️ Temporal tiled bit-planes layout + evaluator (32-bit)
$> ./bin/cellato \
  --evaluator tiled_bit_planes \
  --layout tiled_bit_planes \
  --word_size 32 \
  --traverser temporal \
  [other options…]

```

---

## 📊 Scripts & Benchmarks

First, you'll need to compile the project for benchmarking. Note that this initial compilation can take **up to an hour**, as it builds all temporal blocking variants required for the tests.

```bash
$> (cd src && make clean COMPILE_TO_BENCHMARK)
```

Once compilation is complete, you can use the following scripts located in the `src/_scripts/` directory to reproduce the results from the paper.

```bash
# Measure all the variants producing CSV file `results.csv`
$> python ./src/_scripts/cluster_run/run_all.py > results.csv

# Generate a detailed results table in ASCII format
$> python ./src/_scripts/table-producers/show_result_table.py results.csv

# Generate an HTML report with detailed results, including hyperparameter information
$> python ./src/_scripts/table-producers/to_html_report.py results.csv > report.html

# Generate the graphs presented in the paper
# The final argument specifies the output file name (e.g., .png, .pdf)
$> python ./src/_scripts/graph-producers/0_total_perf.py      results.csv   total_perf.pdf
$> python ./src/_scripts/graph-producers/1_linear_vs_tiled.py results.csv   linear_vs_tiled.pdf
$> python ./src/_scripts/graph-producers/2_time_steps.py      results.csv   time_steps.pdf
```

### ✅ Verification

Due to certain implementation constraints, grid sizes may differ slightly across benchmarks. As a result, the checksums produced by `run_all.py` are not directly comparable for validation.

To verify the correctness of the implementations, you must first perform a quick, specialized compilation. This step configures the specific temporal blocking values that the verification script requires.

```bash
$> (cd src && make clean COMPILE_TO_VERIFY)
```

Now you can run the verification script:

```bash
$> python ./src/_scripts/cluster_run/verify.py

Running all automata
Automata to test: ['critters', 'traffic', 'fluid', 'game-of-life', 'cyclic', 'brian', 'maze', 'forest-fire', 'wire', 'excitable']
Starting correctness validation...
--------------------------------------------------------------------------------

--- Automaton: critters ---
[  OK   ] Standard                  | Automaton: critters, Eval: standard, Layout: standard
[  OK   ] BitArray                  | Automaton: critters, Eval: bit_array, Layout: bit_array, Prec: 32
[  OK   ] BitPlanes                 | Automaton: critters, Eval: bit_planes, Layout: bit_planes, Prec: 32
...
```

⚠️ Please note that the script will skip some tests, as certain parameter combinations are not valid for all automaton types.

---

## 📈 Results

We have performed a comprehensive evaluation of our bit-plane methods. The complete results are available in the [`./results/`](./results/) directory, which contains separate subdirectories for the **H100** and **A100** GPUs.

Each subdirectory includes:

* The full set of measurements in a `.csv` file ([`_h100.csv`](./results/H100/_h100.csv) and [`_a100.csv`](./results/A100/_a100.csv)).
* Graphs generated from the performance data.
* Detailed HTML report tables.

Additionally, the raw data from our initial hyperparameter grid search can be found in [`./results/H100/grid-search-results.csv`](./results/H100/grid-search-results.csv).

---

<!-- ## 🔭 Future Work

* **Higher-dimensional grids:** 3D+ support
* **Non-rectangular topologies:** hexagonal, triangular
* **Probabilistic CAs:** introduce random-node AST types
* **Advanced traversers:** temporal blocking, NUMA-aware scheduling
* **Distributed execution:** MPI-based traverser with halo exchange
* **Framework integration:** embed Cellato evaluators into Kokkos/GridTools

--- -->

## 📝 License

This project is released under the [MIT License](./LICENSE).
