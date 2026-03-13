#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$0")

# Uncomment one of these test configurations:

# type="standard"
# type="bit_array"
# type="bit_planes"
# type="tiled_bit_planes"

# automaton="game-of-life"
# automaton="wire"
# automaton="traffic"
# automaton="cyclic"
# automaton="forest-fire"
automaton="critters"
# automaton="fluid"

# x_size=$((8*4))
# y_size=$((8*4))
# steps=3

# x_size=$((1024*16))
# y_size=$((1024*16))
# steps=256

x_size=$((30*256))
y_size=$((7*30*32))
steps=64

# reference
traverser="simple"
type="standard"
device="CUDA"

# bit planes linear
# traverser="simple"
# type="bit_planes"
# device="CUDA"

# bit tiles
traverser="simple"
type="tiled_bit_planes"
device="CUDA"

# temporal
# traverser="temporal"
# type="bit_planes"
# device="CUDA"

# temporal tiled
# traverser="temporal"
# type="tiled_bit_planes"
# device="CUDA"

args="--automaton ${automaton} --seed 42 --device ${device} --traverser ${traverser} --evaluator ${type} --layout ${type} --word_size 64 --x_size ${x_size} --y_size ${y_size} --steps ${steps} --rounds 2 --warmup_rounds 0 --cuda_block_size_y 8 --temporal_tile_size_y 16 --temporal_steps 4"

# baseline
# args="--reference_impl baseline --automaton ${automaton} --seed 42 --device ${device} --x_size ${x_size} --y_size ${y_size} --steps ${steps} --cuda_block_size_y 4 --rounds 1 --warmup_rounds 1"


# Game of Life with standard grid on CUDA
# args="--automaton wire --word_size 64 \
# --device CUDA --layout tiled_bit_planes --traverser simple --evaluator tiled_bit_planes \
# --warmup_rounds 1 --rounds 3 \
# --steps 1000 --x_size 8192 --y_size 8192"

# Fire automaton with bit_array grid on CUDA
#args="--automaton fire --device CUDA --layout bit_array --steps 50 --x_size 2048 --y_size 2048"

# Wire automaton with bit_array grid and spatial blocking
#args="--automaton wire --device CUDA --layout bit_array --traverser spacial_blocking --x_tile_size 8 --y_tile_size 8 --steps 200 --x_size 4096 --y_size 4096"

# Greenberg automaton on CPU with bit_planes
#args="--automaton excitable --device CPU --layout bit_planes --evaluator bit_planes --steps 150 --x_size 512 --y_size 512 --word_size 64"

# Game of Life on CPU for comparison with CUDA
#args="--automaton game_of_life --device CPU --layout standard --steps 100 --x_size 1024 --y_size 1024"

# Visualize output with print option
#args="--automaton game_of_life --device CUDA --layout bit_array --steps 20 --x_size 32 --y_size 32 --print"

should_remove="${1:-}"

if [ "$should_remove" == "clean" ]; then
    echo "Removing old build..."
    rm -rf "$script_dir/../bin"
fi

cd "$script_dir"
# srun -p gpu-short -A kdss --cpus-per-task=32 --mem=64GB --time=2:00:00 --gres=gpu:A100 make -j run ARGS="$args"
srun -p gpu-short -A kdss --cpus-per-task=32 --mem=64GB --time=2:00:00 --gres=gpu:H100 make -j run ARGS="$args"