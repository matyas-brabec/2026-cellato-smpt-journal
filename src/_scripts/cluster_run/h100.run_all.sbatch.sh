#!/bin/bash

#SBATCH -p gpu-long           # partition name
#SBATCH -A kdss                # account name
#SBATCH --cpus-per-task=64     # number of CPUs
#SBATCH --mem=128GB            # memory
#SBATCH --time=168:00:00       # time limit (HH:MM:SS)
#SBATCH --gres=gpu:H100        # GPU resource
#SBATCH -o ../__slurm__/h100_job-%j.out        # output file (%j expands to job ID)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Print some info for debugging
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "Scripts directory: ${SCRIPT_DIR}"

ID=$(date +%Y%m%d-%H%M%S)
RAND_3_LETTERS=$(tr -dc A-Z </dev/urandom | head -c 3 ; echo '')

# Activate virtual environment if needed (uncomment and modify if you use one)
# source /path/to/your/venv/bin/activate

# Run the Python script from scripts folder regardless of where this sbatch is called from
mkdir -p ./results
python ./run_all.py $@ > ./results/h100_cuda_test_${ID}_${RAND_3_LETTERS}.csv

echo "Job completed"