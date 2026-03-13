import run_all as run_all
import sys
import re

# --- Configuration ---
# Use a moderately sized grid and fewer steps for quick validation.
# The goal is correctness checking, not performance measurement.
GRID_SIZE = (2048)**2
STEPS = 128

# Define the parameter space to check
Y_BLOCK_SIZES = [2, 4]
WORD_SIZES = [32, 64]
TEMPORAL_STEPS = [4, 8, 12, 20]
TEMPORAL_TILE_SIZES_Y = [8, 32]

AUTOMATA_TO_TEST = run_all.AUTOMATA_TO_TEST

# --- Color Codes (imported from cuda_test) ---
GREY_COLOR = run_all.GREY_COLOR
RESET_COLOR = run_all.RESET_COLOR
RED_COLOR = run_all.RED_COLOR
GREEN_COLOR = run_all.GREEN_COLOR
YELLOW_COLOR = run_all.YELLOW_COLOR

# --- Helper Functions ---

def parse_csv_line(csv_line, header_keys):
    """Parses a CSV line into a dictionary using the provided header keys."""
    if not csv_line:
        return None
    values = csv_line.strip().split(',')
    return dict(zip(header_keys, values))

def get_run_info(params_string):
    """Extracts key parameters from the command-line arguments string for reporting."""
    parts = []
    # Use regex to find key-value pairs for a concise summary
    automaton = re.search(r'--automaton\s+(\S+)', params_string)
    evaluator = re.search(r'--evaluator\s+(\S+)', params_string)
    layout = re.search(r'--layout\s+(\S+)', params_string)
    word_size = re.search(r'--word_size\s+(\S+)', params_string)
    temp_steps = re.search(r'--temporal_steps\s+(\S+)', params_string)
    
    if automaton: parts.append(f"Automaton: {automaton.group(1)}")
    if evaluator: parts.append(f"Eval: {evaluator.group(1)}")
    if layout: parts.append(f"Layout: {layout.group(1)}")
    if word_size: parts.append(f"Prec: {word_size.group(1)}")
    if temp_steps: parts.append(f"T-Steps: {temp_steps.group(1)}")

    return ", ".join(parts)


def run_and_validate(impl_name, params_str, executable, header_keys):
    """
    Runs a test implementation and a corresponding baseline, then compares their checksums.
    """
    summary = get_run_info(params_str)
    print(f"{GREY_COLOR}Testing: {impl_name:<25} | {summary}{RESET_COLOR}", end='', flush=True)

    # 1. Run the test implementation
    try:
        test_output = executable.run(params_str, throw_on_error=True)
    except Exception as e:
        exception_msg = str(e)
        if "The temporal tile is too large" in exception_msg:
            print(f"\r{YELLOW_COLOR}[ SKIP  ]{RESET_COLOR} {impl_name:<25} | {summary} -> Skipped (temporal tile too large).")
            return
        print(f"\r{RED_COLOR}[ ERROR ]{RESET_COLOR} {impl_name:<25} | {summary} -> Exception: {exception_msg}")
        return
    
    if not test_output:
        print(f"\r{RED_COLOR}[ ERROR ]{RESET_COLOR} {impl_name:<25} | {summary} -> Test implementation failed to run.")
        return

    test_results = parse_csv_line(test_output, header_keys)
    
    try:
        x_size = test_results['x_size']
        y_size = test_results['y_size']
        steps = test_results['steps']
        test_checksum = test_results['checksum']
    except (KeyError, TypeError):
        print(f"\r{RED_COLOR}[ ERROR ]{RESET_COLOR} {impl_name:<25} | {summary} -> Could not parse test output CSV.")
        return

    # 2. Construct and run the baseline implementation with identical parameters
    baseline_params_str = (
        f"--automaton {test_results['automaton']} --seed {run_all.SEED} --device {run_all.DEVICE} "
        f"--reference_impl baseline --x_size {x_size} --y_size {y_size} --steps {steps} "
        f"--rounds 1 --warmup_rounds 0" # Use minimal rounds for a quick baseline check
    )
    try:
        baseline_output = executable.run(baseline_params_str, throw_on_error=True)
    except:
        baseline_output = executable.run(baseline_params_str.replace('--device CUDA', '--device CPU'))
    
    if not baseline_output:
        print(f"\r{RED_COLOR}[ ERROR ]{RESET_COLOR} {impl_name:<25} | {summary} -> Baseline implementation failed to run.")
        return

    baseline_results = parse_csv_line(baseline_output, header_keys)
    
    try:
        baseline_checksum = baseline_results['checksum']
    except (KeyError, TypeError):
        print(f"\r{RED_COLOR}[ ERROR ]{RESET_COLOR} {impl_name:<25} | {summary} -> Could not parse baseline output CSV.")
        return

    # 3. Compare checksums and report the final result
    if test_checksum == baseline_checksum:
        print(f"\r{GREEN_COLOR}[  OK   ]{RESET_COLOR} {impl_name:<25} | {summary}")
    else:
        print(f"\r{RED_COLOR}[ FAIL  ]{RESET_COLOR} {impl_name:<25} | {summary}")
        print(f"          -> Test Checksum: {test_checksum}, Baseline Checksum: {baseline_checksum}")

# --- Main Script Logic ---

def main():
    executable = run_all.Executable()
    header = executable.get_csv_header()
    header_keys = header.strip().split(',')
    
    print(f"{YELLOW_COLOR}Starting correctness validation...{RESET_COLOR}")
    print("-" * 80)

    base_tc = run_all.TestCase().with_device(run_all.DEVICE).with_elem_count(GRID_SIZE).with_steps(STEPS)

    for automaton in AUTOMATA_TO_TEST:
        tc_automaton = base_tc.clone().with_automaton(automaton)
        print(f"\n{YELLOW_COLOR}--- Automaton: {automaton} ---{RESET_COLOR}")

        for y_block in Y_BLOCK_SIZES:
            tc_block = tc_automaton.clone().with_cuda_block_size_y(y_block)

            # Standard (non-bitpacked) implementation
            run_and_validate("Standard", run_all.StandardImplementation.params(tc_block), executable, header_keys)
            
            for word_size in WORD_SIZES:
                tc_prec = tc_block.clone().with_word_size(word_size)

                # Bit-packed implementations
                run_and_validate("BitArray", run_all.BitArrayImplementation.params(tc_prec), executable, header_keys)
                run_and_validate("BitPlanes", run_all.BitPlanesImplementation.params(tc_prec), executable, header_keys)
                run_and_validate("TiledBitPlanes", run_all.TiledBitPlanesImplementation.params(tc_prec), executable, header_keys)

                # Temporal implementations
                for temporal_steps in TEMPORAL_STEPS:
                    tc_tsteps = tc_prec.clone().with_temporal_steps(temporal_steps)

                    max_y_tile = run_all.biggest_temporal_tile_size_for_automata[word_size][automaton]
                    
                    if max_y_tile is None:
                        continue
                    
                    for temporal_tile_y in TEMPORAL_TILE_SIZES_Y:
                        if temporal_tile_y > max_y_tile or temporal_tile_y < y_block or temporal_tile_y % y_block != 0:
                            continue

                        tc_final = tc_tsteps.clone().with_temporal_tile_size_y(temporal_tile_y)
                        
                        try:
                            # Temporal Linear (BitPlanes layout)
                            _, eff_y = run_all.TemporalLinearImplementation.get_effective_xy_block_size(tc_final)
                            if eff_y > 0:
                                params = run_all.TemporalLinearImplementation.params(tc_final)
                                run_and_validate("TemporalLinear", params, executable, header_keys)
                                
                            # Temporal Tiled Bit Planes
                            _, eff_y = run_all.TemporalTiledBitPlanesImplementation.get_effective_xy_block_size(tc_final)
                            if eff_y > 0:
                                params = run_all.TemporalTiledBitPlanesImplementation.params(tc_final)
                                run_and_validate("TemporalTiledBitPlanes", params, executable, header_keys)
                        except Exception:
                            # Some parameter combinations are invalid and will throw an exception.
                            # This is expected, so we just skip them.
                            pass

    print("-" * 80)
    print(f"{GREEN_COLOR}Validation finished.{RESET_COLOR}")
    return 0

if __name__ == "__main__":
    sys.exit(main())