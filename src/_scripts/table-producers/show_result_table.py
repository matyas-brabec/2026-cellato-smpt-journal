import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abstractions.table_printer import TablePrinter
from abstractions.results_abstractions import IMPLEMENTATIONS, BITS_USED, RunResult, CSVLoader

script_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    if len(sys.argv) < 2:
        print("Usage: python show_result_table.py <path_to_csv_file>")
        sys.exit(1)

    # Get CSV file path from command line or use default
    file = sys.argv[1]

    # Load and process data
    loader = CSVLoader(file)
    size_groups = loader.get_groups_by_sizes([x ** 2 for x in [4096, 8192, 16384, 32768]], tolerance=0.1)
    sorted_groups = sorted(size_groups.items())

    # Detect if colors are supported
    use_colors = True
    if os.name == 'nt' or 'NO_COLOR' in os.environ:
        use_colors = False
    
    # Define table columns (implementations to show)
    implementations = list(IMPLEMENTATIONS.keys())
    
    # Create one table per size group
    for size, group in sorted_groups:
        # Skip if no data for this size
        if not group:
            continue
            
        # Create table title
        title = f"CUDA Performance Comparison - {int(size**0.5)}×{int(size**0.5)} Grid"
        print(f"\n{TablePrinter.COLORS.YELLOW_r}{TablePrinter.COLORS.BOLD_r}{title}{TablePrinter.COLORS.RESET_r}\n" if use_colors else f"\n{title}\n")
        
        # Create table printer
        printer = TablePrinter()
        printer.set_use_colors(use_colors)
        
        # Create header row
        header = ["Automaton", "Baseline (ps)"]
        for impl in implementations[1:]:  # Skip baseline as it's already in the header
            header.append(impl)
            
        # Add colored header
        colored_header = []
        for item in header:
            colored_header.append(f"{TablePrinter.COLORS.CYAN}{item}{TablePrinter.COLORS.RESET}" if use_colors else item)
        
        printer.add_row(colored_header)
        
        # Process data by automaton
        automaton_groups = loader.split_by_automaton(group)
        for automaton, automaton_group in sorted(automaton_groups.items()):
            # Split by implementation type
            impl_groups = loader.split_by_implementation(automaton_group)
            
            # Find baseline implementation
            baseline_result = loader.find_best_implementation(impl_groups["Baseline"], IMPLEMENTATIONS["Baseline"])
            if not baseline_result:
                continue
                
            # Get baseline time
            baseline_time = baseline_result.normalized_time() * 1e9  # Convert to nanoseconds
            
            # Create row with automaton name and baseline time
            automaton_name = automaton
            if automaton in BITS_USED:
                automaton_name += BITS_USED[automaton]
                
            row = [automaton_name, f"{baseline_time:.4f} ps"]
            
            # Add data for each implementation
            for impl in implementations[1:]:  # Skip baseline
                best_result = loader.find_best_implementation(impl_groups[impl], IMPLEMENTATIONS[impl])
                if best_result:
                    time = best_result.normalized_time() * 1e9  # Convert to nanoseconds
                    speedup = baseline_time / time
                    
                    # Format speedup with color
                    color = TablePrinter.COLORS.GREEN if speedup > 1 else TablePrinter.COLORS.RED
                    speedup_str = f"{color}{speedup:.2f}x{TablePrinter.COLORS.RESET}" if use_colors else f"{speedup:.2f}x"
                    
                    row.append(f"{time:.4f} ps ({speedup_str})")
                else:
                    row.append("-")
            
            printer.add_row(row)
        
        # Print the table
        printer.print()


if __name__ == "__main__":
    main()


