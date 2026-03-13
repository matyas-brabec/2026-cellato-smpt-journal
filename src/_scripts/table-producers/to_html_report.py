import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abstractions.table_printer import TablePrinter
from abstractions.results_abstractions import CSVLoader, IMPLEMENTATIONS, BITS_USED

script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    print("Usage: python to_html_report.py <path_to_csv_file>")
    sys.exit(1)

FILE_WITH_DATA_PATH = sys.argv[1]

def generate_html_table(size, automaton_groups, loader):
    title = f"CUDA Performance Comparison - {int(size**0.5)}x{int(size**0.5)} Grid"
    html = f"<h2>{title}</h2>"
    html += "<table>"
    
    # Header row
    implementations = list(IMPLEMENTATIONS.keys())
    html += "<tr>"
    html += f"<th>Automaton</th>"
    html += f"<th>Baseline (ps)</th>"
    for impl in implementations[1:]:
        html += f"<th>{impl}</th>"
    html += "</tr>"
    
    # Data rows
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
            bits_info = BITS_USED[automaton].strip().replace(TablePrinter.COLORS.YELLOW, '').replace(TablePrinter.COLORS.RESET, '')
            automaton_name = f"{automaton} {bits_info}"
        
        # Get baseline params
        baseline_params = format_params(baseline_result.values)
        
        html += "<tr>"
        html += f"<td>{automaton_name}</td>"
        html += f'<td class="tooltip" onclick="showDetailedRuns(\'{automaton}\', \'Baseline\', {size})">{baseline_time:.4f} ps<span class="tooltip-text">{baseline_params}</span></td>'
        
        # Collect performance data for ranking
        performance_data = []
        for impl in implementations[1:]:  # Skip baseline
            best_result = loader.find_best_implementation(impl_groups[impl], IMPLEMENTATIONS[impl])
            if best_result:
                time = best_result.normalized_time() * 1e9  # Convert to nanoseconds
                speedup = baseline_time / time
                performance_data.append((impl, time, speedup, best_result))
            else:
                performance_data.append((impl, None, None, None))
        
        # Sort implementations by speedup (descending)
        ranked_implementations = sorted(
            [data for data in performance_data if data[2] is not None],
            key=lambda x: x[2],
            reverse=True
        )
        
        # Assign rankings for top 3
        impl_rankings = {}
        for i, (impl, _, _, _) in enumerate(ranked_implementations[:3]):
            if i == 0:
                impl_rankings[impl] = "best-impl"
            elif i == 1:
                impl_rankings[impl] = "second-best-impl"
            elif i == 2:
                impl_rankings[impl] = "third-best-impl"
        
        # Add data for each implementation with proper ranking class
        for impl, time, speedup, best_result in performance_data:
            ranking_class = impl_rankings.get(impl, "")
            
            if best_result:
                # Format speedup with color
                speedup_class = "speedup-positive" if speedup > 1 else "speedup-negative"
                speedup_html = f'<span class="{speedup_class}">{speedup:.2f}x</span>'
                
                # Get parameters
                params = format_params(best_result.values)
                
                html += f'<td class="tooltip {ranking_class}" onclick="showDetailedRuns(\'{automaton}\', \'{impl}\', {size})">{time:.4f} ps ({speedup_html})<span class="tooltip-text">{params}</span></td>'
            else:
                html += "<td>-</td>"
        
        html += "</tr>"
    
    html += "</table>"
    return html

def format_params(params_dict):
    """Format hyperparameters for tooltip display, focusing on grid search parameters"""
    result = "<strong>Optimized Hyperparameters:</strong><br>"
    
    # Check for CUDA block size y - relevant for all CUDA implementations
    if 'cuda_block_size_y' in params_dict:
        result += f"CUDA block size y: {params_dict['cuda_block_size_y']}<br>"
    
    # Check if this is a temporal implementation
    is_temporal = params_dict.get('traverser', '').startswith('temporal') or \
                 params_dict.get('traverser', '').endswith('_temporal')
    
    # Show temporal parameters only for temporal implementations
    if is_temporal:
        if 'temporal_steps' in params_dict:
            result += f"Temporal steps: {params_dict['temporal_steps']}<br>"
        if 'temporal_tile_size_y' in params_dict:
            result += f"Temporal tile y: {params_dict['temporal_tile_size_y']}<br>"
    
    # If nothing was found, add a note
    if result == "<strong>Optimized Hyperparameters:</strong><br>":
        result += "No grid search parameters available for this implementation"
        
    return result

def generate_detailed_runs_tables(size_groups, loader):
    """Generate hidden tables with all test runs for each automaton and implementation"""
    html = "<div id='detailedRunsContainer' class='modal'>"
    html += "<div class='modal-content'>"
    html += "<span class='close-button' onclick='closeDetailedRuns()'>&times;</span>"
    html += "<h3 id='detailedRunsTitle'>Detailed Test Runs</h3>"
    html += "<div id='detailedRunsContent'>"
    
    # For each size group
    for size, group in sorted(size_groups.items()):
        if not group:
            continue
        
        automaton_groups = loader.split_by_implementation(group)
        
        # For each automaton
        for automaton, automaton_group in sorted(loader.split_by_automaton(group).items()):
            impl_groups = loader.split_by_implementation(automaton_group)
            
            # For each implementation
            for impl, impl_group in sorted(impl_groups.items()):
                if not impl_group:
                    continue
                    
                # Generate a unique ID for this table
                table_id = f"detailed-{size}-{automaton}-{impl}".replace(" ", "_")
                
                # Sort runs by normalized time
                sorted_runs = sorted(impl_group, key=lambda r: r.normalized_time())
                
                # Start the table
                html += f"<table id='{table_id}' class='detailed-runs-table' data-automaton='{automaton}' data-implementation='{impl}' data-size='{size}' style='display:none;'>"
                html += "<thead><tr><th>Time (ps)</th>"
                
                # Find all unique parameters to display
                relevant_params = set()
                for run in sorted_runs:
                    if 'cuda_block_size_y' in run.values:
                        relevant_params.add('cuda_block_size_y')
                    
                    # Check if this is a temporal implementation
                    is_temporal = run.values.get('traverser', '').startswith('temporal') or \
                                run.values.get('traverser', '').endswith('_temporal')
                    
                    if is_temporal:
                        if 'temporal_steps' in run.values:
                            relevant_params.add('temporal_steps')
                        if 'temporal_tile_size_y' in run.values:
                            relevant_params.add('temporal_tile_size_y')
                
                # Add headers for each parameter
                for param in sorted(relevant_params):
                    param_display = param.replace('cuda_block_size_y', 'Block Y').replace('temporal_steps', 'Temp Steps').replace('temporal_tile_size_y', 'Temp Tile Y')
                    html += f"<th>{param_display}</th>"
                
                html += "</tr></thead><tbody>"

                best_time_ns = sorted_runs[0].normalized_time() * 1e9
                # Add rows for each run
                for run in sorted_runs:
                    time_ns = run.normalized_time() * 1e9
                    html += f"<tr><td>{time_ns:.4f} <span style='color: gray;'>({time_ns / best_time_ns:.2f}x)</span></td>"
                    
                    # Add values for each parameter
                    for param in sorted(relevant_params):
                        value = run.values.get(param, "-")
                        html += f"<td>{value}</td>"
                    
                    html += "</tr>"
                
                html += "</tbody></table>"
    
    html += "</div></div></div>"
    return html

def generate_report(csv_path, output_writer):
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), 'report-template.html')
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Load and process data
    loader = CSVLoader(csv_path)
    size_groups = loader.get_groups_by_sizes([x ** 2 for x in [4096, 8192, 16384, 32768]], tolerance=0.1)
    sorted_groups = sorted(size_groups.items())
    
    # Generate HTML tables
    tables_html = ""
    for size, group in sorted_groups:
        if not group:
            continue
        automaton_groups = loader.split_by_automaton(group)
        tables_html += generate_html_table(size, automaton_groups, loader)
    
    # Generate detailed runs tables
    detailed_runs_html = generate_detailed_runs_tables(size_groups, loader)
    
    # Replace placeholders with content
    html_content = template.replace("<!-- TABLES_PLACEHOLDER -->", tables_html)
    html_content = html_content.replace("<!-- DETAILED_RUNS_PLACEHOLDER -->", detailed_runs_html)
    
    # Write to file
    output_writer.write(html_content)
    
    print(f"HTML report generated", file=sys.stderr)

def main():
    # Get CSV file path from command line or use default
    csv_file = sys.argv[1] if len(sys.argv) > 1 else FILE_WITH_DATA_PATH
    
    # Get output file path or use default
    output_file = sys.stdout
    
    generate_report(csv_file, output_file)

if __name__ == "__main__":
    main()
