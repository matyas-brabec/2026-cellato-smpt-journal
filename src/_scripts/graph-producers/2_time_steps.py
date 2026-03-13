import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Loading ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from abstractions.results_abstractions import CSVLoader

AUTOMATA=[
    "game-of-life",
    "maze",
    "brian",
    "forest-fire",
    "wire",
    "excitable",
    "cyclic",
    "fluid",
    "critters",
    "traffic",
]

if len(sys.argv) > 2 and sys.argv[2].endswith(('.png', '.pdf')):
    path_to_csv = sys.argv[1]
    output_path = sys.argv[2]
else:
    raise Exception("Please provide the path to the CSV file and the output path as command-line arguments.")

size=16384
time_steps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 20, 22, 24]
loader = CSVLoader(path_to_csv)
size_group = loader.get_groups_by_sizes([size**2])[size**2]
print(f"Total results for size {size}x{size}: {len(size_group)}")

# 🆕 Define separate baselines
baseline_linear = {'traverser': 'simple', 'evaluator': 'bit_planes', 'layout': 'bit_planes'} 

baseline_tiled = {'traverser': 'simple', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes'}
baseline_labels = ['Single Step \nLinear Bit Planes (1x)', 'Single Step \nTiled Bit Planes (1x)']

# baseline_tiled = {'traverser': 'simple', 'evaluator': 'bit_planes', 'layout': 'bit_planes'}
# baseline_labels = ['Single Step Linear Bit Planes (1x)', 'Single Step Linear Bit Planes (1x)']

one_step_linear = {'traverser': 'simple', 'evaluator': 'bit_planes', 'layout': 'bit_planes'}
one_step_tiled = {'traverser': 'simple', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes'}

temporal_linear = {'traverser': 'temporal', 'evaluator': 'bit_planes', 'layout': 'bit_planes'}
temporal_tiled = {'traverser': 'temporal', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes'}

# 🆕 Create groups for each baseline
baseline_linear_group = [x for x in size_group if x.is_implementation(baseline_linear)]
baseline_tiled_group = [x for x in size_group if x.is_implementation(baseline_tiled)]

one_step_linear_group = [x for x in size_group if x.is_implementation(one_step_linear)]
one_step_tiled_group = [x for x in size_group if x.is_implementation(one_step_tiled)]
temporal_linear_group = [x for x in size_group if x.is_implementation(temporal_linear)]
temporal_tiled_group = [x for x in size_group if x.is_implementation(temporal_tiled)]

bests_by_automaton = {}
for automaton in AUTOMATA:
    # 🆕 Find best times for each baseline
    baseline_linear_for_automaton = [x for x in baseline_linear_group if x.values.get('automaton') == automaton]
    baseline_tiled_for_automaton = [x for x in baseline_tiled_group if x.values.get('automaton') == automaton]

    one_step_linear_for_automaton = [x for x in one_step_linear_group if x.values.get('automaton') == automaton]
    one_step_tiled_for_automaton = [x for x in one_step_tiled_group if x.values.get('automaton') == automaton]
    temporal_linear_for_automaton = [x for x in temporal_linear_group if x.values.get('automaton') == automaton]
    temporal_tiled_for_automaton = [x for x in temporal_tiled_group if x.values.get('automaton') == automaton]

    baseline_linear_best = min(baseline_linear_for_automaton, key=lambda x: x.normalized_time())
    baseline_tiled_best = min(baseline_tiled_for_automaton, key=lambda x: x.normalized_time())

    one_step_linear_best = min(one_step_linear_for_automaton, key=lambda x: x.normalized_time())
    one_step_tiled_best = min(one_step_tiled_for_automaton, key=lambda x: x.normalized_time())
    
    temporal_linear_bests = []
    temporal_tiled_bests = []

    for ts in time_steps:
        linear_results_for_ts = [x for x in temporal_linear_for_automaton if x.values.get('temporal_steps') == ts]
        tiled_results_for_ts = [x for x in temporal_tiled_for_automaton if x.values.get('temporal_steps') == ts]

        temporal_linear_bests.append(min(linear_results_for_ts, key=lambda x: x.normalized_time()) if linear_results_for_ts else None)
        temporal_tiled_bests.append(min(tiled_results_for_ts, key=lambda x: x.normalized_time()) if tiled_results_for_ts else None)

    bests_by_automaton[automaton] = {
        # 🆕 Store both baseline times
        'baseline_linear_time': baseline_linear_best.normalized_time() * 1e9,
        'baseline_tiled_time': baseline_tiled_best.normalized_time() * 1e9,
        'linear': [one_step_linear_best.normalized_time() * 1e9] + [(x.normalized_time() * 1e9 if x else None) for x in temporal_linear_bests],
        'tiled': [one_step_tiled_best.normalized_time() * 1e9] + [(x.normalized_time() * 1e9 if x else None) for x in temporal_tiled_bests]
    }
print("Finished processing data. Starting plot generation.")


# --- 2. Plotting Phase ---
automaton_names = {
    "game-of-life": "GoL", "forest-fire": "fire", "wire": "wire",
    "excitable": "excitable", "brian": "brian", "cyclic": "cyclic",
    "traffic": "traffic", "fluid": "fluid", "maze": "maze", "critters": "critters"
}

# ⚙️ Graph Configuration
scale = 0.6
plot_config = {
    'plot_mode': 'subplots',
    'y_axis_mode': 'speedup',
    'figure_size': (16*scale, 8*scale),
    'linear_color': '#08519c',
    'tiled_color': '#006d2c'
}

# --- Data Preparation ---
x_values = [1] + time_steps
plot_data = {}
y_axis_label = ""

if plot_config['y_axis_mode'] == 'speedup':
    y_axis_label = "Speedup vs. Simple (Non-Temporal) Version"
    for automaton, results in bests_by_automaton.items():
        # 🆕 Use the specific baseline for each implementation type
        linear_baseline = results['baseline_linear_time']
        tiled_baseline = results['baseline_tiled_time']
        
        plot_data[automaton] = {
            'linear': [linear_baseline / t if t is not None else None for t in results['linear']],
            'tiled': [tiled_baseline / t if t is not None else None for t in results['tiled']]
        }
elif plot_config['y_axis_mode'] == 'throughput':
    y_axis_label = "Throughput (Giga Cell Updates Per Second)"
    for automaton, results in bests_by_automaton.items():
        plot_data[automaton] = {
            'linear': [1000 / t if t is not None else None for t in results['linear']],
            'tiled': [1000 / t if t is not None else None for t in results['tiled']]
        }

# Helper function to handle incomplete data for plotting
def plot_incomplete_line(ax, x_data, y_data, **kwargs):
    segments = []
    current_x, current_y = [], []
    for x, y in zip(x_data, y_data):
        if y is not None:
            current_x.append(x)
            current_y.append(y)
        elif current_x:
            segments.append((current_x, current_y))
            current_x, current_y = [], []
    if current_x:
        segments.append((current_x, current_y))
    
    is_first_segment = True
    for x_seg, y_seg in segments:
        label = kwargs.get('label') if is_first_segment else ""
        ax.plot(x_seg, y_seg, marker='.', **{k: v for k, v in kwargs.items() if k != 'label'}, label=label)
        is_first_segment = False

# --- Plotting Logic ---
if plot_config['plot_mode'] == 'combined':
    fig, ax = plt.subplots(figsize=plot_config['figure_size'])
    # ax.set_title('Overall Effect of Temporal Blocking')

    is_first_linear, is_first_tiled = True, True
    for automaton, y_values in plot_data.items():
        plot_incomplete_line(ax, x_values, y_values['linear'], color=plot_config['linear_color'], alpha=0.5, label='Linear' if is_first_linear else "")
        is_first_linear = False
        plot_incomplete_line(ax, x_values, y_values['tiled'], color=plot_config['tiled_color'], alpha=0.5, label='Tiled' if is_first_tiled else "")
        is_first_tiled = False

    ax.set_xlabel("Temporal Steps")
    ax.set_ylabel(y_axis_label)
    ax.set_xticks(x_values)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    plt.savefig('temporal_scaling_combined.png', dpi=300)
    print("Combined graph saved as temporal_scaling_combined.png")

elif plot_config['plot_mode'] == 'subplots':
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=plot_config['figure_size'], sharey=True)
    # fig.suptitle('Effect of Temporal Blocking by Automaton')

    colors = plt.get_cmap('tab10', len(AUTOMATA))
    linestyles = ['-', '--', ':', '-.']
    
    automaton_styles = {}
    for i, name in enumerate(AUTOMATA):
        automaton_styles[name] = {
            'color': colors(i),
            'linestyle': linestyles[i % len(linestyles)]
        }

    # Plot 1: Linear Implementations
    ax1.set_title('Linear Implementations')
    for automaton, y_values in plot_data.items():
        style = automaton_styles[automaton]
        plot_incomplete_line(ax1, x_values, y_values['linear'], 
                             label=automaton_names[automaton], 
                             color=style['color'], 
                             linestyle=style['linestyle'])

    # Plot 2: Tiled Implementations
    ax2.set_title('Tiled Implementations')
    for automaton, y_values in plot_data.items():
        style = automaton_styles[automaton]
        plot_incomplete_line(ax2, x_values, y_values['tiled'], 
                             label=automaton_names[automaton], 
                             color=style['color'], 
                             linestyle=style['linestyle'])

# Common styling for both subplots
    for i, ax in enumerate([ax1, ax2]):
        ax.set_xlabel("Temporal Steps")
        ax.set_xticks(x_values)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if plot_config['y_axis_mode'] == 'speedup':
            ax.axhline(y=1, color='red', linestyle='--', linewidth=1.2)
            
            # Define properties for the text's background box
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)

            # Place the text at the left, below the line, with the background
            ax.text(x=x_values[0] + 0.1,  # Position slightly right of the y-axis
                    y=0.85,                 # Position just below the line (y=1)
                    s=baseline_labels[i],
                    color='red',
                    ha='left',              # Horizontally align to the left
                    va='top',               # Vertically align to the top
                    fontsize=9,
                    bbox=bbox_props)        # Apply the background box


            
    ax1.set_ylabel(y_axis_label)
    ax1.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Subplots graph saved as {output_path}")