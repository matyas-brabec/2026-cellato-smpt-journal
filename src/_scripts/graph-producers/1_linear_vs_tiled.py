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
loader = CSVLoader(path_to_csv)
size_group = loader.get_groups_by_sizes([size**2])[size**2]
print(f"Total results for size {size}x{size}: {len(size_group)}")

# --- Implementation definitions ---
bit_planes_linear_impl = {'traverser': 'simple', 'evaluator': 'bit_planes', 'layout': 'bit_planes'}
bit_planes_tiled_impl = {'traverser': 'simple', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes'}
temporal_linear_impl = {'traverser': 'temporal', 'evaluator': 'bit_planes', 'layout': 'bit_planes'}
temporal_tiled_impl = {'traverser': 'temporal', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes'}

# --- Group results ---
bit_planes_linear_group = [x for x in size_group if x.is_implementation(bit_planes_linear_impl)]
bit_planes_tiled_group = [x for x in size_group if x.is_implementation(bit_planes_tiled_impl)]
temporal_linear_group = [x for x in size_group if x.is_implementation(temporal_linear_impl)]
temporal_tiled_group = [x for x in size_group if x.is_implementation(temporal_tiled_impl)]

# --- Find the best time for each implementation ---
bests_by_automaton = {}
for automaton in AUTOMATA:
    bpl_best = min([x for x in bit_planes_linear_group if x.values.get('automaton') == automaton], key=lambda x: x.normalized_time())
    bpt_best = min([x for x in bit_planes_tiled_group if x.values.get('automaton') == automaton], key=lambda x: x.normalized_time())
    tl_best = min([x for x in temporal_linear_group if x.values.get('automaton') == automaton], key=lambda x: x.normalized_time())
    tt_best = min([x for x in temporal_tiled_group if x.values.get('automaton') == automaton], key=lambda x: x.normalized_time())

    bests_by_automaton[automaton] = {
        'bit_planes_linear': bpl_best.normalized_time() * 1e9,
        'bit_planes_tiled': bpt_best.normalized_time() * 1e9,
        'temporal_linear': tl_best.normalized_time() * 1e9,
        'temporal_tiled': tt_best.normalized_time() * 1e9
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
    'y_axis_scale': 'linear',
    'figure_size': (16*scale, 6*scale),
    'bar_width': 0.18,
    'group_gap': 0.02,
    'title': f'Linear vs. Tiled Throughput for {size}x{size} Grid', # 🆕 Updated Title
    'add_data_labels': False, # 🆕 Labels are now turned off
    'custom_colors': {
        'bit_planes_linear': '#6baed6', 'bit_planes_tiled': '#08519c',
        'temporal_linear': '#74c476', 'temporal_tiled': '#006d2c'
    },
    'bar_hatches': {
        'bit_planes_linear': '/', 'bit_planes_tiled': '//',
        'temporal_linear': 'x', 'temporal_tiled': 'xx'
    }
}

# --- Data Preparation ---
labels = [automaton_names.get(a, a) for a in AUTOMATA]
implementations = ['bit_planes_linear', 'bit_planes_tiled', 'temporal_linear', 'temporal_tiled']
impl_display_names = {
    'bit_planes_linear': 'Bit Planes (Linear)', 'bit_planes_tiled': 'Bit Planes (Tiled)',
    'temporal_linear': 'Temporal (Linear)', 'temporal_tiled': 'Temporal (Tiled)'
}
data = {impl: [] for impl in implementations}

# 🆕 Y-axis is now throughput. The label and data calculation are changed.
y_axis_label = "Throughput (10$^{12}$ Cell Updates Per Second)"
for automaton in AUTOMATA:
    for impl in implementations:
        time_in_ps = bests_by_automaton[automaton][impl]
        # 🆕 Convert time to throughput for the bar heights
        throughput_tcups = 1 / time_in_ps if time_in_ps > 0 else 0
        data[impl].append(throughput_tcups)

# --- Plotting ---
fig, ax = plt.subplots(figsize=plot_config['figure_size'])
x = np.arange(len(labels))
width = plot_config['bar_width']
gap = plot_config['group_gap']
offsets = {
    'bit_planes_linear': -1.5*width - gap, 'bit_planes_tiled': -0.5*width - gap,
    'temporal_linear': 0.5*width + gap, 'temporal_tiled': 1.5*width + gap
}

for impl in implementations:
    color = plot_config['custom_colors'].get(impl)
    hatch = plot_config['bar_hatches'].get(impl)
    ax.bar(x + offsets[impl], data[impl], width, label=impl_display_names[impl], color=color, hatch=hatch, edgecolor='white')

# --- Styling and Customization ---
ax.set_ylabel(y_axis_label)
# ax.set_title(plot_config['title'])
ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_xticklabels(labels)
ax.set_yscale(plot_config['y_axis_scale'])
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 🆕 The entire data label block is removed.

# 🆕 Set the bottom of the graph to 0 and add a top margin
ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

ax.legend(loc='upper left')
fig.tight_layout()

# --- Saving ---
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved as {output_path}")