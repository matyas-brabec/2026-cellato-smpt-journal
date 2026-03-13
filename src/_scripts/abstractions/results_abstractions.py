
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from table_printer import TablePrinter

IMPLEMENTATIONS = {
    "Baseline": {'reference_impl': 'baseline'},
    "Standard": {'traverser': 'simple', 'evaluator': 'standard', 'layout': 'standard'},
    "Bit Array (32-bit)": {'traverser': 'simple', 'evaluator': 'bit_array', 'layout': 'bit_array', 'word_size': 32},
    "Bit Array (64-bit)": {'traverser': 'simple', 'evaluator': 'bit_array', 'layout': 'bit_array', 'word_size': 64},
    "Bit Planes (32-bit)": {'traverser': 'simple', 'evaluator': 'bit_planes', 'layout': 'bit_planes', 'word_size': 32},
    "Bit Planes (64-bit)": {'traverser': 'simple', 'evaluator': 'bit_planes', 'layout': 'bit_planes', 'word_size': 64},
    "Tiled BP (32-bit)": {'traverser': 'simple', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes', 'word_size': 32},
    "Tiled BP (64-bit)": {'traverser': 'simple', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes', 'word_size': 64},
    "Temporal Linear (32-bit)": {'traverser': 'temporal', 'evaluator': 'bit_planes', 'layout': 'bit_planes', 'word_size': 32},
    "Temporal Linear (64-bit)": {'traverser': 'temporal', 'evaluator': 'bit_planes', 'layout': 'bit_planes', 'word_size': 64},
    "Temporal Tiled (32-bit)": {'traverser': 'temporal', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes', 'word_size': 32},
    "Temporal Tiled (64-bit)": {'traverser': 'temporal', 'evaluator': 'tiled_bit_planes', 'layout': 'tiled_bit_planes', 'word_size': 64},
}

# These are the bits used by each automaton - for display purposes
BITS_USED = {
    "game-of-life": f"       {TablePrinter.COLORS.YELLOW}1 bit{TablePrinter.COLORS.RESET}",
    "forest-fire": f"        {TablePrinter.COLORS.YELLOW}2 bits{TablePrinter.COLORS.RESET}",
    "wire": f"               {TablePrinter.COLORS.YELLOW}2 bits{TablePrinter.COLORS.RESET}",
    "excitable": f"          {TablePrinter.COLORS.YELLOW}3 bits{TablePrinter.COLORS.RESET}",
    "brian": f"              {TablePrinter.COLORS.YELLOW}2 bit{TablePrinter.COLORS.RESET}",
    "cyclic": f"             {TablePrinter.COLORS.YELLOW}5 bits{TablePrinter.COLORS.RESET}",
    "traffic": f"            {TablePrinter.COLORS.YELLOW}2 bits{TablePrinter.COLORS.RESET}",
    "fluid": f"              {TablePrinter.COLORS.YELLOW}4 bits{TablePrinter.COLORS.RESET}",
    "maze": f"               {TablePrinter.COLORS.YELLOW}1 bit{TablePrinter.COLORS.RESET}",
    "critters": f"           {TablePrinter.COLORS.YELLOW}1 bit{TablePrinter.COLORS.RESET}",
}

AUTOMATA = list(BITS_USED.keys())

class RunResult: 
    def __init__(self, csv_header, csv_line):
        self.values = {}
        headers = csv_header.strip().split(',')
        line_values = csv_line.strip().split(',')
        for header, value in zip(headers, line_values):
            if value.replace('.', '', 1).isdigit():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            self.values[header] = value

    def is_implementation(self, impl_dict):
        for key, val in impl_dict.items():
            if key not in self.values or str(self.values[key]) != str(val):
                return False
        return True
    
    def falls_roughly_to_size(self, total_elements, tolerance=0.1):
        actual_elements = self.values.get('x_size', 1) * self.values.get('y_size', 1)
        return abs(actual_elements - total_elements) <= tolerance * total_elements

    def normalized_time(self):
        average_time = self.values.get('average_time_ms')
        elem_count = self.values.get('x_size', 1) * self.values.get('y_size', 1)
        steps = self.values.get('steps', 1)

        time_per_step_per_elem = average_time / (steps * elem_count)
        return time_per_step_per_elem


class CSVLoader:
    def __init__(self, csv_path):
        self.results = []
        with open(csv_path, 'r') as f:
            header = f.readline()
            for line in f:
                if line.strip():
                    self.results.append(RunResult(header, line))

    def get_groups_by_sizes(self, sizes, tolerance=0.1):
        groups = {size: [] for size in sizes}
        for result in self.results:
            for size in sizes:
                if result.falls_roughly_to_size(size, tolerance):
                    groups[size].append(result)
                    break
        return groups
    
    def split_by_implementation(self, group):
        impl_groups = {key: [] for key in IMPLEMENTATIONS.keys()}
        for result in group:
            for impl_key, impl_dict in IMPLEMENTATIONS.items():
                if result.is_implementation(impl_dict):
                    impl_groups[impl_key].append(result)
                    break
        return impl_groups
    
    def split_by_automaton(self, group):
        automaton_groups = {}
        for result in group:
            automaton = result.values.get('automaton', 'unknown')
            if automaton not in automaton_groups:
                automaton_groups[automaton] = []
            automaton_groups[automaton].append(result)
        return automaton_groups

    def find_best_implementation(self, group, impl_dict):
        best_result = None
        best_time = float('inf')
        for result in group:
            if result.is_implementation(impl_dict):
                norm_time = result.normalized_time()
                if norm_time < best_time:
                    best_time = norm_time
                    best_result = result
        return best_result
    
    def get_sorted_by_best_time(self, group, impl_dict):
        filtered = [r for r in group if r.is_implementation(impl_dict)]
        return sorted(filtered, key=lambda r: r.normalized_time())
