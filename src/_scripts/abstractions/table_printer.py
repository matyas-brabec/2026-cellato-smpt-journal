import re


class TablePrinter:

    class COLORS:
        YELLOW = '<<YELLOW>>'
        CYAN = '<<CYAN>>'
        GREEN = '<<GREEN>>'
        RED = '<<RED>>'
        RESET = '<<RESET>>'
        BOLD = '<<BOLD>>'

        YELLOW_r = '\033[93m'
        CYAN_r = '\033[96m'
        GREEN_r = '\033[92m'
        RED_r = '\033[91m'
        RESET_r = '\033[0m'
        BOLD_r = '\033[1m'

    def __init__(self):
        self.rows = []
        self.use_colors = True

    def set_use_colors(self, use_colors):
        self.use_colors = use_colors

    def add_row(self, row):
        self.rows.append([str(item) for item in row])

    def print(self):
        if not self.rows:
            print("No data to display.")
            return

        # Calculate column widths
        col_widths = []
        for i in range(len(self.rows[0])):
            max_width = 0
            for row in self.rows:
                if i < len(row):  # Make sure the index is valid
                    width = self.len_without_color(row[i])
                    max_width = max(max_width, width)
            col_widths.append(max_width)

        # Print header
        header = self.rows[0]
        header_line = " | ".join(self.format_cell(header[i], col_widths[i]) for i in range(len(header)))
        print(self.colorize(header_line) if self.use_colors else header_line)
        
        # Print separator
        print("-" * self.len_without_color(header_line))

        # Print rows
        for row in self.rows[1:]:
            row_line = " | ".join(self.format_cell(row[i], col_widths[i]) for i in range(len(row)))
            print(self.colorize(row_line) if self.use_colors else row_line)

    def format_cell(self, cell, width):
        # Strip color codes for width calculation
        clean_text = re.sub(r'<<.*?>>', '', cell)
        # Left-align the text with correct width
        return f"{cell:<{width + (len(cell) - len(clean_text))}}"

    def len_without_color(self, text):
        clean_text = re.sub(r'<<.*?>>', '', str(text))
        return len(clean_text)
    
    def colorize(self, text):
        colors = {
            TablePrinter.COLORS.YELLOW: TablePrinter.COLORS.YELLOW_r,
            TablePrinter.COLORS.CYAN: TablePrinter.COLORS.CYAN_r,
            TablePrinter.COLORS.GREEN: TablePrinter.COLORS.GREEN_r,
            TablePrinter.COLORS.RED: TablePrinter.COLORS.RED_r,
            TablePrinter.COLORS.RESET: TablePrinter.COLORS.RESET_r,
            TablePrinter.COLORS.BOLD: TablePrinter.COLORS.BOLD_r,
        }

        for placeholder, ansi_code in colors.items():
            text = text.replace(placeholder, ansi_code)
        return text
