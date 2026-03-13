#!/bin/bash

script_dir=$(dirname "$0")
base_dir=$script_dir/../../../results

CSVs=($base_dir/H100/_h100.csv $base_dir/A100/_a100.csv)

python $script_dir/to_html_report.py $base_dir/A100/_a100.csv > $base_dir/A100/a100_report.html
python $script_dir/to_html_report.py $base_dir/H100/_h100.csv > $base_dir/H100/h100_report.html

python $script_dir/to_html_report.py $base_dir/H100/grid-search-results.csv > $base_dir/H100/h100_report_initial_grid_search.html

