#!/bin/bash

script_dir=$(dirname "$0")
base_dir=$script_dir/../../../results

CSVs=($base_dir/H100/_h100.csv $base_dir/A100/_a100.csv)
out_types=("png" "pdf")

for csv in "${CSVs[@]}"; do
 for out_type in "${out_types[@]}"; do
   out_file="${csv%.*}_total_perf.${out_type}"
   python $script_dir/0_total_perf.py      $csv $out_file
   
   out_file="${csv%.*}_linear_vs_tiled.${out_type}"
   python $script_dir/1_linear_vs_tiled.py $csv $out_file

   out_file="${csv%.*}_time_steps.${out_type}"
   python $script_dir/2_time_steps.py      $csv $out_file
 done
done