#!/bin/bash

output_file="./scripts/errors/output_ov.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}
run_command "python fs.py tcga_ov_os cnv"
run_command "python fs.py tcga_ov_dfs cnv"

run_command "python fs.py tcga_ov_os mut"
run_command "python fs.py tcga_ov_dfs mut"

run_command "python fs.py tcga_ov_os pro"
run_command "python fs.py tcga_ov_dfs pro"

run_command "python fs.py tcga_ov_os dna"
run_command "python fs.py tcga_ov_dfs dna"

run_command "python fs.py tcga_ov_os rna"
run_command "python fs.py tcga_ov_dfs rna"
python ./scripts/check_errors.py "$output_file"