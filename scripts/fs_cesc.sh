#!/bin/bash

output_file="./scripts/errors/output_cesc.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

run_command "python fs.py tcga_cesc_os cnv"
run_command "python fs.py tcga_cesc_dfs cnv"

run_command "python fs.py tcga_cesc_os mut"
run_command "python fs.py tcga_cesc_dfs mut"

run_command "python fs.py tcga_cesc_os pro"
run_command "python fs.py tcga_cesc_dfs pro"

run_command "python fs.py tcga_cesc_os dna"
run_command "python fs.py tcga_cesc_dfs dna"

run_command "python fs.py tcga_cesc_os rna"
run_command "python fs.py tcga_cesc_dfs rna"
python ./scripts/check_errors.py "$output_file"