#!/bin/bash

output_file="./scripts/errors/output_brca.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

run_command "python fs.py tcga_brca_os dna"
run_command "python fs.py tcga_brca_dfs dna"

run_command "python fs.py tcga_brca_os rna"
run_command "python fs.py tcga_brca_dfs rna"

run_command "python fs.py tcga_brca_os cnv"
run_command "python fs.py tcga_brca_dfs cnv"

run_command "python fs.py tcga_brca_os mut"
run_command "python fs.py tcga_brca_dfs mut"

run_command "python fs.py tcga_brca_os pro"
run_command "python fs.py tcga_brca_dfs pro"

python ./scripts/check_errors.py "$output_file"