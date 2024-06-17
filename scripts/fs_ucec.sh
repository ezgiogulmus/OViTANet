#!/bin/bash

output_file="./scripts/errors/output_ucec.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

run_command "python fs.py tcga_ucec_os dna"
run_command "python fs.py tcga_ucec_dfs dna"

run_command "python fs.py tcga_ucec_os rna"
run_command "python fs.py tcga_ucec_dfs rna"

run_command "python fs.py tcga_ucec_os cnv"
run_command "python fs.py tcga_ucec_dfs cnv"

run_command "python fs.py tcga_ucec_os mut"
run_command "python fs.py tcga_ucec_dfs mut"

run_command "python fs.py tcga_ucec_os pro"
run_command "python fs.py tcga_ucec_dfs pro"

python ./scripts/check_errors.py "$output_file"