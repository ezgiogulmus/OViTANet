#!/bin/bash

data_name="tcga_brca"
output_file="./scripts/errors/output_fseval_"$data_name".txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

# OS All
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics cnv --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics dna --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics mut --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics rna --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics pro --mlp_type big --max_epochs 20"

# OS Selected
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics rna --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics cnv --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics dna --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_os --data_name "$data_name"_os --omics pro --selected_features --mlp_type small --max_epochs 20"

# DFS All
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics cnv --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics mut --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna --mlp_type big --max_epochs 20"
# run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics pro --mlp_type big --max_epochs 20"

# DFS Selected
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics cnv --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval/"$data_name"_dfs --data_name "$data_name"_dfs --omics pro --selected_features --mlp_type small --max_epochs 20"

python ./scripts/check_errors.py "$output_file"