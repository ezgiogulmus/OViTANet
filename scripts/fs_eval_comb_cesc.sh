#!/bin/bash

data_name="tcga_cesc"
output_file="./scripts/errors/output_fsevalcombo_"$data_name".txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

# OS
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,dna --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics dna,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics dna,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,dna,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,dna,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics dna,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_os --data_name "$data_name"_os --omics rna,dna,mut,pro --selected_features --mlp_type small --max_epochs 20"

# DFS
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,cnv --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,cnv --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics cnv,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics cnv,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna,cnv --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,cnv,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,cnv,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,cnv,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,cnv,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics cnv,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna,cnv,mut --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna,cnv,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,dna,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics rna,cnv,mut,pro --selected_features --mlp_type small --max_epochs 20"
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --gc 32 --model_type mlp --early_stopping -1 --results_dir ./results/fs_eval_comb/"$data_name"_dfs --data_name "$data_name"_dfs --omics dna,cnv,mut,pro --selected_features --mlp_type small --max_epochs 20"
python ./scripts/check_errors.py "$output_file"