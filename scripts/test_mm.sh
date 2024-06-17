#!/bin/bash

data_name="tcga_ov_os"
cancer_type="TCGA_OV"
output_file="./scripts/errors/output_test_mm.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,cnv --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,cnv --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics cnv,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics cnv,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics mut,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna,cnv --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,cnv,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,cnv,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,mut,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,cnv,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,cnv,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,mut,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics cnv,mut,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna,cnv,mut --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna,cnv,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,dna,mut,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics rna,cnv,mut,pro --mlp_type small --model_type mlp"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/"$data_name" --data_name "$data_name" --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --separate_branches --fusion bilinear --omics dna,cnv,mut,pro --mlp_type small --model_type mlp"

python ./scripts/check_errors.py "$output_file"