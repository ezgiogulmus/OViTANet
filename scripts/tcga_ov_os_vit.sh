#!/bin/bash
data_name="tcga_ov_os"
cancer_type="TCGA_OV"
omics="cnv"
output_file="./scripts/error_out_"$data_name"_vit.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}
# Omic only
run_command "CUDA_VISIBLE_DEVICES=1 python main.py --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --results_dir ./results/tcga_ov_os/dna/ --data_name tcga_ov_os --omics dna --selected --mm_fusion crossatt --mm_fusion_type mid --model_type vit"

run_command "CUDA_VISIBLE_DEVICES=1 python main.py --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --results_dir ./results/tcga_ov_os/rna/ --data_name tcga_ov_os --omics rna --selected --mm_fusion crossatt --mm_fusion_type mid --model_type vit"

run_command "CUDA_VISIBLE_DEVICES=1 python main.py --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --results_dir ./results/tcga_ov_os/mut/ --data_name tcga_ov_os --omics mut --selected --mm_fusion crossatt --mm_fusion_type mid --model_type vit"

run_command "CUDA_VISIBLE_DEVICES=1 python main.py --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --results_dir ./results/tcga_ov_os/cnv/ --data_name tcga_ov_os --omics cnv --selected --mm_fusion crossatt --mm_fusion_type mid --model_type vit"