#!/bin/bash
data_name="tcga_ucec_os"
cancer_type="TCGA_UCEC"

output_file="./scripts/errors/out_"$data_name"_wsi.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}
# WSI only
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/wsi --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type mil --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping 20 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/wsi --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type clam_mb --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping 20 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/wsi --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type clam_sb --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping 20 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/wsi --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type transmil --gc 2 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping 10 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/"$data_name"/wsi --data_name "$data_name" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type porpamil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --lambda_reg 1e-5 --max_epochs 20 --early_stopping 20"