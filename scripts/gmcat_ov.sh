#!/bin/bash
data_name="tcga_ov_os"
cancer_type="TCGA_OV"
omics="rna"
output_file="./scripts/error_gmcat_"$data_name".txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location early --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location early --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location early --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location early --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location late --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location late --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location late --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location late --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location mid --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location mid --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location mid --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location mid --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion crossatt --fusion_location mid --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location ms --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location ms --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location ms --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location ms --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion crossatt --fusion_location ms --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 30 --early_stopping 20"

python ./scripts/check_errors.py "$output_file"