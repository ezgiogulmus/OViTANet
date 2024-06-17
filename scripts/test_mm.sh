#!/bin/bash

data_name="tcga_ov_os"
cancer_type="TCGA_OV"
omics="cnv"
output_file="./scripts/error_output_test_mm.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

# # Omic only
# # SNN
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --model_type snn --apply_sig --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --reg_type omic"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --model_type snn --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --reg_type omic"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --model_type snn --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --reg_type omic"

# MLP
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --model_type mlp --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --reg_type omic"

# # WSI only
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type mil --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --drop_out 0.25 --inst_loss svm"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type clam_mb --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --drop_out 0.25 --inst_loss svm"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type clam_sb --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --drop_out 0.25 --inst_loss svm"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type transmil --gc 2 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 10 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type porpamil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --lambda_reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

# # Omic + WSI: bilinear fusion
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type porpmmf --fusion bilinear --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --lambda_reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --reg_type pathomic"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type amil --fusion bilinear --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type deepset --fusion bilinear --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

# # Omic + WSI: coattention
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type mcat --fusion bilinear --apply_sig --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type motcat --fusion bilinear --apply_sig --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --bs_micro 16384 --ot_impl pot-uot-l2 --ot_reg 0.1 --ot_tau 0.5"

# # Selected omics
# # Omic + WSI: bilinear fusion
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type porpmmf --fusion bilinear --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --lambda_reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --reg_type pathomic"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type amil --fusion bilinear --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type deepset --fusion bilinear --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

# # Omic + WSI: coattention
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type mcat --fusion bilinear --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type motcat --fusion bilinear --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 2 --k 1 --early_stopping 20 --bs_micro 16384 --ot_impl pot-uot-l2 --ot_reg 0.1 --ot_tau 0.5"

# GMCAT
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location early --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location early --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location early --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location early --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location late --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location late --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location late --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location late --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location mid --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location mid --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location mid --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location mid --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion crossatt --fusion_location mid --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location ms --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location ms --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location ms --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location ms --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion crossatt --fusion_location ms --apply_sig --selected_features --omics ${omics} --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20"

python ./scripts/check_errors.py "$output_file"