#!/bin/bash

data_name="tcga_ov_os"
cancer_type="TCGA_OV"
omics="cnv"
output_file="./scripts/errors/output_test.txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

# # Omic only
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --apply_sig --omics ${omics} --reg_type omic --model_type snn"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --omics ${omics} --reg_type omic --model_type snn"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --selected_features --omics ${omics} --mlp_type small --model_type mlp"

# # WSI only
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --model_type mil"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --model_type clam_mb"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --model_type clam_sb"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --model_type transmil"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --model_type porpamil"

# # Omic + WSI: bilinear fusion
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --selected_features --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --model_type amil"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --selected_features --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --model_type deepset"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --selected_features --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --reg_type pathomic --model_type porpmmf"

# # Omic + WSI: coattention all features
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --apply_sig --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --model_type mcat"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --apply_sig --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --bs_micro 16384 --ot_impl pot-uot-l2 --ot_reg 0.1 --ot_tau 0.5 --model_type motcat"
# # Omic + WSI: coattention selected features
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --selected_features --apply_sig --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --model_type mcat"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --omics ${omics} --selected_features --apply_sig --weighted_sample --max_epochs 1 --k 1 --early_stopping 20 --fusion bilinear --bs_micro 16384 --ot_impl pot-uot-l2 --ot_reg 0.1 --ot_tau 0.5 --model_type motcat"

# GMCAT
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location early --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location early --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location early --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location early --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"

# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location late --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location late --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location late --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location late --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"

# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location mid --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location mid --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location mid --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location mid --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion crossatt --fusion_location mid --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"

# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion bilinear --fusion_location ms --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion adaptive --fusion_location ms --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion multiply --fusion_location ms --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion concat --fusion_location ms --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"
# run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/test/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type gmcat --fusion crossatt --fusion_location ms --selected_features --omics ${omics} --weighted_sample --max_epochs 1 --k 1 --early_stopping 20"

python ./scripts/check_errors.py "$output_file"