#!/bin/bash
cd /media/nfs/Desktop/OViTANet
data_name="tcga_ucec"
cancer_type="TCGA_UCEC"

output_file="./scripts/errors/out_encoders_"$data_name".txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

surv_type="os"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/PLIP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CTP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/RESNET50/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/SSL/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "

surv_type="dfs"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/PLIP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CTP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/RESNET50/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/SSL/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
python ./scripts/check_errors.py "$output_file"