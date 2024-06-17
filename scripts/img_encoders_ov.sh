#!/bin/bash
cd /media/nfs/Desktop/OViTANet
data_name="tcga_ov"
cancer_type="TCGA_OV"

output_file="./scripts/errors/out_encoders_"$data_name".txt"

> "$output_file"

run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

surv_type="os"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_UNI_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/CONCH/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_CONCH_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/PLIP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/PLIP/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_PLIP_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CTP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/CTP/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_CTP_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/RESNET50/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/RESNET50/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_RESNET50_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/SSL/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/SSL/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_SSL_path/run/"

surv_type="dfs"
run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/UNI/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_UNI_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CONCH/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/CONCH/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_CONCH_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/PLIP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/PLIP/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_PLIP_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/CTP/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/CTP/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_CTP_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/RESNET50/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/RESNET50/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_RESNET50_path/run/"

run_command "CUDA_VISIBLE_DEVICES=0 python main.py --results_dir results/img_encoders_mil/"$data_name"_"$surv_type" --data_name "$data_name"_"$surv_type" --feats_dir /media/nfs/SURV/"$cancer_type"/Feats1024/SSL/ --model_type mil --weighted_sample --gc 32 --lr 2e-4 --reg 1e-5 --max_epochs 20 --early_stopping -1 --drop_out 0.25 --inst_loss svm "
run_command "CUDA_VISIBLE_DEVICES=0 python eval.py --data_name baskent_"$surv_type" --feats_dir /media/nfs/SURV/BasOVER/Feats1024/SSL/ --load_from ./results/img_encoders_mil/tcga_ov_"$surv_type"/MIL_SSL_path/run/"
python ./scripts/check_errors.py "$output_file"