# OViTANet: Leveraging Cross-Attention and Multi-Modal Data Fusion for Improved Ovarian Cancer Prognostics
Introducing OViTANet, the Ovarian Cancer Vision Transformer Aggregated Network, a novel deep learning framework designed to enhance prognosis accuracy in serous ovarian cancer. This innovative model integrates histopathological imagery with a comprehensive set of clinical and genetic information. It leverages data from the Cancer Genome Archive (TCGA), including copy-number variations, DNA methylation patterns, mutational data, and gene expression profiles at both mRNA and protein levels. We experimented with three different image encoders to extract patch-level features from whole slide images (WSI), using a transformer-based architecture for synthesizing these features into patient-level insights. The model uses a discrete-time survival analysis and a cross-attention mechanism was integrated for effective multi-modal data fusion. It also offers various other data integration strategies, such as concatenation and multiplication, to accommodate diverse analytical needs.
![OViTANet Architecture Overview](arc.png "OViTANet Architecture")

# Installation Instructions

Whole slide images are accessible through the NIH Genomic Data Commons Data Portal [here](https://portal.gdc.cancer.gov/). Meanwhile, detailed clinical and genetic datasets can be explored via the cBioPortal [here](https://www.cbioportal.org/).

# Data Preparation

The widely recognized CLAM algorithm [available here](https://github.com/mahmoodlab/CLAM) was used for tissue segmentation from WSIs, facilitating the generation of 1024x1024 image patches. Subsequent feature extraction at the patch level was achieved through the application of:
1) ResNet-50, with ImageNet weights, detailed [here](https://pytorch.org/vision/stable/models.html#initializing-pre-trained-models).
2) CTransPath, pretrained on TCGA datasets, more information [here](https://github.com/Xiyue-Wang/TransPath).
3) HistoSSL, also pretrained on TCGA, with additional details [here](https://github.com/owkin/HistoSSLscaling).

# Train-Validation-Test Splits
Data was separated using the [cv_splits.ipynb](./cv_splits.ipynb) file to stratify patients according to survival distribution, ensuring the ratio of FFPE to frozen slides was preserved. The case IDs for each split are saved in the [splits folder](./splits/).

# Running Experiments
To specify the run configuration, adjust the following parameters:
- `run_name`: run
- `csv_path`: data_file.csv (This file should include `case_id`, `slide_id`, `event`, `survival_months`, and the clinical variables to be fused with the patch features.)
- `csv_dir`: /path/to/csv_file (default: `./datasets_csv/`)
- `data_root_dir`: /path/to/patch_features
- `split_dir`: /path/to/cv_splits (default: `./splits/`)
- `wandb`: False (The parameter dictionary at the end of [main.py](./main.py) should be edited for hyperparameter search.)

Default training configurations include:
- `optimizer`: Adam
- `lr`: 1e-3
- `reg`: 1e-2 (weight decay)
- `batch_size`: 1 (each patient is processed individually)
- `gc`: 64 (gradients are calculated for every 64 patients)
- `max_epochs`: 30
- `early_stopping`: 10
- `k`: 5 (cross-validation)
- `train_fraction`: 0.5 (percentage of patches used for training for each patient)
- `bootstrapping`: False (1000-iteration bootstrapping during testing)

Default model configurations include:
- `surv_model`: discrete (An MLP model is used for continuous-time analysis.)
- `drop_out`: 0.25
- `activation`: relu
- `mm_fusion`: choices=["crossatt", "concat", "adaptive", "multiply", "bilinear", None]
- `mm_fusion_type`: choices=["early", "mid", "late", None]
- `depth`: 5 (ViT depth)
- `mha_heads`: 4
- `dim_head`: 32

Example commands:
```
python main.py --run_name run01 --split_dir tcga_ov --csv_path tcga_ov --tabular_data dna --mm_fusion crossatt --mm_fusion_type mid
python main.py --run_name run02 --split_dir tcga_ov_dfs --csv_path tcga_ov_dfs --tabular_data pro,dna --mm_fusion crossatt --mm_fusion_type mid
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments

* The code is highly inspired by the [PORPOISE](https://github.com/mahmoodlab/PORPOISE) and [CLAM](https://github.com/mahmoodlab/CLAM) models.
* The ViT model is adapted from the extensive implementation by Phil Wang, available in the [lucidrains repo](https://github.com/lucidrains/vit-pytorch).
