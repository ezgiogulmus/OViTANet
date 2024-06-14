import os
import pandas as pd
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import shap

parser.add_argument('data_name', type=str)
parser.add_argument('target_data', type=str)
parser.add_argument('--dataset_dir', type=str, default="./datasets_csv")
parser.add_argument('--split_dir', type=str, default='./splits')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--verbose', action="store_true", default=False)

args.dname = "tcga_cesc_os"

target_data = "mut"
cli_df = pd.read_csv(os.path.join(args.dataset_dir, f"{args.dname}.csv"))

cli_df = cli_df.drop_duplicates("case_id").drop(["slide_id", "group"], axis=1).reset_index(drop=True)
tests = pd.read_csv(os.path.join(args.split_dir, f"{args.dname}/splits_0.csv"))["test"].dropna().values
cli_df["split"] = [pd.NA] * len(cli_df)
cli_df.loc[cli_df["case_id"].isin(tests), "split"] = "test"
cli_df.loc[~cli_df["case_id"].isin(tests), "split"] = "train"

if target_data != "cli":
    target_df = pd.read_csv(os.path.join(args.dataset_dir, f"{args.dname}_{target_data}.csv.zip"), compression="zip")
    target_df = pd.merge(target_df, cli_df[["case_id", "split", "survival_months", "event"]], on="case_id")
else:
    target_df = cli_df

target_df.reset_index(drop=True, inplace=True)
target_df.shape, target_df.isna().any().any(), target_df["case_id"].duplicated().any(), target_df.columns.duplicated().any(), target_df.columns.isna().any()

def split_data(args, df):
    train_df = df[df["split"] != "test"].drop(["split"], axis=1)
    test_df = df[df["split"] == "test"].drop(["split"], axis=1)
    train_df = train_df.drop_duplicates("case_id").reset_index(drop=True)
    test_df = test_df.drop_duplicates("case_id").reset_index(drop=True)

    train_ids = train_df[["case_id"]]
    X_train = train_df.drop(["case_id", "event", "survival_months"], axis=1)
    y_train = train_df["survival_months"]
    y_train_event = train_df["event"]

    test_ids = test_df[["case_id"]]
    X_test = test_df.drop(["case_id", "event", "survival_months"], axis=1)
    y_test = test_df["survival_months"]
    y_test_event = test_df["event"]
    print(f"Train set shape: {X_train.shape}, test set shape: {X_test.shape}\n")
    print("Train survival times:")
    print(y_train.describe())
    print("\nTest survival times:")
    print(y_test.describe())
    return X_train, y_train, y_train_event