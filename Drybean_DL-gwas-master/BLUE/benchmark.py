from __future__ import print_function
import os

from keras.src.layers import Flatten
from pandas import read_csv

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
import numpy as np
import pandas as pd
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.layers import *
import keras
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from keras.models import load_model
import csv
import argparse
import subprocess
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math as m
import keras.backend as K

import rpy2
from rpy2.robjects.packages import importr
rrBLUP = importr("rrBLUP")
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from cnn_pipeline import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"  # New
tf.config.set_visible_devices([], 'GPU')  # New
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

##one of K encoding

nb_classes = 4

NUM_FOLDS = 5
NUM_REPEATS = 10

if __name__ == '__main__':

    # os.chdir("MOISTURE")

    parser = argparse.ArgumentParser()
    parser.add_argument('IMP_file', help="Imputed file")
    parser.add_argument('QA_file', help="QA file")
    parser.add_argument('--pheno', help="Phenotype file")
    parser.add_argument('--fold', type=int, default=None, help="Fold number (1–10)")
    parser.add_argument('--summary', action='store_true', help="Run saliency summary after all folds")
    parser.add_argument('--skip-cropformer', action='store_true', help="Skip Cropformer benchmark")
    parser.add_argument('--cropformer-epochs', type=int, default=100, help="Maximum Cropformer training epochs")
    parser.add_argument('--cropformer-max-features', type=int, default=10000, help="Cropformer SNP feature count after train-only selection/padding")
    args = parser.parse_args()

    IMP_input = args.IMP_file
    QA_input = args.QA_file

    # Data cleaning by file format to produce tsvs for the pipeline
    if IMP_input.endswith(".vcf"):
        IMP_input = vcf_preprocessing(IMP_input)
    elif IMP_input.endswith(".csv"):
        IMP_input = csv_preprocessing(IMP_input)

    if QA_input.endswith(".vcf"):
        QA_input = vcf_preprocessing(QA_input)
    elif QA_input.endswith(".csv"):
        QA_input = csv_preprocessing(QA_input)
    # Add normalized and raw phenotypes
    if args.pheno:
        pheno_path = args.pheno
        IMP_input = combine_pheno(IMP_input, pheno_path)
        QA_input = combine_pheno(QA_input, pheno_path)
    # Add empty fold column if absent
    IMP_input = dummy_folds_column(IMP_input)
    QA_input = dummy_folds_column(QA_input)

    if not args.summary:
        GBLUP_input = gblup_preprocessing(QA_input, pheno_path)
    else:
        GBLUP_input = None
    folds = [args.fold] if args.fold else range(1, NUM_FOLDS + 1)

    for i in range(1, 11):
        assign_folds_to_file(IMP_input, seed=i)
        sync_folds_column(IMP_input, QA_input)
        if GBLUP_input:
            sync_folds_column(IMP_input,GBLUP_input)
        if args.summary:
            run_saliency_summary(IMP_input=QA_input, QA_input=IMP_input, repeat=i)
        else:
            for fold in folds:
                main(IMP_input, QA_input, repeat=i, run_fold=fold)
                gblup_main(QA_input, repeat=i, run_fold=fold)
                run_rrblup(QA_input, repeat=i, run_fold=fold)
                if not args.skip_cropformer:
                    run_cropformer(
                        QA_input,
                        repeat=i,
                        run_fold=fold,
                        max_features=args.cropformer_max_features,
                        epochs=args.cropformer_epochs
                    )
                print(f"Repeat {i} fold {fold} complete")
    # Find average saliency for all repeats and extract top snps
    if args.summary:
        merged_df = None
        for i in range(1, 11):
            path = f"Repeat_{i}/top_saliency_snps.csv"
            df = pd.read_csv(path)
            # Rename saliency column
            df.rename(columns={"Saliency": f"Saliency_{i}"}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='SNP', how='inner')
        saliency_cols = [col for col in merged_df.columns if col.startswith("Saliency_")]
        merged_df["avg_saliency"] = merged_df[saliency_cols].mean(axis=1)
        export_top_k_saliency(snp_names=merged_df["SNP"], saliency_values=merged_df["avg_saliency"])
        plot_average_saliency(avg_saliency=merged_df["avg_saliency"])

        df = pd.read_csv("RRBLUP_u_effects.csv")
        # Extract only SNP columns
        snp_names = df.columns[2:]
        u_matrix = df[snp_names]
        summary = pd.DataFrame({
            "SNP": snp_names,
            "Mean_Effect": u_matrix.mean(axis=0).values,
            "SD_Effect": u_matrix.std(axis=0).values,
            "Abs_Mean_Effect": np.abs(u_matrix.mean(axis=0)).values
        })
        summary.to_csv("RRBLUP_SNP_summary.csv", index=False)
    else:
        # Compute evaluation metrics for the experiment
        tau_summary()
        tau_summary("GB")
        tau_summary("RB")
        error_summary()
        error_summary("GB")
        error_summary("RB")
        if not args.skip_cropformer:
            tau_summary("CF")
            error_summary("CF")

