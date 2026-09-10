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
import pysam
from pysam import VariantFile
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



def match_vcf_features(ORG_input, QA_input):
    """Match QA_input VCF features to the SNPs and order used in ORG_input."""

    # ORG_input is the original/reference VCF.
    # Its SNPs and order define what the model expects.
    org_vcf = VariantFile(ORG_input)

    org_snps = [
        f"Chr{record.chrom}_{record.pos}"
        for record in org_vcf
    ]

    org_vcf.close()

    # Read QA_input and store records by SNP name
    qa_vcf = VariantFile(QA_input)

    qa_records = {}

    for record in qa_vcf:
        snp_name = f"Chr{record.chrom}_{record.pos}"
        qa_records[snp_name] = record

    # Compare SNP sets
    org_set = set(org_snps)
    qa_set = set(qa_records)

    missing = org_set - qa_set
    extra = qa_set - org_set

    # Allow up to 5% of the original ORG SNPs to be missing
    missing_fraction = len(missing) / len(org_snps)

    if missing_fraction > 0.05:
        qa_vcf.close()
        raise ValueError(
            f"QA_input is missing {len(missing)} SNPs required by ORG_input "
            f"({missing_fraction:.2%}). Maximum allowed is 5%. "
            f"Examples: {list(missing)[:10]}"
        )

    if missing:
        print(
            f"Warning: QA_input is missing {len(missing)} SNPs "
            f"({missing_fraction:.2%}) required by ORG_input. "
            f"These will be represented as -1 downstream."
        )

    if extra:
        print(
            f"Warning: QA_input contains {len(extra)} extra SNPs. "
            "These will be excluded."
        )

    # Create output path
    base, ext = os.path.splitext(QA_input)
    output_path = f"{base}_matched{ext}"

    # Preserve QA_input's VCF header
    output_vcf = VariantFile(
        output_path,
        "w",
        header=qa_vcf.header
    )

    # Write QA variants in ORG_input's exact SNP order
    for snp in org_snps:
        if snp in qa_records:
            output_vcf.write(qa_records[snp])

    output_vcf.close()
    qa_vcf.close()

    return output_path
def readpredData(QA_input):
    data = pd.read_csv(QA_input, sep='\t', header=0, na_values='nan')
    Lines = data.iloc[:, 0]
    snp_df = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    SNP = snp_df.values
    snp_names = snp_df.columns.tolist()
    return Lines, SNP.astype(np.int8), snp_names

def prediction_main(QA_input, repeat, fold):
    Lines, SNPset, snp_names = readpredData(QA_input)
    SNP_encoded = np.array([indices_to_one_hot(x, nb_classes) for x in SNPset], dtype=np.float32)

    model = load_model(f'Repeat_{repeat}/model_QA/model_{fold}.h5',  custom_objects={"isru": isru})

    pred = model.predict(SNP_encoded).flatten()

    return Lines, pred
def collect_prediction_saliency(SNPset, repeat):
    all_fold_means = []

    for i in range(1, NUM_FOLDS + 1):
        print(f"Processing Repeat {repeat} fold {i}...")

        model_path = f"Repeat_{repeat}/model_QA/model_{i}.h5"
        model = load_model(model_path, custom_objects={"isru": isru})
        model.compile(loss='mean_squared_error', optimizer='adam')

        fold_saliencies = []

        for idx in range(len(SNPset)):
            snp_vector = indices_to_one_hot(
                SNPset[idx], nb_classes
            ).astype(np.float32)

            sal = get_saliency(snp_vector, model)
            fold_saliencies.append(sal)

        # One saliency vector representing this fold
        fold_mean = np.mean(np.stack(fold_saliencies), axis=0)
        all_fold_means.append(fold_mean)

    # One saliency vector representing this repeat
    repeat_saliency = np.mean(np.stack(all_fold_means), axis=0)

    print("Finished collecting saliency data from all folds.")

    return repeat_saliency
def run_pred_saliency(QA_input, repeat):
    Lines, SNPset, snp_names = readpredData(QA_input)

    avg_saliency = collect_prediction_saliency(
        SNPset,
        repeat
    )

    plot_average_saliency(avg_saliency, output_file="avg_pred_saliency_across_folds.png", repeat= repeat )

    export_top_k_saliency(
        snp_names,
        avg_saliency,
        k=len(snp_names),
        repeat=repeat,
        output_file= "top_pred_saliency_snps.csv"
    )

def pred_vcf_preprocessing(QA_input, ORG_input, output_path=None):
    from pysam import VariantFile

    # Get the SNP order expected by the model
    org_vcf = VariantFile(ORG_input)

    org_snps = [
        f"Chr{record.chrom}_{record.pos}"
        for record in org_vcf
    ]

    org_vcf.close()

    # Read QA VCF
    qa_vcf = VariantFile(QA_input)

    data = {}
    samples = list(qa_vcf.header.samples)

    for record in qa_vcf:
        snp_name = f"Chr{record.chrom}_{record.pos}"

        gts = [
            sum(s['GT']) if s['GT'] and None not in s['GT'] else -1
            for s in record.samples.values()
        ]

        data[snp_name] = gts

    qa_vcf.close()

    # Build matrix using ORG SNP order
    rows = []

    for snp in org_snps:
        if snp in data:
            rows.append(data[snp])
        else:
            # Entire SNP missing from QA
            rows.append([-1] * len(samples))

    df = pd.DataFrame(
        rows,
        index=org_snps,
        columns=samples
    )

    df_final = df.T.reset_index().rename(columns={'index': 'Line'})

    df_final['Line'] = df_final['Line'].str.replace(
        r'\.\d+$', '', regex=True
    )

    df_final = df_final.drop_duplicates(
        subset="Line",
        keep="first"
    )

    if output_path is None:
        base = os.path.splitext(QA_input)[0]
        output_path = f"{base}_processed.tsv"

    df_final.to_csv(
        output_path,
        sep='\t',
        index=False
    )

    return output_path

if __name__ == '__main__':

    # os.chdir("MOISTURE")
    parser = argparse.ArgumentParser()
    parser.add_argument('QA_file', help="QA file")
    parser.add_argument('ORG_file', help="File originally used to train the models")
    parser.add_argument('--summary', action='store_true', help="Run saliency summary after all folds")
    args = parser.parse_args()


    QA_input = args.QA_file
    ORG_input = args.ORG_file

    #Direct to LD pruned data
    ORG_input = f"LD_{ORG_input}"

    #Feature matching of new data to old
    QA_input = match_vcf_features(ORG_input, QA_input)

    # Data cleaning

    if QA_input.endswith(".vcf"):
        QA_input = pred_vcf_preprocessing(QA_input, ORG_input)


    folds = range(1, NUM_FOLDS + 1)

    for i in range(1, 11):
        if args.summary:
            run_pred_saliency(QA_input, repeat=i)
        else:
                repeat_preds = []

                for fold in folds:
                    Lines, pred = prediction_main(QA_input, repeat = i, fold = fold)
                    repeat_preds.append(pred)

                # Average the 5 folds AFTER the fold loop finishes
                repeat_pred = np.mean(repeat_preds, axis=0)

                # Save one prediction file for this repeat
                repeat_data = pd.DataFrame({
                    "Line": Lines,
                    "predicted": repeat_pred
                })

                repeat_data.to_csv(
                    f"Repeat_{i}/new_repeat_data.csv",
                    index=False
                )
    # Find average saliency for all repeats and extract top snps
    if args.summary:
        merged_df = None
        for i in range(1, 11):
            path = f"Repeat_{i}/top_pred_saliency_snps.csv"
            df = pd.read_csv(path)
            # Rename saliency column
            df.rename(columns={"Saliency": f"Saliency_{i}"}, inplace=True)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='SNP', how='inner')
        saliency_cols = [col for col in merged_df.columns if col.startswith("Saliency_")]
        merged_df["avg_saliency"] = merged_df[saliency_cols].mean(axis=1)
        export_top_k_saliency(snp_names=merged_df["SNP"], saliency_values=merged_df["avg_saliency"], output="New Predictions")
        plot_average_saliency(avg_saliency=merged_df["avg_saliency"], output= "New Predictions")

    else:
        # Compute evaluation metrics for the experiment
        top_mean_predictions(filename="new_repeat_data.csv", model="CNN", output="New Predictions")
        top_selection_frequency(filename="new_repeat_data.csv", model="CNN", output="New Predictions")
