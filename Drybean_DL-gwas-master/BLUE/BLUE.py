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


def assign_folds_to_file(filename, output_filename=None, seed=42):
    if output_filename is None:
        output_filename = filename  # Overwrite original

    df = pd.read_csv(filename, sep='\t')
    num_samples = len(df)

    fold_numbers = np.arange(1, NUM_FOLDS + 1)

    # Generate new fold assignments
    np.random.seed(seed)
    np.random.shuffle(fold_numbers)
    new_folds = np.tile(fold_numbers, int(np.ceil(num_samples / NUM_FOLDS)))[:num_samples]
    assert len(new_folds) == num_samples, "Mismatch in fold assignment length!"

    df.iloc[:, 0] = new_folds  # Replace first column with new folds

    # Double check before writing
    assert len(df) == num_samples, f"Row count changed! Expected {num_samples}, got {len(df)}"

    df.to_csv(output_filename, sep='\t', index=False)


def sync_folds_column(source_file, target_file, output_file=None):
    """
    Copy the first column (fold assignment) from source_file to target_file.
    Optionally write the result to output_file (or overwrite target_file).
    """
    if output_file is None:
        output_file = target_file

    source_df = pd.read_csv(source_file, sep='\t')
    target_df = pd.read_csv(target_file, sep='\t')

    if len(source_df) != len(target_df):
        raise ValueError(f"Row count mismatch: {source_file} has {len(source_df)}, {target_file} has {len(target_df)}")

    target_df.iloc[:, 0] = source_df.iloc[:, 0]  # Copy fold column
    target_df.to_csv(output_file, sep='\t', index=False)


def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)

    return np.eye(nb_classes)[targets]


def predict_height_from_all_folds(snp_vector, model_dir="model_IMP"):
    one_hot = indices_to_one_hot(snp_vector, nb_classes)
    one_hot = np.expand_dims(one_hot, axis=0).astype(np.float32)

    predictions = []

    for i in range(1, NUM_FOLDS + 1):
        model_path = f"{model_dir}/model_{i}.h5"
        model = load_model(model_path, custom_objects={"isru": isru})
        model.compile(loss='mean_squared_error', optimizer='adam')

        pred = model.predict(one_hot, verbose=0)
        predictions.append(pred[0][0])

    return np.mean(predictions)


def readData(input):
    data = pd.read_csv(input, sep='\t', header=0, na_values='nan')
    snp_df = data.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
    SNP = snp_df.values
    snp_names = snp_df.columns.tolist()
    pheno = data.iloc[:, 1].apply(pd.to_numeric, errors='coerce').values
    folds = data.iloc[:, 0].apply(pd.to_numeric, errors='coerce').values
    Lines = data.iloc[:, 2]
    # arr = np.empty(shape=(SNP.shape[0],SNP.shape[1] , nb_classes))
    # arr = np.memmap('snp_encoded.dat', dtype='float32', mode='w+', shape=(SNP.shape[0], SNP.shape[1], nb_classes))

    # for i in range(0,SNP.shape[0]):
    # 	arr[i] = indices_to_one_hot(pd.to_numeric(SNP[i],downcast='signed'), nb_classes)

    return SNP.astype(np.int8), pheno, folds, snp_names, Lines


def attention_block(x):
    attn = MultiHeadAttention(
        num_heads=4,
        key_dim=10
    )(x,x)

    x = Add()([x,attn])

    x = LayerNormalization()(x)

    return x

def resnet(input):
    inputs = Input(shape=(input.shape[1], nb_classes))

    x = Conv1D(10, 4, padding='same', activation='linear', kernel_initializer='TruncatedNormal',
               kernel_regularizer=regularizers.l2(0.1), bias_regularizer=regularizers.l2(0.01))(inputs)

    x = Conv1D(10, 20, padding='same', activation='linear', kernel_initializer='TruncatedNormal',
               kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)
    x = layers.Activation(isru)(x)

    x = Dropout(0.75)(x)

    shortcut = Conv1D(10, 4, padding='same', activation='linear', kernel_initializer='TruncatedNormal',
                      kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.add([shortcut, x])

    x = Conv1D(10, 4, padding='same', activation='linear', kernel_initializer='TruncatedNormal',
               kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)

    x = Dropout(0.75)(x)

    x = Flatten()(x)

    x = Dropout(0.75)(x)

    outputs = Dense(1, activation='linear', bias_regularizer=regularizers.l2(0.01), kernel_initializer='TruncatedNormal',
                    name='out')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

    return model


def show_images_plot(saliency, wald, outname):
    plt.figure(figsize=(15, 8), facecolor='w')

    plt.subplot(2, 1, 1)
    x = np.median(saliency, axis=-1)
    plt.plot(x, 'b.')
    line = sorted(x, reverse=True)[10]
    plt.axhline(y=line, color='b', linestyle='--')
    plt.ylabel('saliency value', fontdict=None, labelpad=None, fontsize=15)

    plt.subplot(2, 1, 2)
    plt.plot(wald, 'r1')
    line = sorted(wald, reverse=True)[10]
    plt.axhline(y=line, color='r', linestyle='--')

    plt.xlabel('SNPs', fontdict=None, labelpad=None, fontsize=15)
    plt.ylabel('Wald', fontdict=None, labelpad=None, fontsize=15)

    plt.savefig(outname)
    plt.clf()
    plt.cla()
    plt.close()


def plot_average_saliency(avg_saliency, output_file="avg_saliency_across_folds.png", repeat=0):
    if repeat != 0:
        output_file = f"Repeat_{repeat}/{output_file}"
    plt.figure(figsize=(15, 6))
    plt.plot(avg_saliency, '.', markersize=3)
    plt.xlabel("SNP Index")
    plt.ylabel("Average Saliency")
    plt.title("Average Saliency per SNP Across All Folds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def get_saliency(input_tensor, model):
    """
    Compute saliency map using TensorFlow GradientTape.
    Args:
        input_tensor: one-hot encoded SNPs for a single sample (shape: [num_SNPs, 4])
        model: trained Keras model
    Returns:
        saliency: np.array of shape (num_SNPs,) with max gradient across channels
    """
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_tensor, axis=0))  # shape: (1, num_SNPs, 4)
    input_tensor = tf.cast(input_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor, training=False)  # shape: (1, 1)

    gradient = tape.gradient(prediction, input_tensor)  # shape: (1, num_SNPs, 4)
    saliency = tf.reduce_max(tf.abs(gradient), axis=-1)  # max across classes
    return saliency.numpy().squeeze()  # shape: (num_SNPs,)


def export_top_k_saliency(snp_names, saliency_values, k=20, output_file="top_saliency_snps.csv", repeat=0):
    # Sort SNPs by saliency (descending)
    top_indices = np.argsort(saliency_values)[::-1][:k]
    top_snps = [(snp_names[i], saliency_values[i]) for i in top_indices]
    if repeat != 0:
        output_file = f"Repeat_{repeat}/{output_file}"
    # Write to CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNP", "Saliency"])
        for snp, score in top_snps:
            writer.writerow([snp, f"{score:.6f}"])
    print(f"📁 Top {k} SNPs by saliency saved to '{output_file}'")


def collect_saliency_across_folds(imp_SNP, imp_pheno, folds, repeat):
    all_saliencies = []

    for i in range(1, NUM_FOLDS + 1):
        print(f"Processing Repeat {repeat} fold {i}...")

        # Load the trained model for this fold
        model_path = f"Repeat_{repeat}/model_QA/model_{i}.h5"
        model = load_model(model_path, custom_objects={"isru": isru})
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Get test indices for this fold
        test_idx = np.where(folds == i)[0]

        # Compute saliency for each test sample
        for idx in test_idx:
            snp_vector = indices_to_one_hot(imp_SNP[idx], nb_classes).astype(np.float32)
            sal = get_saliency(snp_vector, model)
            all_saliencies.append(sal)

    print("Finished collecting saliency data from all folds.")
    return np.mean(np.stack(all_saliencies), axis=0)  # shape: (num_SNPs,)


a = 0.03  # height


def isru(x):
    return x / (tf.math.sqrt(1 + a * tf.math.square(x)))


class SNPGenerator(tf.keras.utils.Sequence):
    def __init__(self, SNP, labels, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.SNP = SNP
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.SNP) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.SNP[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_encoded = np.array([indices_to_one_hot(x, nb_classes) for x in batch_x], dtype=np.float32)
        return batch_encoded, batch_y


def model_train(testSNP, valSNP, trainSNP, testPheno, valPheno, trainPheno, model_save, weights_save):
    batch_size = 4
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, mode='min')

    model = resnet(trainSNP)

    # Use generators for training and validation
    train_gen = SNPGenerator(trainSNP, trainPheno, batch_size)
    val_gen = SNPGenerator(valSNP, valPheno, batch_size)

    history = model.fit(
        train_gen,
        epochs=1000,
        validation_data=val_gen,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=1
    )

    model.save(model_save)
    model.save_weights(weights_save)

    # Test set: one-hot encode manually
    test_encoded = np.array([indices_to_one_hot(x, nb_classes) for x in testSNP], dtype=np.float32)
    pred = model.predict(test_encoded)
    pred = pred.flatten()

    corr = pearsonr(pred, testPheno)[0]
    return history, pred, corr


def safe_delete(*varnames):
    for name in varnames:
        if name in locals():
            del locals()[name]
        elif name in globals():
            del globals()[name]

def plot_training_history(history, output_file="training_history.png"):

    plt.figure(figsize=(10,5))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def run_saliency_summary(IMP_input, QA_input, repeat, interactive=False):
    imp_SNP, imp_pheno, folds, snp_names, _ = readData(IMP_input)

    avg_saliency = collect_saliency_across_folds(imp_SNP, imp_pheno, folds, repeat)
    plot_average_saliency(avg_saliency, repeat=repeat)

    # After folds are run
    if repeat == NUM_REPEATS:
        if os.path.exists("fold_pcc_log.csv"):
            df = pd.read_csv("fold_pcc_log.csv", header=None, names=["Repeat", "Fold", "PCC_Imputed", "PCC_NonImputed"])
            df = df.sort_values(["Repeat", "Fold"])
            df.to_csv("fold_pcc_summary.csv", index=False)

            print("✅ Summary CSV generated: fold_pcc_summary.csv")
            print("📈 Average PCC (imputed):", round(df['PCC_Imputed'].mean(), 4))
            print("📈 Average PCC (non-imputed):", round(df['PCC_NonImputed'].mean(), 4))
        else:
            print("❌ PCC log not found. Did folds run correctly?")

        if os.path.exists("GB_fold_pcc_log.csv"):
            Gdf = pd.read_csv("GB_fold_pcc_log.csv", header=None, names=["Repeat", "Fold", "PCC_GBLUP"])
            Gdf = Gdf.sort_values(["Repeat", "Fold"])
            Gdf.to_csv("GB_fold_pcc_summary.csv", index=False)

            print("✅ Summary CSV generated: GB_fold_pcc_summary.csv")
            print("📈 Average PCC for GBLUP:", round(Gdf['PCC_GBLUP'].mean(), 4))

        else:
            print("❌ GBLUP PCC log not found. Did folds run correctly?")

        if os.path.exists("RB_fold_pcc_log.csv"):
            df = pd.read_csv("RB_fold_pcc_log.csv", header=None, names=["Repeat", "Fold", "PCC_rrBLUP"])
            df = df.sort_values(["Repeat", "Fold"])
            df.to_csv("RB_fold_pcc_summary.csv", index=False)

            print("✅ Summary CSV generated: RB_fold_pcc_summary.csv")
            print("📈 Average PCC for rrBLUP:", round(df['PCC_rrBLUP'].mean(), 4))

        else:
            print("❌ GBLUP PCC log not found. Did folds run correctly?")
    print("✅ Average saliency map and plot generated.")

    if repeat == NUM_REPEATS:
        if os.path.exists("CF_fold_pcc_log.csv"):
            df = pd.read_csv("CF_fold_pcc_log.csv", header=None, names=["Repeat", "Fold", "PCC_Cropformer"])
            df = df.sort_values(["Repeat", "Fold"])
            df.to_csv("CF_fold_pcc_summary.csv", index=False)

            print("Summary CSV generated: CF_fold_pcc_summary.csv")
            print("Average PCC for Cropformer:", round(df['PCC_Cropformer'].mean(), 4))
        else:
            print("Cropformer PCC log not found. Did Cropformer folds run correctly?")

    export_top_k_saliency(snp_names, avg_saliency, k=len(snp_names), repeat=repeat)

    # Interactive SNP viewer
    if interactive:
        while True:
            snp_query = input("Enter SNP name to view saliency (or type 'q' to quit): ")
            if snp_query.lower() in ['q', 'quit', 'exit']:
                break

            try:
                idx = snp_names.index(snp_query)
                print(f"🔬 Average saliency for {snp_query}: {avg_saliency[idx]:.6f}\n")
            except ValueError:
                print("❌ SNP not found. Please check the name and try again.\n")


def main(IMP_input, QA_input, repeat, run_fold=None):
    IMP_corr = []
    QA_corr = []

    # Load data once
    imp_SNP, imp_pheno, folds, snp_names, _ = readData(IMP_input)
    QA_SNP, QA_pheno, folds, _, Lines = readData(QA_input)
    PHENOTYPE = imp_pheno


    # If a specific fold is requested, just run that one; otherwise run all
    fold_range = [run_fold] if run_fold else range(1, NUM_FOLDS + 1)
    for i in fold_range:
        print(f"\n🔁 Starting Repeat{repeat}fold {i} ...")

        # Identify test fold
        testIdx = np.where(folds == i)[0]

        # If NUM_FOLDS >= 3, use separate fold for validation
        if NUM_FOLDS >= 3:
            val_fold = (i % NUM_FOLDS) + 1
            valIdx = np.where(folds == val_fold)[0]
            trainIdx = np.where((folds != i) & (folds != val_fold))[0]
        else:
            # With 2 folds: split the other fold randomly into train/val
            other_idx = np.where(folds != i)[0]
            np.random.seed(repeat)
            np.random.shuffle(other_idx)
            val_size = int(0.2 * len(other_idx))
            valIdx = other_idx[:val_size]
            trainIdx = other_idx[val_size:]

        if len(trainIdx) == 0 or len(valIdx) == 0:
            raise ValueError(f"Fold {i}: Training or validation set is empty. Check NUM_FOLDS or data distribution.")

        # Partition data
        trainSNP, trainSNP_QA, trainPheno = imp_SNP[trainIdx], QA_SNP[trainIdx], PHENOTYPE[trainIdx]
        valSNP, valSNP_QA, valPheno = imp_SNP[valIdx], QA_SNP[valIdx], PHENOTYPE[valIdx]
        testSNP, testSNP_QA, testPheno, testLines = imp_SNP[testIdx], QA_SNP[testIdx], PHENOTYPE[testIdx], Lines[
            testIdx]

        # Train and evaluate on imputed data
        history, pred, corr = model_train(
            testSNP, valSNP, trainSNP, testPheno, valPheno, trainPheno,
            f'Repeat_{repeat}/model_IMP/model_{i}.h5',
            f'Repeat_{repeat}/model_IMP/model_weights{i}.weights.h5'
        )
        IMP_corr.append(float(f'{corr:.4f}'))
        print(f"✅ Fold {i} (imputed) PCC: {corr:.4f}")


        # Train and evaluate on non-imputed data
        history, pred, corr = model_train(
            testSNP_QA, valSNP_QA, trainSNP_QA, testPheno, valPheno, trainPheno,
            f'Repeat_{repeat}/model_QA/model_{i}.h5',
            f'Repeat_{repeat}/model_QA/model_weights{i}.weights.h5'
        )
        if repeat == 10:
            if i == 2:
                plot_training_history(history)
        if repeat == 5:
            if i == 2:
                plot_training_history(history, output_file="poor_training_history.png")
        QA_corr.append(float(f'{corr:.4f}'))
        print(f"✅ Fold {i} (non-imputed) PCC: {corr:.4f}")
        fold_pred = pd.DataFrame({"fold": i, "Line": testLines, "predicted": pred, "phenotype": testPheno})
        fold_pred.to_csv(f"Repeat_{repeat}/fold_data.csv", mode="a",
                         header=not os.path.exists(f"Repeat_{repeat}/fold_data.csv"), index=False)
        # 🧹 Clear memory
        from keras import backend as K
        import gc
        K.clear_session()
        gc.collect()

    if run_fold is not None:
        with open("fold_pcc_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([repeat, run_fold, IMP_corr[0], QA_corr[0]])
            print(f"✅ Saved fold {run_fold} to fold_pcc_log.csv")

    if run_fold is None:
        with open("fold_pcc_summary.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Repeat", "Fold", "PCC_Imputed", "PCC_NonImputed"])
            for fold in range(NUM_FOLDS):
                writer.writerow([repeat, fold + 1, IMP_corr[fold], QA_corr[fold]])


def vcf_preprocessing(IMP_input, output_path=None):
    from pysam import VariantFile
    vcf = VariantFile(IMP_input)
    data = []
    index = []

    for record in vcf:
        index.append(f"Chr{record.chrom}_{record.pos}")

        gts = [sum(s['GT']) if s['GT'] and None not in s['GT'] else int(-1)
               for s in record.samples.values()]

        data.append(gts)
    # Create the final DataFrame
    df = pd.DataFrame(data, index=index, columns=list(vcf.header.samples))
    df_T = df.T
    df_final = df_T.reset_index().rename(columns={'index': 'Line'})
    df_final['Line'] = df_final['Line'].str.replace(r'\.\d+$', '', regex=True)

    df_final = df_final.drop_duplicates(subset="Line", keep="first")
    if output_path is None:
        base = os.path.splitext(IMP_input)[0]
        output_path = f"{base}_processed.tsv"

    df_final.to_csv(output_path, sep='\t', index=False)
    return output_path


def csv_preprocessing(input_path, output_path=None):
    df = pd.read_csv(input_path)
    df.drop(df.columns[0], inplace=True, axis=1)

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_processed.tsv"

        df = df.drop_duplicates(subset="Line", keep="first")

    df.to_csv(output_path, sep='\t', index=False)
    return output_path


def combine_pheno(input_path, pheno_path, output_path=None):
    Geno = pd.read_csv(input_path, sep='\t')
    pheno = pd.read_csv(pheno_path, sep='\t')
    Geno["Line"] = Geno["Line"].str.replace(" ", "_")
    pheno["Line"] = pheno["Line"].str.replace(" ", "_")
    merged_df = pd.merge(Geno, pheno, on='Line', how='left')

    missing = merged_df["BLUEs"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} phenotype values are missing after merging.")

    # Reorder columns
    # Using BLUE values without normalization
    fixed_cols = ['BLUEs', 'Line', 'norm_phe']
    snp_cols = [c for c in merged_df.columns if c not in fixed_cols]
    # Finalize the DataFrame
    final_df = merged_df[fixed_cols + snp_cols]

    if output_path is None:
        output_path = input_path

    final_df.to_csv(output_path, sep='\t', index=False)
    return output_path


def dummy_folds_column(input_path, output_path=None):
    df = pd.read_csv(input_path, sep='\t')
    if 'folds' not in df.columns:
        df.insert(0, 'folds', 0)

    if output_path is None:
        output_path = input_path

    df.to_csv(output_path, sep='\t', index=False)
    return output_path


def find_kendall_tau(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["predicted", "phenotype"])
    tau, p = kendalltau(df["predicted"], df["phenotype"])
    return tau, p


def prediction_error(path):
    df = pd.read_csv(path)
    df["Error"] = abs(df["predicted"] - df["phenotype"])
    new_df = df[["Line", "Error"]]
    return new_df

def gblup_preprocessing(tsv_path, pheno_path):
    vcf_path = tsv_path.replace("_processed.tsv", ".vcf")
    vcf_path = vcf_path.replace("LD_", "")
    matrix_path = vcf_preprocessing(vcf_path)
    matrix_path = combine_pheno(matrix_path, pheno_path)
    matrix_path = dummy_folds_column(matrix_path)
    return matrix_path

def gblup_main(GBLUP_input, repeat, run_fold=None):
    GB_corr = []
    SNP, PHENOTYPE, folds, snp_names, Lines = readData(GBLUP_input)

    print("PHENOTYPE dtype:", PHENOTYPE.dtype)
    print("Total NaNs in PHENOTYPE:", pd.isna(PHENOTYPE).sum())

    if pd.isna(PHENOTYPE).any():
        idx = np.where(pd.isna(PHENOTYPE))[0]
        print("NaN indices:", idx)
        print("Corresponding lines:", Lines[idx])

    p = SNP.mean(axis=0) / 2  # Allele frequencies
    Z = SNP - 2 * p  # Center genotypes
    denom = 2 * np.sum(p * (1 - p))  # VanRaden normalization

    G = (Z @ Z.T) / denom

    print(np.isnan(G).any())
    print(np.isinf(G).any())
    print(denom)

    fold_range = [run_fold] if run_fold else range(1, NUM_FOLDS + 1)
    for i in fold_range:
        print(f"\n🔁 Starting GBLUP for Repeat{repeat}fold {i} ...")

        # Identify test fold
        testIdx = np.where(folds == i)[0]

        # If NUM_FOLDS >= 3, use separate fold for validation
        if NUM_FOLDS >= 3:
            val_fold = (i % NUM_FOLDS) + 1
            valIdx = np.where(folds == val_fold)[0]
            trainIdx = np.where((folds != i) & (folds != val_fold))[0]
        else:
            # With 2 folds: split the other fold randomly into train/val
            other_idx = np.where(folds != i)[0]
            np.random.seed(repeat)
            np.random.shuffle(other_idx)
            val_size = int(0.2 * len(other_idx))
            valIdx = other_idx[:val_size]
            trainIdx = other_idx[val_size:]

        if len(trainIdx) == 0 or len(valIdx) == 0:
            raise ValueError(f"Fold {i}: Training or validation set is empty. Check NUM_FOLDS or data distribution.")

        # Partition data
        G_train, y_train =G[np.ix_(trainIdx, trainIdx)], PHENOTYPE[trainIdx]
        G_val_train, y_val = G[np.ix_(valIdx, trainIdx)], PHENOTYPE[valIdx]
        G_test_train, y_test, testLines = G[np.ix_(testIdx, trainIdx)], PHENOTYPE[testIdx], Lines[
            testIdx]

        mu = np.mean(y_train)
        y_train_centered = y_train - mu

        # Candidate regularization strengths
        candidate_lambdas = np.logspace(-4, 2, 20)

        best_lambda = None
        best_corr = -np.inf
        best_u = None

        n = G_train.shape[0]

        for lam in candidate_lambdas:
            G_reg = G_train + lam * np.eye(n)

            print("NaN in G_reg:", np.isnan(G_reg).any())

            u = np.linalg.solve(G_reg, y_train_centered)



            # Predict validation phenotypes
            y_val_pred = G_val_train @ u + mu

            # Validation correlation
            corr = np.corrcoef(y_val, y_val_pred)[0, 1]

            print(f"λ = {lam}")
            print(f"corr = {corr}")
            print(f"is NaN? {np.isnan(corr)}")

            if np.isnan(corr):
                continue

            if corr > best_corr:
                best_corr = corr
                best_lambda = lam
                best_u = u

        if best_lambda is None:
            raise ValueError("No valid lambda produced a finite validation correlation.")

        print(f"Best λ = {best_lambda:.4f} | Validation correlation = {best_corr:.4f}")

        # Use the best model to predict the test set
        y_test_pred = G_test_train @ best_u + mu
        test_corr = pearsonr(y_test_pred,y_test)[0]
        GB_corr.append(float(f'{test_corr:.4f}'))
        fold_pred = pd.DataFrame({"fold": i, "Line": testLines, "predicted": y_test_pred, "phenotype": y_test})
        fold_pred.to_csv(f"Repeat_{repeat}/GB_fold_data.csv", mode="a",
                         header=not os.path.exists(f"Repeat_{repeat}/GB_fold_data.csv"), index=False)

        if run_fold is not None:
            with open("GB_fold_pcc_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([repeat, run_fold,  GB_corr[0]])
                print(f"✅ Saved fold {run_fold} to GB_fold_pcc_log.csv")

        if run_fold is None:
            with open("GB_fold_pcc_summary.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Repeat", "Fold", "PCC_GBLUP"])
                for fold in range(NUM_FOLDS):
                    writer.writerow([repeat, fold + 1, GB_corr[fold]])

def run_rrblup(input, repeat, run_fold=None):
    SNP, PHENOTYPE, folds, snp_names, Lines = readData(input)
    RB_corr = []

    u_file = "RRBLUP_u_effects.csv"

    if not os.path.exists(u_file):
        header = ["Repeat", "Fold"] + list(snp_names)
        pd.DataFrame(columns=header).to_csv(u_file, index=False)

    fold_range = [run_fold] if run_fold else range(1, NUM_FOLDS + 1)
    for i in fold_range:
        print(f"\n🔁 Starting GBLUP for Repeat{repeat}fold {i} ...")

        # Identify test fold
        testIdx = np.where(folds == i)[0]

        # If NUM_FOLDS >= 3, use separate fold for validation
        if NUM_FOLDS >= 3:
            val_fold = (i % NUM_FOLDS) + 1
            valIdx = np.where(folds == val_fold)[0]
            trainIdx = np.where((folds != i) & (folds != val_fold))[0]
        else:
            # With 2 folds: split the other fold randomly into train/val
            other_idx = np.where(folds != i)[0]
            np.random.seed(repeat)
            np.random.shuffle(other_idx)
            val_size = int(0.2 * len(other_idx))
            valIdx = other_idx[:val_size]
            trainIdx = other_idx[val_size:]

        if len(trainIdx) == 0 or len(valIdx) == 0:
            raise ValueError(f"Fold {i}: Training or validation set is empty. Check NUM_FOLDS or data distribution.")

        # Partition data
        X_train, y_train =SNP[trainIdx], PHENOTYPE[trainIdx]
        X_val, y_val = SNP[valIdx], PHENOTYPE[valIdx]
        X_test, y_test, testLines = SNP[testIdx], PHENOTYPE[testIdx], Lines[testIdx]


        with localconverter(default_converter + numpy2ri.converter):
            fit = rrBLUP.mixed_solve(y=y_train, Z=X_train)
            fit_dict = dict(zip(fit.names(), fit))

            beta = np.array(fit_dict["beta"]).item()
            u = np.array(fit_dict["u"]).flatten()
        # Extract SNP effects
        print("Number of SNP effects:", len(u))
        print("Number of SNP names:", len(snp_names))
        u_row = pd.DataFrame(
            [[repeat, i] + list(u)],
            columns=["Repeat", "Fold"] + list(snp_names)
        )
        # Append to CSV
        u_row.to_csv(
            u_file,
            mode="a",
            header=False,
            index=False
        )

        y_val_pred = X_val @ u + beta
        corr = np.corrcoef(y_val, y_val_pred)[0, 1]

        y_test_pred = X_test @ u + beta
        test_corr = pearsonr(y_test_pred, y_test)[0]
        RB_corr.append(float(f'{test_corr:.4f}'))
        fold_pred = pd.DataFrame({"fold": i, "Line": testLines, "predicted": y_test_pred, "phenotype": y_test})
        fold_pred.to_csv(f"Repeat_{repeat}/RB_fold_data.csv", mode="a",
                         header=not os.path.exists(f"Repeat_{repeat}/RB_fold_data.csv"), index=False)

        if run_fold is not None:
            with open("RB_fold_pcc_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([repeat, i, RB_corr[0]])
                print(f"✅ Saved fold {run_fold} to RB_fold_pcc_log.csv")

        if run_fold is None:
            with open("RB_fold_pcc_summary.csv", mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Repeat", "Fold", "PCC_rrBLUP"])
                for fold in range(NUM_FOLDS):
                    writer.writerow([repeat, fold + 1, RB_corr[fold]])

def top_mean_predictions(filename, model, k=15):
    merged_df = None
    for i in range(1, NUM_REPEATS + 1):
        df = pd.read_csv(f"Repeat_{i}/{filename}")
        # Keep only the sample ID and predicted value
        repeat_df = df[["Line", "predicted"]].copy()
        repeat_df.rename(columns={"predicted": f"repeat_{i}"}, inplace=True)
        if merged_df is None:
            merged_df = repeat_df
        else:
            merged_df = merged_df.merge(repeat_df, on="Line")

    # Average prediction across repeats
    merged_df["avg_prediction"] = merged_df.iloc[:, 1:].mean(axis=1)
    # Rank from highest to lowest
    top_k_df = (
        merged_df
        .sort_values("avg_prediction", ascending=False)
        .head(k)
    )
    top_k_df.to_csv(f"{model}_mean_prediction_ranking.csv", index=False)

def top_selection_frequency(filename, model, k=10):
    frequency = {}
    for i in range(1, NUM_REPEATS + 1):
        df = pd.read_csv(f"Repeat_{i}/{filename}")
        # Get the top k predictions for this repeat
        top_k = (
            df.sort_values("predicted", ascending=False)
              .head(k)
        )
        # Count appearances
        for line in top_k["Line"]:
            if line in frequency:
                frequency[line] += 1
            else:
                frequency[line] = 1

    # Convert dictionary to dataframe
    frequency_df = pd.DataFrame(
        frequency.items(),
        columns=["Line", "Top_10_Frequency"]
    )

    # Rank by frequency
    frequency_df = frequency_df.sort_values(
        "Top_10_Frequency",
        ascending=False
    )

    frequency_df.to_csv(
        f"{model}_top_{k}_selection_frequency.csv",
        index=False
    )


def run_cropformer(input, repeat, run_fold=None, max_features=10000, epochs=100, batch_size=32, patience=5):
    import gc
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    class Cropformer(nn.Module):
        def __init__(
                self,
                input_size,
                cnn_channels=64,
                embed_dim=64,
                num_heads=8,
                dropout=0.5,
                output_dim=1
        ):
            super(Cropformer, self).__init__()

            # CNN feature extractor
            self.cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=cnn_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_channels),

                nn.Conv1d(
                    in_channels=cnn_channels,
                    out_channels=embed_dim,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU()
            )
            # Reduce SNP dimension before attention
            self.pool = nn.AdaptiveAvgPool1d(256)

            # Multi-head self attention
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            self.norm1 = nn.LayerNorm(embed_dim)

            # Transformer feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )

            self.norm2 = nn.LayerNorm(embed_dim)

            # Regression head
            self.output = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, output_dim)
            )

        def forward(self, x):
            # (batch, SNPs)
            x = x.unsqueeze(1)

            # (batch, embed_dim, SNPs)
            x = self.cnn(x)

            # Reduce sequence length
            x = self.pool(x)

            # (batch, sequence_length, embed_dim)
            x = x.transpose(1, 2)

            # Self attention
            attn_out, _ = self.attention(x, x, x)

            x = self.norm1(x + attn_out)

            # Feed-forward block
            ffn_out = self.ffn(x)

            x = self.norm2(x + ffn_out)

            # Global average pooling
            x = torch.mean(x, dim=1)

            return self.output(x)

    def select_and_pad_features(train_x, val_x, test_x, train_y, snp_names, feature_count):
        selected_indices = np.arange(train_x.shape[1])
        if train_x.shape[1] > feature_count:
            scores = mutual_info_regression(train_x, train_y, random_state=repeat)
            selected_indices = np.argsort(scores)[-feature_count:]
            train_x = train_x[:, selected_indices]
            val_x = val_x[:, selected_indices]
            test_x = test_x[:, selected_indices]
        elif train_x.shape[1] < feature_count:
            missing = feature_count - train_x.shape[1]
            train_x = np.pad(train_x, ((0, 0), (0, missing)), mode="constant")
            val_x = np.pad(val_x, ((0, 0), (0, missing)), mode="constant")
            test_x = np.pad(test_x, ((0, 0), (0, missing)), mode="constant")

        selected_snps = pd.DataFrame({"SNP": np.array(snp_names)[selected_indices]})
        return train_x, val_x, test_x, selected_snps

    SNP, PHENOTYPE, folds, snp_names, Lines = readData(input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_corr = []
    fold_range = [run_fold] if run_fold else range(1, NUM_FOLDS + 1)

    os.makedirs(f"Repeat_{repeat}/model_CF", exist_ok=True)

    for i in fold_range:
        print(f"\nStarting Cropformer for Repeat {repeat} fold {i} ...")
        np.random.seed(repeat * 100 + i)
        torch.manual_seed(repeat * 100 + i)

        testIdx = np.where(folds == i)[0]
        if NUM_FOLDS >= 3:
            val_fold = (i % NUM_FOLDS) + 1
            valIdx = np.where(folds == val_fold)[0]
            trainIdx = np.where((folds != i) & (folds != val_fold))[0]
        else:
            other_idx = np.where(folds != i)[0]
            np.random.shuffle(other_idx)
            val_size = int(0.2 * len(other_idx))
            valIdx = other_idx[:val_size]
            trainIdx = other_idx[val_size:]

        if len(trainIdx) == 0 or len(valIdx) == 0:
            raise ValueError(f"Fold {i}: Training or validation set is empty. Check NUM_FOLDS or data distribution.")

        trainX, trainY = SNP[trainIdx].astype(np.float32), PHENOTYPE[trainIdx].astype(np.float32)
        valX, valY = SNP[valIdx].astype(np.float32), PHENOTYPE[valIdx].astype(np.float32)
        testX, testY, testLines = SNP[testIdx].astype(np.float32), PHENOTYPE[testIdx].astype(np.float32), Lines[testIdx]

        trainX, valX, testX, selected_snps = select_and_pad_features(trainX, valX, testX, trainY, snp_names, max_features)
        selected_snps.to_csv(f"Repeat_{repeat}/model_CF/selected_snps_fold_{i}.csv", index=False)

        scaler = StandardScaler()
        trainX = scaler.fit_transform(trainX)
        valX = scaler.transform(valX)
        testX = scaler.transform(testX)

        train_data = TensorDataset(torch.from_numpy(trainX).float(), torch.from_numpy(trainY).float().reshape(-1, 1))
        val_data = TensorDataset(torch.from_numpy(valX).float(), torch.from_numpy(valY).float().reshape(-1, 1))
        test_data = TensorDataset(torch.from_numpy(testX).float(), torch.from_numpy(testY).float().reshape(-1, 1))

        generator = torch.Generator()
        generator.manual_seed(repeat * 100 + i)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


        model = Cropformer(input_size=max_features,cnn_channels=32,embed_dim=32,num_heads=4).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        best_val_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                loss = loss_function(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    val_losses.append(loss_function(model(x_batch), y_batch).item())

            val_loss = float(np.mean(val_losses))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            torch.save(model.state_dict(), f"Repeat_{repeat}/model_CF/model_{i}.pth")

            model.eval()
            predictions = []
            with torch.no_grad():
                for x_batch, _ in test_loader:
                    pred = model(x_batch.to(device))
                    predictions.extend(pred.detach().cpu().numpy().reshape(-1).tolist())

            test_corr = pearsonr(predictions, testY)[0]
            cf_corr.append(float(f"{test_corr:.4f}"))
            fold_pred = pd.DataFrame({"fold": i, "Line": testLines, "predicted": predictions, "phenotype": testY})
            fold_pred.to_csv(f"Repeat_{repeat}/CF_fold_data.csv", mode="a",
                             header=not os.path.exists(f"Repeat_{repeat}/CF_fold_data.csv"), index=False)
            print(f"Cropformer Repeat {repeat} fold {i} PCC: {test_corr:.4f}")

            del model, train_data, val_data, test_data, train_loader, val_loader, test_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if run_fold is not None:
        with open("CF_fold_pcc_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([repeat, run_fold, cf_corr[0]])
            print(f"Saved Cropformer fold {run_fold} to CF_fold_pcc_log.csv")

    if run_fold is None:
        with open("CF_fold_pcc_summary.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Repeat", "Fold", "PCC_Cropformer"])
            for fold in range(NUM_FOLDS):
                writer.writerow([repeat, fold + 1, cf_corr[fold]])

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
        # Compute kendall's tau for the experiment
        tau_df = []
        for i in range(1, 11):
            path = f"Repeat_{i}/fold_data.csv"
            tau, p = find_kendall_tau(path)
            tau_vals = pd.DataFrame({"Repeat": [i], "tau": [tau], "p-value": [p]})
            tau_df.append(tau_vals)
        tau_df = pd.concat(tau_df, ignore_index=True)
        final_tau = tau_df["tau"].mean()
        std = tau_df["tau"].std()
        tau_df.to_csv("Kendall_tau.csv", sep='\t', index=False)
        print(f"The overall tau value is {final_tau:.3f} ± {std:.3f}")

        # Error statistics
        Error_df = None
        for i in range(1, 11):
            path = f"Repeat_{i}/fold_data.csv"
            df = prediction_error(path)
            # Rename error columns
            df.rename(columns={"Error": f"Error_{i}"}, inplace=True)
            if Error_df is None:
                Error_df = df.copy()
            else:
                Error_df = pd.merge(Error_df, df, on='Line', how='inner')
        Error_cols = [col for col in Error_df.columns if col.startswith("Error_")]
        Error_df["avg_error"] = Error_df[Error_cols].mean(axis=1)
        Error_cleaned = Error_df[["Line", "avg_error"]].sort_values(by="avg_error", ascending=True)
        Error_cleaned.to_csv("Prediction_Error.csv", sep='\t', index=False)

    #Same metrics for GBLUP
        GB_tau_df = []
        for i in range(1, 11):
            path = f"Repeat_{i}/GB_fold_data.csv"
            tau, p = find_kendall_tau(path)
            tau_vals = pd.DataFrame({"Repeat": [i], "tau": [tau], "p-value": [p]})
            GB_tau_df.append(tau_vals)
        GB_tau_df = pd.concat(GB_tau_df, ignore_index=True)
        final_tau = GB_tau_df["tau"].mean()
        std = GB_tau_df["tau"].std()
        GB_tau_df.to_csv("GB_Kendall_tau.csv", sep='\t', index=False)
        print(f"The overall tau value for GBLUP is {final_tau:.3f} ± {std:.3f}")

        # Error statistics
        GB_Error_df = None
        for i in range(1, 11):
            path = f"Repeat_{i}/GB_fold_data.csv"
            df = prediction_error(path)
            # Rename error columns
            df.rename(columns={"Error": f"Error_{i}"}, inplace=True)
            if GB_Error_df is None:
                GB_Error_df = df.copy()
            else:
                GB_Error_df = pd.merge(GB_Error_df, df, on='Line', how='inner')
        Error_cols = [col for col in GB_Error_df.columns if col.startswith("Error_")]
        GB_Error_df["avg_error"] = GB_Error_df[Error_cols].mean(axis=1)
        Error_cleaned = GB_Error_df[["Line", "avg_error"]].sort_values(by="avg_error", ascending=True)
        Error_cleaned.to_csv("GB_Prediction_Error.csv", sep='\t', index=False)

        #RRBLUP metrics
        # Compute kendall's tau for the experiment
        tau_df = []
        for i in range(1, 11):
            path = f"Repeat_{i}/RB_fold_data.csv"
            tau, p = find_kendall_tau(path)
            tau_vals = pd.DataFrame({"Repeat": [i], "tau": [tau], "p-value": [p]})
            tau_df.append(tau_vals)
        tau_df = pd.concat(tau_df, ignore_index=True)
        final_tau = tau_df["tau"].mean()
        std = tau_df["tau"].std()
        tau_df.to_csv("RB_Kendall_tau.csv", sep='\t', index=False)
        print(f"The overall tau value for rr-BLUP is {final_tau:.3f} ± {std:.3f}")

        # Error statistics
        Error_df = None
        for i in range(1, 11):
            path = f"Repeat_{i}/RB_fold_data.csv"
            df = prediction_error(path)
            # Rename error columns
            df.rename(columns={"Error": f"Error_{i}"}, inplace=True)
            if Error_df is None:
                Error_df = df.copy()
            else:
                Error_df = pd.merge(Error_df, df, on='Line', how='inner')
        Error_cols = [col for col in Error_df.columns if col.startswith("Error_")]
        Error_df["avg_error"] = Error_df[Error_cols].mean(axis=1)
        Error_cleaned = Error_df[["Line", "avg_error"]].sort_values(by="avg_error", ascending=True)
        Error_cleaned.to_csv("RB_Prediction_Error.csv", sep='\t', index=False)

        top_mean_predictions(filename="fold_data.csv",model="CNN")
        top_selection_frequency(filename="fold_data.csv",model="CNN")
        top_mean_predictions(filename="GB_fold_data.csv", model="GBLUP")
        top_selection_frequency(filename="GB_fold_data.csv", model="GBLUP")
        top_mean_predictions(filename="RB_fold_data.csv", model="RRBLUP")
        top_selection_frequency(filename="RB_fold_data.csv", model="RRBLUP")
        if not args.skip_cropformer:
            tau_df = []
            for i in range(1, 11):
                path = f"Repeat_{i}/CF_fold_data.csv"
                tau, p = find_kendall_tau(path)
                tau_vals = pd.DataFrame({"Repeat": [i], "tau": [tau], "p-value": [p]})
                tau_df.append(tau_vals)
            tau_df = pd.concat(tau_df, ignore_index=True)
            final_tau = tau_df["tau"].mean()
            std = tau_df["tau"].std()
            tau_df.to_csv("CF_Kendall_tau.csv", sep='\t', index=False)
            print(f"The overall tau value for Cropformer is {final_tau:.3f} Â± {std:.3f}")

            Error_df = None
            for i in range(1, 11):
                path = f"Repeat_{i}/CF_fold_data.csv"
                df = prediction_error(path)
                df.rename(columns={"Error": f"Error_{i}"}, inplace=True)
                if Error_df is None:
                    Error_df = df.copy()
                else:
                    Error_df = pd.merge(Error_df, df, on='Line', how='inner')
            Error_cols = [col for col in Error_df.columns if col.startswith("Error_")]
            Error_df["avg_error"] = Error_df[Error_cols].mean(axis=1)
            Error_cleaned = Error_df[["Line", "avg_error"]].sort_values(by="avg_error", ascending=True)
            Error_cleaned.to_csv("CF_Prediction_Error.csv", sep='\t', index=False)

            top_mean_predictions(filename="CF_fold_data.csv", model="Cropformer")
            top_selection_frequency(filename="CF_fold_data.csv", model="Cropformer")
