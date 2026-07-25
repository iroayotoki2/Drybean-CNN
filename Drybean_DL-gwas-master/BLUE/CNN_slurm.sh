#!/bin/bash
#SBATCH --account=def-cottenie_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=drybean_blue
#SBATCH --output=drybean_blue_%j.out
#SBATCH --error=drybean_blue_%j.err
#SBATCH --gpus-per-node=a100:1

set -euo pipefail

# Move to project directory
cd ~/scratch/Drybean-CNN/Drybean_DL-gwas-master/BLUE/

# Thread settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load PLINK environment
module purge
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64

which plink
plink --version

# LD pruning
python ld_pruning.py --vcf imputed_GenotypicData.vcf
python ld_pruning.py --vcf Raw_GenotypicData.vcf


# Load Python/CUDA environment
module load StdEnv/2023
module load python/3.10
module load r/4.5.0
module load cuda

# Activate virtual environment
cd ~/scratch/Drybean-CNN
source ENV/bin/activate

cd ~/scratch/Drybean-CNN/Drybean_DL-gwas-master/BLUE/

which python
python --version

# Verify GPU access
python - <<EOF
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
EOF

# Run Cropformer/CNN/GBLUP/rrBLUP pipeline
python BLUE.py \
    LD_imputed_GenotypicData.vcf \
    LD_Raw_GenotypicData.vcf \
    --pheno pheno_normalized.tsv

# Generate final summaries
python BLUE.py \
    LD_imputed_GenotypicData_processed.tsv \
    LD_Raw_GenotypicData_processed.tsv \
    --summary