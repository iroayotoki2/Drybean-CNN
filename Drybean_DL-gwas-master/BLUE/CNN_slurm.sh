#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=drybean_blue
#SBATCH --output=drybean_blue_%j.out
#SBATCH --error=drybean_blue_%j.err

set -euo pipefail

module load python/3.10

cd ~/scratch/Drybean-CNN
source ENV/bin/activate

cd ~/scratch/Drybean-CNN/Drybean_DL-gwas-master/BLUE/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 BLUE.py imputed_GenotypicData.vcf Raw_GenotypicData.vcf --pheno pheno_normalized.tsv
python3 BLUE.py  imputed_GenotypicData_processed.tsv  Raw_GenotypicData_processed.tsv  --summary
