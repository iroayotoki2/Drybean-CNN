#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=drybean_blue
#SBATCH --output=drybean_blue_%j.out
#SBATCH --error=drybean_blue_%j.err

set -euo pipefail

cd ~/scratch/Drybean-CNN/Drybean_DL-gwas-master/BLUE/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load StdEnv/2020
module load plink/1.9b_6.21-x86_64

which plink
plink --version

python ld_pruning.py --vcf imputed_GenotypicData.vcf
python ld_pruning.py --vcf Raw_GenotypicData.vcf

module load StdEnv/2023
module load python/3.10

cd ~/scratch/Drybean-CNN
source ENV/bin/activate

cd ~/scratch/Drybean-CNN/Drybean_DL-gwas-master/BLUE/

which python
python --version

python BLUE.py LD_imputed_GenotypicData.vcf LD_Raw_GenotypicData.vcf --pheno pheno_normalized.tsv
python BLUE.py LD_imputed_GenotypicData_processed.tsv LD_Raw_GenotypicData_processed.tsv --summary
