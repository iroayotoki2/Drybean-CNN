#!/bin/bash
#SBATCH --account=def-cottenie
#SBATCH --job-name=drybean_blue
#SBATCH --output=drybean_blue_%j.out
#SBATCH --error=drybean_blue_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module load python/3.10

cd ~/scratch/Drybean-CNN
source ENV/bin/activate

cd ~/scratch/Drybean-CNN/Drybean_DL-gwas-master/BLUE/

python3 BLUE.py imputed_GenotypicData.vcf Raw_GenotypicData.vcf --pheno pheno_normalized.tsv

