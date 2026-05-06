# Drybean-CNN

A saliency map approach was applied to measure phenotype contribution for genome wide association study.  It creates output files on average saliency graph along with top SNPs with highest average saliency. On program completion, it provides the average PCC (Pearson correlation coefficient) along with a prompt for the user to search saliency based on SNP name. The program is implemented using Keras3.5 and Tensorflow backend with python 3.1. It was adapted from the [Soybean-CNN](https://github.com/ProductiveOwl/Soybean-CNN) research by Jake Goode, Madhurika Madhu, Raeein Bagheri, and Yan Yan.

### Prerequisites

Python packages required:

```
absl_py==2.1.0 
astunparse==1.6.3 
certifi==2025.1.31 
charset_normalizer==3.4.1 
contourpy==1.3.1 
cycler==0.12.1 
flatbuffers==25.1.24 
fonttools==4.56.0 
gast==0.6.0 
google-pasta==0.2.0 
grpcio==1.67.0 
h5py==3.12.0 
idna==3.10 
joblib==1.4.2 
keras==3.5.0 
kiwisolver==1.4.8 
libclang==14.0.1 
Markdown==3.7 
markdown_it_py==3.0.0 
MarkupSafe==2.1.5 
matplotlib==3.10.0 
mdurl==0.1.2 
ml_dtypes==0.4.0 
namex==0.0.8 
np_utils==0.6.0
numpy==1.26.4 
opt_einsum==3.4.0 
optree==0.12.1 
packaging==24.2 
pandas==2.2.3 
pillow==11.1.0 
protobuf==4.25.4 
pygments==2.19.1 
pyparsing==3.2.3
pysam==0.24.0
python_dateutil==2.9.0.post0 
pytz==2025.2
requests==2.32.3 
rich==13.9.4 
scikit_learn==1.5.2 
scipy==1.15.1 
six==1.17.0 
sklearn==0.0 
tensorboard==2.17.1 
tensorboard_data_server==0.7.2 
tensorflow==2.17.0 
tensorflow_io_gcs_filesystem==0.32.0 
termcolor==2.5.0 
threadpoolctl==3.6.0 
typing_extensions==4.12.2 
tzdata==2025.2
urllib3==2.3.0 
werkzeug==3.1.3 
wrapt==1.17.2

```

## Running the program

The scripts train and test model with 10 fold cross validation and plot a comparison of genotype contribution using saliency map value and Wald test value.

* **BLUE.py** - *Executive scripts.*
* **run_all_folds.py** - *Runs all 10 folds individually.*
* **summarize_folds.py** - *Summarizes all 10 folds without re-running program.*
* **requirements.txt** - *Requirements of Python modules for program.*
* **imputed_GenotypicData.vcf/** - *Inputs of imputed genotype matrix.*
* **Raw_GenotypicData.vcf/Raw_GenotypicData_UoG.csv** - *Inputs of quality assured non-imputed genotype matrix which ca be run with either vcf or csv input.*


In  python file, change the global variable NUM_FOLDS to the folds you wish to run.

### Running on Digital Alliance Servers

After SSH'ing into servers, preform the following to run the CNN for the BLUEs.


```
module load python/3.10
virtualenv --no-download ENV
source ENV/bin/activate
cd BLUE
python3 BLUE.py IMP_file QA_file --pheno pheno_file

alternative:
python3 BLUE.py IMP_file QA_file --pheno pheno_file --fold 3 (Runs individual fold)
```

Alternative (SLURM Job):
 This can be run with the SLURM script CNN_slurm.sh. Edit the account name seen in line 1 (#SBATCH --account=def-cottenie) to your own account (sshare -U). Run with: 
 ```
sbatch CNN_slurm.sh
 ```
Summarize after 10 folds completed:
```
python3 summarize_folds.py IMP_file QA_file

alternative:

python3 BLUE.py IMP_file QA_file --pheno pheno_file  --summary
```



## License
GNU v2.0
