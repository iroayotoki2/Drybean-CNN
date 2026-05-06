import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('IMP_file', help="Imputed file")
parser.add_argument('QA_file', help="QA file")
args = parser.parse_args()

NUM_FOLDS = 10

print(f"Updating data files with {NUM_FOLDS} folds.")

with open("fold_pcc_log.csv", "w") as f:
    f.write("")  # Truncate the file

for fold in range(1, NUM_FOLDS + 1):
    print(f"\n🚀 Running fold {fold}...\n")
    result = subprocess.run(['python3', 'BLUE.py', args.IMP_file, args.QA_file, '--fold', str(fold)])

    if result.returncode != 0:
        print(f"❌ Fold {fold} failed with return code {result.returncode}")
        break

# After all folds, run the full summary block
print("\n✅ All folds completed. Generating saliency summary and top SNPs...\n")
subprocess.run(['python3', 'BLUE.py', args.IMP_file, args.QA_file, '--summary'])
