#!/usr/bin/env python3
import subprocess
import argparse
import os
import shutil
import tempfile

def inline_vcf_ld_prune(vcf_in, window_snps=50, step_snps=5, r2_threshold=0.9):
    """
    Prunes a VCF file at an r2 threshold of 0.9 and overwrites/outputs
    the result back to a VCF format safely using isolated workspaces.
    """
    print(f"[*] Starting LD Pruning on: {vcf_in} (r2={r2_threshold})")

    # 1. Create an isolated temporary directory to completely avoid collisions
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_prefix = os.path.join(tmpdir, "plink_workspace")
        temp_vcf_out = os.path.join(tmpdir, "temp_filtered_final_output")

        # 2. Run pairwise sliding-window linkage detection
        plink_detect_cmd = [
            "plink",
            "--vcf", vcf_in,
            "--indep-pairwise", str(window_snps), str(step_snps), str(r2_threshold),
            "--out", temp_prefix,
            "--allow-extra-chr"
        ]
        subprocess.run(plink_detect_cmd, check=True, stdout=subprocess.DEVNULL)

        # 3. Extract unlinked variants list and write directly to VCF format
        plink_export_cmd = [
            "plink",
            "--vcf", vcf_in,
            "--extract", f"{temp_prefix}.prune.in",
            "--keep-allele-order",  # Keep REF/ALT order
            "--recode", "vcf-iid",  # Maintain original sample IDs
            "--out", temp_vcf_out,
            "--allow-extra-chr"
        ]
        subprocess.run(plink_export_cmd, check=True, stdout=subprocess.DEVNULL)

        # PLINK appends '.vcf' automatically to the --recode path
        actual_plink_vcf = f"{temp_vcf_out}.vcf"

        # 4. Safely overwrite the original VCF file with the newly pruned subset
        shutil.move(actual_plink_vcf, vcf_in)

    print(f"[+] Overwrote {vcf_in} successfully with independent epistatic-safe variants.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="In-place VCF LD Pruner Subprocess Component")
    parser.add_argument("--vcf", required=True, help="Path to targeted VCF file")
    args = parser.parse_args()

    inline_vcf_ld_prune(args.vcf, window_snps=50, step_snps=5, r2_threshold=0.9)
