def vcf_preprocessing(IMP_input, output_path =None):
	from pysam import VariantFile
	import pandas as pd
	import os
	vcf = VariantFile(IMP_input)
	data = []
	index = []

	for record in vcf:
		index.append(record.id if record.id else f"{record.chrom}_{record.pos}")

		gts = [sum(s['GT']) if s['GT'] and None not in s['GT'] else int(-1)
		       for s in record.samples.values()]

		data.append(gts)
	# Create the final DataFrame
	df = pd.DataFrame(data, index=index, columns=list(vcf.header.samples))
	df_T = df.T
	df_final = df_T.reset_index().rename(columns={'index': 'LINE'})
	df_final['LINE'] = df_final['LINE'].str.replace(r'\.\d+$', '', regex=True)

	if output_path is None:
		base = os.path.splitext(IMP_input)[0]
		output_path = f"{base}_processed.tsv"

	df_final.to_csv(output_path, sep='\t', index=False)
	return output_path

IMP_input="2026-04-20-GenotypicData-UoG_corrected.vcf"
vcf_preprocessing(IMP_input)

