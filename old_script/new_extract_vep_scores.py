import argparse
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

def process_row(args, row):
    uniprot_id = row['uniprot_id']
    variant = row['variant']
    residue = variant[1:-1]

    csv_file = os.path.join(args.csv_folder, f"{uniprot_id}.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file not found for uniprot_id: {uniprot_id}")
        return None

    try:
        csv_df = pd.read_csv(csv_file, usecols=['variant', args.vep])
    except KeyError:
        print(f"KeyError for uniprot_id: {uniprot_id}")
        return None

    try:
        residue_filter = csv_df['variant'].str[1:-1] == residue
        variant_scores = csv_df.loc[residue_filter, ['variant', args.vep]]
    except KeyError:
        print(f"KeyError for uniprot_id: {uniprot_id}")
        return None

    result_dict = {'uniprot_id': uniprot_id, 'variant': variant, 'class': row['class'], 'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}
    for amino_acid in 'ARNDCQEGHILKMFPSTWY':
        match_filter = (variant_scores['variant'].str[-1] == amino_acid) & (variant_scores['variant'].str[1:-1] == residue)
        matched_rows = variant_scores.loc[match_filter]

        if not matched_rows.empty:
            max_score_index = np.argmax(matched_rows[args.vep].values)
            max_score_row = matched_rows.iloc[max_score_index]
            result_dict[amino_acid] = max_score_row[args.vep]
        else:
            result_dict[amino_acid] = None

    return result_dict

def main():
    parser = argparse.ArgumentParser(description="Generate a TSV file with scores for disease-associated residues")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_tsv", help="Path to the output TSV file")
    parser.add_argument("--vep", choices=['ESM-1v', 'ESM-1b', 'REVEL', 'AlphaMissense'], default='AlphaMissense', help="VEP whose scores to retrieve")

    args = parser.parse_args()

    disease_df = pd.read_csv(args.input_tsv, sep='\t')

    with Pool(processes=16) as pool:
        results = [result for result in pool.starmap(process_row, [(args, row) for _, row in disease_df.iterrows()]) if result is not None]

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_tsv, sep='\t', index=False)

if __name__ == "__main__":
    main()
 
