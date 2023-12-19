import argparse
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

def process_row(args, row, csv_df):
    uniprot_id = row['uniprot_id']
    variant = row['variant']
    residue = variant[1:-1]

    try:
        residue_filter = csv_df['variant'].str[1:-1] == residue
        variant_scores = csv_df.loc[residue_filter, ['variant', args.vep]]
    except KeyError:
        print(f"KeyError for uniprot_id: {uniprot_id}")
        return None

    result_dict = {'uniprot_id': uniprot_id, 'variant': variant, 'class': row['class'], 'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}
    for amino_acid in 'ARNDCQEGHILKMFPSTWYV':
        match_filter = (variant_scores['variant'].str[-1] == amino_acid) & (variant_scores['variant'].str[1:-1] == residue)
        matched_rows = variant_scores.loc[match_filter]

        if not matched_rows.empty:
            max_score_index = np.argmax(matched_rows[args.vep].values)
            max_score_row = matched_rows.iloc[max_score_index]
            result_dict[amino_acid] = max_score_row[args.vep]
        else:
            result_dict[amino_acid] = 0

    return result_dict

def process_chunk(args, chunk, csv_folder):
    results = []
    for _, row in chunk.iterrows():
        uniprot_id = row['uniprot_id']
        csv_file = os.path.join(csv_folder, f"{uniprot_id}.csv")
        if not os.path.exists(csv_file):
            print(f"CSV file not found for uniprot_id: {uniprot_id}")
            continue
        try:
            csv_df = pd.read_csv(csv_file, usecols=['variant', args.vep])
        except ValueError as e:
            print(f"Error reading CSV for uniprot_id {uniprot_id}: {e}")
            # Print the column names for debugging purposes
            with open(csv_file, 'r') as f:
                header_line = f.readline().strip()
                column_names = header_line.split(',')
                print(f"Column names in CSV file: {column_names}")
            continue

        result = process_row(args, row, csv_df)
        if result is not None:
            results.append(result)

    return results

def main():
    parser = argparse.ArgumentParser(description="Generate a TSV file with scores for disease-associated residues")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_tsv", help="Path to the output TSV file")
    parser.add_argument("--vep", choices=['ESM-1v', 'ESM-1b', 'REVEL', 'AlphaMissense'], default='AlphaMissense', help="VEP whose scores to retrieve")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of rows to process per chunk")

    args = parser.parse_args()

    disease_df = pd.read_csv(args.input_tsv, sep='\t', iterator=True, chunksize=args.chunk_size)

    with Pool(processes=16) as pool:
        results = pool.starmap(process_chunk, [(args, chunk, args.csv_folder) for chunk in disease_df])

    # Concatenate results from all chunks
    results = [result for sublist in results for result in sublist]
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_tsv, sep='\t', index=False)

if __name__ == "__main__":
    main()
