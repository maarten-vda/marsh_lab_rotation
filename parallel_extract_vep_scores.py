import argparse
import os
import pandas as pd
from multiprocessing import Pool

def process_row(args, row):
    uniprot_id = row['uniprot_id']
    variant = row['variant']
    residue = variant[1:-1]
    substitution = variant[-1]

    csv_file = os.path.join(args.csv_folder, f"{uniprot_id}.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file not found for uniprot_id: {uniprot_id}")
        return None

    try:
        csv_df = pd.read_csv(csv_file)[['variant', args.vep]]
    except KeyError:
        print(f"KeyError for uniprot_id: {uniprot_id}")
        return None

    try:
        variant_scores = csv_df.loc[csv_df['variant'].str[1:-1] == residue, ['variant', args.vep]]
    except KeyError:
        print(f"KeyError for uniprot_id: {uniprot_id}")
        return None

    for _, v_row in variant_scores.iterrows():
        residue2 = v_row['variant'][1:-1]
        substitution2 = v_row['variant'][-1]
        vep_score = v_row[args.vep]

        if substitution2 == substitution and residue2 == residue:
            return {'uniprot_id': uniprot_id, 'variant': variant, 'score': vep_score}
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate a TSV file with scores for disease-associated residues")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_tsv", help="Path to the output TSV file")
    parser.add_argument("--vep", choices=['ESM-1v', 'ESM-1b', 'REVEL', 'AlphaMissense'], default='AlphaMissense', help="VEP whose scores to retrieve")

    args = parser.parse_args()

    disease_df = pd.read_csv(args.input_tsv, sep='\t')
    output_df = disease_df[['uniprot_id', 'variant', 'class']].copy()

    with Pool(processes=16) as pool:  # Adjust the number of processes based on your needs
        results = [result for result in pool.starmap(process_row, [(args, row) for _, row in disease_df.iterrows()]) if result is not None]

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_tsv, sep='\t', index=False)

if __name__ == "__main__":
    main()
