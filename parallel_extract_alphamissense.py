import argparse
import os
import pandas as pd
from multiprocessing import Pool, cpu_count

def process_variant_score(args):
    disease_df, output_df, row, csv_folder = args
    uniprot_id = row['uniprot_id']
    variant = row['variant']
    print(uniprot_id)

    csv_file = os.path.join(csv_folder, f"{uniprot_id}.csv")
    if not os.path.exists(csv_file):
        print(f"CSV file not found for uniprot_id: {uniprot_id}")
        return

    csv_df = pd.read_csv(csv_file)
    csv_df['residue'] = csv_df['variant'].str.extract('(\d+)', expand=False).astype(float)

    try:
        variant_scores = csv_df[csv_df['residue'] == float(variant)][['variant', 'AlphaMissense']]
    except KeyError:
        print(f"KeyError for uniprot_id: {uniprot_id}")
        return

    for _, v_row in variant_scores.iterrows():
        residue = v_row['variant'][1:-1]
        substitution = v_row['variant'][-1]
        esm_1v_score = v_row['AlphaMissense']

        output_df['variant'] = output_df['variant'].astype(str)
        output_df.loc[(output_df['variant'] == residue) & (output_df['uniprot_id'] == uniprot_id), substitution] = esm_1v_score

def main():
    parser = argparse.ArgumentParser(description="Generate a TSV file with ESM-1v scores for disease-associated residues")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_tsv", help="Path to the output TSV file")

    args = parser.parse_args()

    disease_df = pd.read_csv(args.input_tsv, sep='\t')
    output_df = disease_df.copy()

    amino_acids = "ARNDCQEGHILKMFPSTWVY"
    for aa in amino_acids:
        column_name = aa
        output_df[column_name] = 0

    # Define the number of worker processes (adjust as needed)
    num_processes = min(cpu_count(), len(disease_df))

    # Create a Pool of worker processes
    pool = Pool(processes=num_processes)

    # Create a list of arguments for the process_variant_score function
    args_list = [(disease_df, output_df, row, args.csv_folder) for _, row in disease_df.iterrows()]

    # Use the Pool to parallelize processing
    pool.map(process_variant_score, args_list)
    pool.close()
    pool.join()

    with open(args.output_tsv, 'w', newline='') as output_file:
        output_file.write(output_df.to_csv(sep='\t', index=False))

if __name__ == "__main__":
    main()
