import os
import argparse
import pandas as pd

def process_input_tsv(input_tsv, input_folder, output_tsv, vep):
    # Read the input TSV into a DataFrame
    input_df = pd.read_csv(input_tsv, sep='\t')

    # Initialize an empty DataFrame for output
    output_df = input_df[['uniprot_id', 'variant', 'class']].copy()

    # Add columns for every amino acid
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    output_df[amino_acids] = pd.DataFrame(index=output_df.index, columns=amino_acids)

    # Loop over entries in the input DataFrame
    for index, row in input_df.iterrows():
        print(f"Processing row {index + 1} of {len(input_df)}")
        uniprot_id = row['uniprot_id']
        variant = row['variant']
        residue = variant[1:-1]
        substitution = variant[-1]

        # Load the corresponding CSV file
        csv_file = os.path.join(input_folder, f"{uniprot_id}.csv")
        if not os.path.exists(csv_file):
            print(f"CSV file not found for uniprot_id: {uniprot_id}")
            continue

        # Read the CSV file into a DataFrame
        try:
            csv_df = pd.read_csv(csv_file)[['variant', vep]]
        except pd.errors.EmptyDataError:
            print(f"Empty CSV file for uniprot_id: {uniprot_id}")
            continue

        # Filter rows where the 'residue' matches the variant from input_df
        variant_scores = csv_df[csv_df['variant'].str[1:-1] == residue][['variant', vep]]

        # Update output_df with scores for each amino acid
        for _, v_row in variant_scores.iterrows():
            residue2 = v_row['variant'][1:-1]
            substitution2 = v_row['variant'][-1]
            vep_score = v_row[vep]

            if substitution2 == substitution and residue2 == residue:
                # Update the score for the corresponding amino acid
                output_df.at[index, residue] = vep_score

    # Save the output DataFrame to a new TSV file
    output_df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Output saved to {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TSV and CSV files to create an output TSV.")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_tsv", help="Path to the output file")
    parser.add_argument("--vep", choices=['ESM-1v', 'ESM-1b', 'REVEL', 'AlphaMissense'], default='AlphaMissense', help="VEP whose scores to retrieve")

    args = parser.parse_args()
    process_input_tsv(args.input_tsv, args.input_folder, args.output_tsv, args.vep)
