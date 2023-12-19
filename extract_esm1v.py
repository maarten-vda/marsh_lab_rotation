import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate a TSV file with ESM-1v scores for disease-associated residues")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_tsv", help="Path to the output TSV file")

    args = parser.parse_args()

    # Read the input TSV file
    disease_df = pd.read_csv(args.input_tsv, sep='\t')

    # Create a new DataFrame to store the output data
    output_df = disease_df.copy()

    # Add columns for each amino acid substitution
    amino_acids = "ARNDCQEGHILKMFPSTWVY"
    for aa in amino_acids:
        # Create a new column with the amino acid name and initialize it with a default value
        column_name = aa
        output_df[column_name] = 0  # Set Default value to 0 because that is variant score for the same amino acid substitution


    with open(args.output_tsv, 'w', newline='') as output_file:
        for _, row in disease_df.iterrows():
            uniprot_id = row['uniprot_id']
            variant = row['variant']


            # Load the corresponding CSV file
            csv_file = os.path.join(args.csv_folder, f"{uniprot_id}.csv")
            if not os.path.exists(csv_file):
                print(f"CSV file not found for uniprot_id: {uniprot_id}")
                continue

            # Read the CSV file into a DataFrame
            csv_df = pd.read_csv(csv_file)

            # Extract numeric residues from 'variant' column in csv_df
            csv_df['residue'] = csv_df['variant'].str.extract('(\d+)', expand=False).astype(float)

            try:
                # Filter rows where the 'residue' matches the variant from disease_df
                variant_scores = csv_df[csv_df['residue'] == float(variant)][['variant', 'ESM-1v']]
            except KeyError:
                print(f"KeyError for uniprot_id: {uniprot_id}")
                continue

            # Iterate over rows in variant_scores
            for _, v_row in variant_scores.iterrows():
                residue = v_row['variant'][1:-1]  # Extract residue from the variant
                substitution = v_row['variant'][-1]  # Extract substitution from the variant
                esm_1v_score = v_row['ESM-1v']

                # Convert the 'variant' column to string before comparison
                output_df['variant'] = output_df['variant'].astype(str)

                # Update the cell with the ESM-1v score in the original DataFrame
                output_df.loc[(output_df['variant'] == residue) & (output_df['uniprot_id'] == uniprot_id), substitution] = esm_1v_score

        output_file.write(output_df.to_csv(sep='\t', index=False))  # Write the DataFrame to the

if __name__ == "__main__":
    main()
