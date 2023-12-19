import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate a TSV file with scores for disease-associated residues")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("csv_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_tsv", help="Path to the output TSV file")
    parser.add_argument("--vep", choices=['ESM-1v', 'ESM-1b', 'REVEL', 'AlphaMissense'], default='AlphaMissense', help="VEP whose scores to retrieve")

    args = parser.parse_args()

    # Read the input TSV file
    disease_df = pd.read_csv(args.input_tsv, sep='\t')

    # Create a new DataFrame to store the output data
    output_df = disease_df[['uniprot_id', 'variant', 'class']].copy()

    with open(args.output_tsv, 'w', newline='') as output_file:
        counter = 0
        for _, row in disease_df.iterrows():
            counter += 1
            print(f"Processing row {counter} of {len(disease_df)}")
            uniprot_id = row['uniprot_id']
            variant = row['variant']
            residue = variant[1:-1]
            substitution = variant[-1]

            # Load the corresponding CSV file
            csv_file = os.path.join(args.csv_folder, f"{uniprot_id}.csv")
            if not os.path.exists(csv_file):
                print(f"CSV file not found for uniprot_id: {uniprot_id}")
                continue

            # Read the CSV file into a DataFrame
            csv_df = pd.read_csv(csv_file)[['variant', args.vep]]

            try:
                # Filter rows where the 'residue' matches the variant from disease_df
                variant_scores = csv_df.loc[csv_df['variant'].str[1:-1] == residue, ['variant', args.vep]]
            except KeyError:
                print(f"KeyError for uniprot_id: {uniprot_id}")
                continue

            # Iterate over rows in variant_scores
            for _, v_row in variant_scores.iterrows():
                residue2 = v_row['variant'][1:-1]  # Extract residue from the variant
                substitution2 = v_row['variant'][-1]  # Extract substitution from the variant
                vep_score = v_row[args.vep]

                if substitution2 == substitution and residue2 == residue:
                    # Convert the 'variant' column to string before comparison
                    output_df['variant'] = output_df['variant'].astype(str)
                    output_df['score'] = vep_score

        output_file.write(output_df.to_csv(sep='\t', index=False))  # Write the DataFrame to the

if __name__ == "__main__":
    main()
