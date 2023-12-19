import os
import argparse
import pandas as pd
from multiprocessing import Pool

def process_chunk(args, chunk):
    results = []

    for _, row in chunk.iterrows():
        uniprot_id = row['uniprot_id']
        variant = row['variant']
        residue = variant[1:-1]

        csv_file = os.path.join(args.input_folder, f"{uniprot_id}.csv")
        if not os.path.exists(csv_file):
            print(f"CSV file not found for uniprot_id: {uniprot_id}")
            continue

        try:
            csv_df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            print(f"Empty CSV file for uniprot_id: {uniprot_id}")
            continue

        try:
            variant_scores = csv_df.loc[csv_df['variant'].str[1:-1] == residue, ['variant', args.vep]]
        except KeyError:
            print(f"KeyError for uniprot_id: {uniprot_id}")
            continue
        
        del csv_df

        output_dict = {'uniprot_id': uniprot_id, 'variant': variant, "A": 0, "R": 0, "N": 0, "D": 0, "C": 0, "Q": 0, "E": 0, "G": 0, "H": 0, "I": 0, "L": 0, "K": 0, "M": 0, "F": 0, "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0}

        def process_variant_score(v_row):
            substitution2 = v_row['variant'][-1]
            vep_score = v_row[args.vep]

            if substitution2 == variant[-1]:
                output_dict[variant[-1]] = vep_score

        variant_scores.apply(process_variant_score, axis=1)

        results.append(output_dict)

    return results

def main():
    parser = argparse.ArgumentParser(description="Process TSV and CSV files to create an output TSV.")
    parser.add_argument("input_tsv", help="Path to the input TSV file")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("output_tsv", help="Path to the output file")
    parser.add_argument("--vep", choices=['ESM-1v', 'ESM-1b', 'REVEL', 'AlphaMissense'], default='AlphaMissense', help="VEP whose scores to retrieve")

    args = parser.parse_args()

    # Specify chunksize when reading the input TSV into a DataFrame
    chunksize = 1000  # Adjust the chunksize based on your system's memory constraints
    input_chunks = pd.read_csv(args.input_tsv, sep='\t', chunksize=chunksize)

    results = []

    # Create a pool of processes
    with Pool(processes=32) as pool:  # Adjust the number of processes based on your needs
        # Use starmap to process chunks in parallel
        results.extend(pool.starmap(process_chunk, [(args, chunk) for chunk in input_chunks]))

    # Flatten the list of results
    results = [item for sublist in results for item in sublist]

    # Create the output DataFrame from the results
    output_df = pd.DataFrame(results)

    # Save the output DataFrame to a new TSV file
    output_df.to_csv(args.output_tsv, sep='\t', index=False)
    print(f"Output saved to {args.output_tsv}")

if __name__ == "__main__":
    main()
