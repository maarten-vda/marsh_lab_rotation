import pandas as pd
import argparse

def merge_tsv_files(input_file1, input_file2, output_file):
    # Read the first TSV file
    df1 = pd.read_csv(input_file1, sep='\t')

    # Read the second TSV file
    df2 = pd.read_csv(input_file2, sep='\t')

    # Add the 'class' column to the second dataframe and set it to 'B'
    df2['class'] = 'B'

    # Check if all columns in columns_to_keep exist in both dataframes
    columns_to_keep = ['uniprot_id', 'chr', 'pos', 'ref', 'alt', 'variant', 'class']
    missing_columns_df1 = [col for col in columns_to_keep if col not in df1.columns]
    missing_columns_df2 = [col for col in columns_to_keep if col not in df2.columns]

    if missing_columns_df1:
        print(f"Columns missing in {input_file1}: {', '.join(missing_columns_df1)}")
        return

    if missing_columns_df2:
        print(f"Columns missing in {input_file2}: {', '.join(missing_columns_df2)}")
        return

    # Merge the two dataframes
    merged_df = pd.concat([df1[columns_to_keep], df2[columns_to_keep]])

    # Save the merged dataframe to a new TSV file
    merged_df.to_csv(output_file, sep='\t', index=False)

    print(f'Merged data saved to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two TSV files.")
    parser.add_argument("input_file1", help="First input TSV file")
    parser.add_argument("input_file2", help="Second input TSV file")
    parser.add_argument("output_file", help="Output merged TSV file")
    
    args = parser.parse_args()

    merge_tsv_files(args.input_file1, args.input_file2, args.output_file)
