import argparse
import pandas as pd

# Define a function to filter the TSV data
def filter_tsv(input_file, output_file):
    # Read the TSV file into a Pandas DataFrame
    df = pd.read_csv(input_file, sep='\t')

    df.columns = df.columns.str.strip()

    # Clean and normalize the 'class' column by removing leading/trailing whitespace
    df['class'] = df['class'].str.strip()

    # Create a set of unique uniprot_ids with class 'LP' or 'P'
    unique_ids_with_lp_or_p = set(df[(df['class'] == 'LP') | (df['class'] == 'P')]['uniprot_id'])

    # Filter rows based on conditions
    condition1 = df['class'] == 'B'  # Rows where 'class' is 'B'
    condition2 = df['uniprot_id'].isin(unique_ids_with_lp_or_p)  # Rows with 'uniprot_id' in the set
    condition3 = (df['class'] == 'LP') | (df['class'] == 'P')  # Rows where 'class' is 'LP' or 'P'

    # Combine conditions using the "or" operator to include rows with class 'LP' or 'P'
    combined_condition = (condition1 & condition2) | condition3

    # Apply condition to remove duplicates based on 'uniprot_id', 'chr', 'pos', and 'variant'
    filtered_rows = df[combined_condition].drop_duplicates(subset=['uniprot_id', 'chr', 'pos', 'variant'])

    # Write the filtered dataframe to a new TSV file
    filtered_rows.to_csv(output_file, sep='\t', index=False)

def main():
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Filter a TSV file using Pandas.')
    parser.add_argument('input', help='Input TSV file name')
    parser.add_argument('output', help='Output file name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the filter_tsv function to process the input file and create the filtered output
    filter_tsv(args.input, args.output)

if __name__ == '__main__':
    main()
