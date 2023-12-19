import argparse
import pandas as pd

def clean_tsv(input_file, output_file):
    # Read the TSV file into a Pandas DataFrame
    df = pd.read_csv(input_file, sep='\t')

    # Clean and normalize all columns (remove leading/trailing whitespace)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Write the cleaned DataFrame to a new TSV file
    df.to_csv(output_file, sep='\t', index=False)

def main():
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description='Clean a TSV file.')
    parser.add_argument('input', help='Input TSV file name')
    parser.add_argument('output', help='Output file name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the clean_tsv function to clean the input file and create the cleaned output
    clean_tsv(args.input, args.output)

if __name__ == '__main__':
    main()
