import pandas as pd
import sys

def main(input_file, output_file):
    try:
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(input_file, sep='\t')

        # Extract the first and sixth columns
        df = df.iloc[:, [0, 5]]

        # Remove letters from the entries in the sixth column while preserving numbers
        df.iloc[:, 1] = df.iloc[:, 1].str.replace(r'[a-zA-Z]', '')

        # Drop duplicate rows
        df = df.drop_duplicates()

        # Save the resulting DataFrame to a TSV file
        df.to_csv(output_file, sep='\t', index=False)
        
        print(f"Processed and saved the result to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.tsv output.tsv")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)
