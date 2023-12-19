import argparse
import csv
import random

def random_subset(input_file, output_file, n):
    with open(input_file, 'r') as in_file, open(output_file, 'w', newline='') as out_file:
        reader = csv.reader(in_file, delimiter='\t')
        writer = csv.writer(out_file, delimiter='\t')

        # Read the header row and write it to the output file
        header = next(reader)
        writer.writerow(header)

        # Read the data into a list
        data = list(reader)

        # Check if n is greater than the number of available rows
        if n > len(data):
            print(f"Warning: The requested subset size {n} is greater than the available rows. Using all rows.")
            n = len(data)

        # Randomly select n rows
        random_subset = random.sample(data, n)

        # Write the selected rows to the output file
        for row in random_subset:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Generate a random subset of rows from a TSV file')
    parser.add_argument('-n', type=int, required=True, help='Number of rows in the subset')
    parser.add_argument('input_file', type=str, help='Input TSV file')
    parser.add_argument('output_file', type=str, help='Output TSV file')
    
    args = parser.parse_args()
    
    random_subset(args.input_file, args.output_file, args.n)
    print(f"Random subset of {args.n} rows written to {args.output_file}")

if __name__ == "__main__":
    main()
