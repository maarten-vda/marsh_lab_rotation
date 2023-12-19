import csv
import argparse
from collections import defaultdict

def find_minimum_and_append(input_file, input_file2, output_file, n):
    with open(input_file2, 'r') as filtered_variants:
        reader = csv.reader(filtered_variants, delimiter='\t')
        next(reader)  # Skip the header row
        class_dict = defaultdict()

        for row in reader:
            if row:
                uniprot_id = row[0]
                variant = row[5]
                variant = ''.join(filter(lambda char: not char.isalpha() and char != '*', variant))
                var_class = row[6]
                if var_class == 'LP':
                    var_class = 'P'
                unique_id = f'{uniprot_id}_{variant}'
                class_dict[unique_id] = var_class

    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile)
        next(reader)  # Skip the header row
        for row in reader:
            if row:
                header_value = row[0]
                non_empty_values = [value for value in row[1:] if value != '']
                if len(non_empty_values) >= n:
                    sorted_values = sorted(map(float, non_empty_values))
                    min_value = sum(sorted_values[-n:]) / n
                    class_value = class_dict.get(header_value, "Class Not Found")
                    writer.writerow([header_value, min_value, class_value])
                else:
                    min_value = sum(sorted(map(float, non_empty_values))) / len(non_empty_values) if len(non_empty_values) != 0 else ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find minimum values in a CSV file and append them to a new CSV file")
    parser.add_argument("input_file", help="Distances matrix")
    parser.add_argument("input_file2", help="Filtered variants TSV")
    parser.add_argument("output_file", help="Output CSV file")
    parser.add_argument("-n", type=int, default=1, help="Number of values to average (default: 1)")

    args = parser.parse_args()

    find_minimum_and_append(args.input_file, args.input_file2, args.output_file, args.n)
