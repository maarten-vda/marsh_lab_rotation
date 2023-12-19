import pandas as pd
import argparse
import csv
import matplotlib.pyplot as plt

def create_unique_identifier(row):
    uniprot_id = row['uniprot_id']
    variant = ''.join(filter(str.isdigit, row['variant']))
    return f"{uniprot_id}_{variant}"

class_combinations = {'P_P': [], 'P_B': [], 'B_B': []}

def main(filtered_variants_file, distances_file, output_directory):
    class_to_unique_ids = {}

    with open(filtered_variants_file, 'r', newline='') as fvf:
        reader = csv.DictReader(fvf, delimiter='\t')
        for row in reader:
            unique_id = create_unique_identifier(row)
            class_value = row['class']
            class_to_unique_ids[unique_id] = class_value

    distances_df = pd.read_csv(distances_file, delimiter='\t')

    for col_index in distances_df.columns[1:]:
        col_class = class_to_unique_ids.get(col_index, "B")  # Default to "B" if class is not found

        if col_class == "LP":
            col_class = "P"

        counter = 0
        for row_index in distances_df.columns[1:]:
            row_class = class_to_unique_ids.get(row_index, "B")  # Default to "B" if class is not found

            if row_class == "LP":
                row_class = "P"

            combination = f"{row_class}_{col_class}"
            if combination == "B_P":
                combination = "P_B"

            class_combinations[combination].append(distances_df[col_index][counter])
            counter += 1

    for combination, values in class_combinations.items():
        plt.hist(values, bins=20, alpha=0.5, label=combination)
        plt.legend()
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Scores for {combination} Class Combination")
        
        output_file = f"{output_directory}/{combination}.png"
        plt.savefig(output_file)
        plt.clf()  # Clear the plot for the next combination

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process TSV files")
    parser.add_argument('filtered_variants_file', help="Path to the filtered_variants TSV file")
    parser.add_argument('distances_file', help="Path to the distances TSV file")
    parser.add_argument('output_directory', help="Directory to save output PNG files")
    args = parser.parse_args()

    main(args.filtered_variants_file, args.distances_file, args.output_directory)
