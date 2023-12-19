import argparse
import csv
import numpy as np

def read_tsv(file_path):
    data = []
    with open(file_path, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def calculate_distribution(similarities_data):
    flat_data = [float(value) for row in similarities_data[1:] for value in row[1:] if value]
    sorted_data = np.sort(flat_data)
    threshold_index = int(0.9 * len(sorted_data))
    threshold = sorted_data[threshold_index]
    linspace_threshold = np.linspace(0.5*max(flat_data), max(flat_data), num=100)
    return linspace_threshold.tolist()

def main():
    parser = argparse.ArgumentParser(description="Extract similarities based on effect TSV")
    parser.add_argument("effect_tsv", help="Path to the effect TSV file")
    parser.add_argument("similarities_tsv", help="Path to the similarities TSV file")
    args = parser.parse_args()

    try:
        effect_data = read_tsv(args.effect_tsv)
        similarities_data = read_tsv(args.similarities_tsv)
    except Exception as exception:
        print(f"Error reading files: {exception}")
        return

    for threshold in calculate_distribution(similarities_data):
        correct_predictions = 0
        total_predictions = 0
        for i in range(1, len(effect_data)):
            for j in range(1, len(effect_data[i])):
                if i != j:
                    if effect_data[i][j] != "":
                        row_header = effect_data[i][0]
                        col_header = effect_data[0][j]

                        # Check if row_header and col_header exist in similarities_data
                        if row_header in similarities_data[0] and col_header in similarities_data[0]:
                            row_index = similarities_data[0].index(row_header)
                            col_index = similarities_data[0].index(col_header)
                            similarity_value = similarities_data[row_index][col_index]
                            if similarity_value and float(similarity_value) <= threshold:
                                similarity_value = 1
                            else:
                                similarity_value = 0
                            if similarity_value == int(effect_data[i][j]):
                                correct_predictions += 1
                            total_predictions += 1
                        else:
                            # Skip if either row_header or col_header is not in similarities_data
                            continue
                else:
                    continue

        print(threshold, correct_predictions / total_predictions)

if __name__ == "__main__":
    main()
