import argparse
import numpy as np
import csv

def process_tsv(tsv_file, output_file, n):
    with open(tsv_file, 'r') as f:
        lines = f.readlines()

    header = np.array(lines[0].strip().split('\t'))
    data = [line.strip().split('\t') for line in lines[1:]]

    results = []
    for row in data:
        row_index = row[0]
        row_data = np.array(row[1:], dtype=float)
        top_n_indices = np.argpartition(row_data, -n)[-n:]
        top_n_indices = top_n_indices.astype(int).ravel()  # Ensure 1-dimensional array
        top_n_scores = row_data[top_n_indices]
        top_n_headers = header[top_n_indices]
        avg_top_n = np.mean(top_n_scores)
        results.append({
            "Row": row_index,
            "Average_Top_{}".format(n): avg_top_n,
            "Indices_Top_{}".format(n): list(top_n_indices),
            "Headers_Top_{}".format(n): list(top_n_headers)
        })

    with open(output_file, 'w', newline='') as out:
        writer = csv.writer(out, delimiter='\t')

        # Write header
        writer.writerow(["Row", "Average_Top_{}".format(n), "Headers_Top_{}".format(n)])

        # Write data
        for result in results:
            row_data = [
                result["Row"],
                result["Average_Top_{}".format(n)],
                result["Headers_Top_{}".format(n)]  # Updated column name
            ]
            writer.writerow(map(str, row_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TSV file and calculate averages of top N scores.")
    parser.add_argument("tsv_file", help="Path to the TSV file")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument("--n", type=int, required=True, help="Number of top scores to consider")

    args = parser.parse_args()
    process_tsv(args.tsv_file, args.output_file, args.n)
