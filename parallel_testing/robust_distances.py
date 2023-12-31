import argparse
import csv
import os
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil

def calculate_cosine_similarity(matrix1, matrix2):
    return cosine_similarity(matrix1, matrix2)

def read_data_in_chunks(file_path, chunk_size):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip header row
        data = []
        ids = []

        for row in reader:
            try:
                data.append(np.array(row[:20], dtype=float))
                ids.append(row[21] + '_' + row[22])
            except ValueError:
                print(f"Skipping row in {file_path}: {row}")

            if len(data) == chunk_size:
                yield data, ids
                data = []
                ids = []

        if data:
            yield data, ids

def write_similarity_to_file(similarity_matrix, output_file, header_row, header_column):
    with open(output_file, 'a') as out_file:
        # Write the header row
        header_row_str = '\t'.join(header_row)
        out_file.write(f"\t{header_row_str}\n")

        for i, row in enumerate(similarity_matrix):
            # Write the header column followed by the data row
            row_str = f"{header_column[i]}\t" + '\t'.join(format(val, '.6f') for val in row)
            out_file.write(row_str + '\n')



def main():
    parser = argparse.ArgumentParser(description="Calculate cosine similarity between two TSV files in chunks and write to an output file.")
    parser.add_argument("tsv_file1", help="Path to the first TSV file")
    parser.add_argument("directory_path", help="Path to the directory containing TSV files")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of chunks for processing")

    args = parser.parse_args()

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    try:
        # Initialize a temporary directory to store intermediate results
        temp_dir = tempfile.mkdtemp()

        # Process data in chunks from tsv_file1
        for chunk_idx, (chunk_data1, chunk_ids1) in enumerate(read_data_in_chunks(args.tsv_file1, args.chunk_size)):
            # Iterate over amino acids
            for amino_acid in amino_acids:
                tsv_file2_name = f"alphamissense_non_B_ref_{amino_acid}.tsv"
                tsv_file2_path = os.path.join(args.directory_path, tsv_file2_name)

                # Load data from TSV file2 for each chunk of file1
                with open(tsv_file2_path, 'r') as file2:
                    reader2 = csv.reader(file2, delimiter='\t')
                    header_row2 = next(reader2)  # Store the header row in file2

                    # Extract data from columns 1-20
                    data2 = []
                    data2_ids = []
                    for row in reader2:
                        try:
                            data2.append(np.array(row[:20], dtype=float))
                            unique_id = row[21] + '_' + row[22]
                            data2_ids.append(unique_id)
                        except ValueError:
                            print(f"Skipping row in {tsv_file2_path}: {row}")

                    # Check if there is any data to process
                    if not chunk_data1 or not data2:
                        raise ValueError("No valid data found in one or both files.")

                    # Calculate cosine similarity for the current chunk and amino acid
                    chunk_similarity = calculate_cosine_similarity(chunk_data1, data2)

                    # Write the current chunk's similarity matrix to a temporary file
                    temp_file_path = os.path.join(temp_dir, f"temp_chunk_{chunk_idx}_acid_{amino_acid}.tsv")
                    write_similarity_to_file(chunk_similarity, temp_file_path, data2_ids, chunk_ids1)

                    print(f"Processed chunk {chunk_idx + 1} with {tsv_file2_path}")

        # Merge intermediate results from temporary files to the final output file
        for temp_file_name in sorted(os.listdir(temp_dir)):
            temp_file_path = os.path.join(temp_dir, temp_file_name)
            with open(temp_file_path, 'r') as temp_file:
                shutil.copyfileobj(temp_file, open(args.output_file, 'a'))

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup: Remove the temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
