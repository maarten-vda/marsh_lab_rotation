import argparse
import csv
import os
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from scipy.sparse import save_npz, load_npz, csr_matrix, vstack
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def calculate_cosine_similarity(matrix1, matrix2):
    return cosine_similarity(matrix1, matrix2)

def sameref_cosine_similarity(matrix1, matrix2, matrix1_ids, matrix2_ids, amino_acid):
        
    ref_array1 = np.array(matrix1_ids)
    ref_array2 = np.array(matrix2_ids)
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)

    ref_aas_1 = np.array([s.split('_')[1][0] if '_' in s else '' for s in ref_array1])
    ref_aas_2 = np.array([s.split('_')[1][0] if '_' in s else '' for s in ref_array2])

    # Create a boolean mask based on the condition (e.g., character after underscore is 'A')
    mask1 = ref_aas_1[:, np.newaxis] == f"{amino_acid}"
    mask2 = ref_aas_2[:, np.newaxis] == f"{amino_acid}"

    # Apply the mask to get the filtered arrays and replace false entries with 0
    filtered_ref_array1 = np.where(mask1, array1, 0)
    filtered_ref_array2 = np.where(mask2, array2, 0)

    # Convert the filtered arrays to sparse matrices
    sparse_matrix1 = csr_matrix(filtered_ref_array1)
    sparse_matrix2 = csr_matrix(filtered_ref_array2)

    similarity_matrix = calculate_cosine_similarity(sparse_matrix1, sparse_matrix2)
    for i in range(len(matrix1_ids)):
        for j in range(len(matrix2_ids)):
            if matrix1_ids[i] == matrix2_ids[j]:
                similarity_matrix[i][j] = 0
    return similarity_matrix

def samesub_cosine_similarity(matrix1, matrix2, matrix1_ids, matrix2_ids, amino_acid, amino_acid2):
    ref_array1 = np.array(matrix1_ids)
    ref_array2 = np.array(matrix2_ids)
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)

    ref_aas_1 = np.array([s.split('_')[1][0] if '_' in s else '' for s in ref_array1])
    ref_aas_2 = np.array([s.split('_')[1][0] if '_' in s else '' for s in ref_array2])
    sub_aas_1 = np.array([s.split('_')[-1][-1] if '_' in s else '' for s in ref_array1])
    sub_aas_2 = np.array([s.split('_')[-1][-1] if '_' in s else '' for s in ref_array2])

    # Create a boolean mask based on the condition (e.g., character after underscore is 'A')
    mask1 = ref_aas_1[:, np.newaxis] == f"{amino_acid}"
    mask2 = ref_aas_2[:, np.newaxis] == f"{amino_acid}"
    mask3 = sub_aas_1[:, np.newaxis] == f"{amino_acid2}"
    mask4 = sub_aas_2[:, np.newaxis] == f"{amino_acid2}"

    final_mask_dataset1 = np.logical_and(mask1, mask3)
    final_mask_dataset2 = np.logical_and(mask2, mask4)


    # Apply the mask to get the filtered arrays and replace false entries with 0
    filtered_ref_array1 = np.where(final_mask_dataset1, array1, 0)
    filtered_ref_array2 = np.where(final_mask_dataset2, array2, 0)

    # Convert the filtered arrays to sparse matrices
    sparse_matrix1 = csr_matrix(filtered_ref_array1)
    sparse_matrix2 = csr_matrix(filtered_ref_array2)

    similarity_matrix = calculate_cosine_similarity(sparse_matrix1, sparse_matrix2)
    for i in range(len(matrix1_ids)):
        for j in range(len(matrix2_ids)):
            if matrix1_ids[i] == matrix2_ids[j]:
                similarity_matrix[i][j] = 0

    return similarity_matrix


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

def write_similarity_to_file(similarity_matrix, output_file, header_row, header_column, chunk_idx):
    with open(output_file, 'a') as out_file:
        # Write the header row only if it's the first chunk
        if chunk_idx == 0:
            header_row_str = '\t'.join([""] + header_row) + '\n'
            out_file.write(header_row_str)

        # Convert the similarity_matrix to a formatted string
        formatted_matrix = "\n".join(
            '\t'.join([f"{header_column[i]}"] + [f"{val:.6f}" for val in row])
            for i, row in enumerate(similarity_matrix)
        )

        # Write the entire matrix to the file
        out_file.write(formatted_matrix + '\n')

        # Flush the buffer to ensure immediate writing
        out_file.flush()


def write_similarity_to_file_sparse(similarity_matrix, output_file, header_row, header_column, chunk_idx):
    with open(output_file, 'a') as out_file:
        # Write the header row only if it's the first chunk
        if chunk_idx == 0:
            header_row_str = '\t'.join([""] + header_row) + '\n'
            out_file.write(header_row_str)

        # Convert the similarity_matrix to a formatted string
        formatted_matrix = "\n".join(
            '\t'.join([f"{header_column[i]}"] + [f"{val:.6f}" for val in row])
            for i, row in enumerate(similarity_matrix)
        )

        # Convert the formatted matrix string to a dense matrix
        dense_matrix = np.array([list(map(float, row.split('\t')[1:])) for row in formatted_matrix.split('\n') if row])

        # Convert the dense matrix to a sparse matrix
        sparse_matrix = csr_matrix(dense_matrix)

        # Save the sparse matrix in NPZ format
        save_npz(output_file, sparse_matrix)

        # Flush the buffer to ensure immediate writing
        out_file.flush()


def merge_temp_files(temp_dir, output_file):
    temp_files = [os.path.join(temp_dir, file) for file in sorted(os.listdir(temp_dir))]
    with open(output_file, 'w') as out_file:
        for temp_file_path in temp_files:
            with open(temp_file_path, 'r') as temp_file:
                shutil.copyfileobj(temp_file, out_file)

    return output_file

def merge_temp_files_sparse(temp_dir, output_file):
    # Load and concatenate sparse matrices
    matrices = [load_npz(os.path.join(temp_dir, file)) for file in sorted(os.listdir(temp_dir))]
    concatenated_matrix = vstack(matrices)

    # Save the final sparse matrix in NPZ format
    save_npz(output_file, concatenated_matrix)


def process_chunk(chunk_idx, chunk_data1, chunk_ids1, args, temp_dir):
    global amino_acids

    concatenated_similarity = []
    data2_ids = []

    if args.type == "default" or args.type == "sameref":
        # Loop over amino acids
        for amino_acid in amino_acids:
            tsv_file2_name = f"alphamissense_non_B_ref_{amino_acid}.tsv"
            tsv_file2_path = os.path.join(args.directory_path, tsv_file2_name)

            amino_acid_ids = []

            # Load data from TSV file2 for each chunk of file1
            with open(tsv_file2_path, 'r') as file2:
                reader2 = csv.reader(file2, delimiter='\t')

                # Extract data from columns 1-20
                data2 = []
                for row in reader2:
                    try:
                        data2.append(np.array(row[:20], dtype=float))
                        unique_id = row[21] + '_' + row[22]
                        data2_ids.append(unique_id)
                        amino_acid_ids.append(unique_id)
                    except ValueError:
                        print(f"Skipping row in {tsv_file2_path}: {row}")

                # Check if there is any data to process
                if not chunk_data1 or not data2:
                    print(f"Warning: No valid data found in one or both files for chunk {chunk_idx + 1}. Skipping...")
                    continue

                if args.type == "default":
                    # Calculate cosine similarity for the current chunk and amino acid
                    chunk_similarity = calculate_cosine_similarity(chunk_data1, data2)
                elif args.type == "sameref":
                    # Calculate cosine similarity for the current chunk and amino acid
                    chunk_similarity = sameref_cosine_similarity(chunk_data1, data2, chunk_ids1, amino_acid_ids, amino_acid)

                # Concatenate horizontally to the temporary matrix
                concatenated_similarity.append(chunk_similarity)


    elif args.type == "samesub":
        # Nested Loop over amino acids
        for amino_acid in amino_acids:
            for amino_acid2 in amino_acids:
                tsv_file2_name = f"alphamissense_non_B_ref_{amino_acid}_sub_{amino_acid2}.tsv"
                tsv_file2_path = os.path.join(args.directory_path, tsv_file2_name)

                amino_acid_ids = []

                # Load data from TSV file2 for each chunk of file1
                with open(tsv_file2_path, 'r') as file2:
                    reader2 = csv.reader(file2, delimiter='\t')

                    # Extract data from columns 1-20
                    data2 = []
                    for row in reader2:
                        try:
                            data2.append(np.array(row[:20], dtype=float))
                            unique_id = row[21] + '_' + row[22]
                            data2_ids.append(unique_id)
                            amino_acid_ids.append(unique_id)
                        except ValueError:
                            print(f"Skipping row in {tsv_file2_path}: {row}")

                    # Check if there is any data to process
                    if not chunk_data1 or not data2:
                        print(f"Warning: No valid data found in one or both files for chunk {chunk_idx + 1} ref {amino_acid} sub {amino_acid2}. Skipping...")
                        continue

                    # Calculate cosine similarity for the current chunk and amino acid
                    chunk_similarity = samesub_cosine_similarity(chunk_data1, data2, chunk_ids1, amino_acid_ids, amino_acid, amino_acid2)

                    # Concatenate horizontally to the temporary matrix
                    concatenated_similarity.append(chunk_similarity)    

    concatenated_similarity = np.concatenate(concatenated_similarity, axis=1)
    # Concatenate horizontally to the temporary file
    output_file_path = os.path.join(temp_dir, f"temp_concatenated_similarity_chunk_{chunk_idx}.npz")

    write_similarity_to_file_sparse(concatenated_similarity, output_file_path, data2_ids, chunk_ids1, chunk_idx)
    print(f"Processed chunk {chunk_idx + 1}. Concatenated similarity matrix written to {output_file_path}")
    



def main():
    parser = argparse.ArgumentParser(description="Calculate cosine similarity between two TSV files in chunks and write to an output file.")
    parser.add_argument("tsv_file1", help="Path to the first TSV file")
    parser.add_argument("directory_path", help="Path to the directory containing TSV files")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of chunks for processing")
    parser.add_argument("--type", default="default", help="Which residues to calculate similarity for", choices=["default", "sameref", "samesub"])


    args = parser.parse_args()

    try:
        # Process data in chunks from tsv_file1
        temp_dir = tempfile.mkdtemp()
        

        # Determine the number of CPU cores
        num_cores = cpu_count()

        # Create a pool of processes with the number of available cores
        with Pool(num_cores) as pool:
            # Use Pool.map to parallelize the processing of chunks
            pool.starmap(process_chunk, [(idx, chunk_data1, chunk_ids1, args, temp_dir) for idx, (chunk_data1, chunk_ids1) in read_data_in_chunks(args.tsv_file1, args.chunk_size)])

        # Merge intermediate results from temporary files to the final output file
        final_output_file_path = args.output_file
        merge_temp_files_sparse(temp_dir, final_output_file_path)
        print(f"Final result written to {final_output_file_path}")

    #except Exception as e:
    #    print(f"An error occurred: {e}")

    finally:
        # Cleanup: Remove the temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
