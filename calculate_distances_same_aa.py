import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock, minkowski
import argparse
from collections import defaultdict

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Calculate vector similarity between vectors in a TSV file.')
parser.add_argument('input_file', help='Input TSV file containing vector data')
parser.add_argument('input_file2', help='Filtered variants TSV file')
parser.add_argument('output_dir', help='Output directory to save vector similarities for each reference amino acid')
parser.add_argument('--type', choices=['euclidean', 'manhattan', 'minkowski', 'cosine'], default='cosine',
                    help='Specify the type of vector similarity metric (default: cosine)')
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
output_dir = args.output_dir
similarity_type = args.type

class_dict = defaultdict(list)

with open(input_file2, 'r') as filtered_variants:
    reader = csv.reader(filtered_variants, delimiter='\t')

    for row in reader:
        if reader.line_num == 1:
            # Skip the first row (header)
            continue
        if row:
            uniprot_id = row[0]
            variant = row[5]
            ref_aa = variant[0]
            variant = ''.join(filter(lambda char: not char.isalpha() and char != '*', variant))
            var_class = row[6]
            if var_class == 'LP':
                var_class = 'P'
            unique_id = f'{uniprot_id}_{variant}'
            class_dict[(ref_aa, var_class)].append(unique_id)

variant_vectors = {}

with open(input_file, 'r') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        uniprot_id = row['uniprot_id']
        variant = row['variant']
        unique_id = f'{uniprot_id}_{variant}'
        scores = [float(row[aa]) if row[aa] else 0.0 for aa in 'ARNDCQEGHILKMFPSTWVY']

        # Check if all values are 0.0, if so, skip this row
        if not all(score == 0.0 for score in scores):
            variant_vectors[unique_id] = scores

# Get the list of variants
variants = list(variant_vectors.keys())

# Calculate vector similarity and write to separate TSV files for each reference amino acid
for ref_aa, var_class in class_dict:
    output_file = f"{output_dir}/{ref_aa}_{var_class}.tsv"
    header_row = [''] + class_dict[(ref_aa, var_class)]

    with open(output_file, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        # Write the header row
        writer.writerow(header_row)

        for i in range(len(class_dict[(ref_aa, var_class)])):
            row = [class_dict[(ref_aa, var_class)][i]]
            for j in range(len(class_dict[(ref_aa, var_class)])):
                if i == j:
                    similarity = ''
                else:
                    if similarity_type == 'euclidean':
                        similarity = euclidean(variant_vectors[class_dict[(ref_aa, var_class)][i]], variant_vectors[class_dict[(ref_aa, var_class)][j]])
                    elif similarity_type == 'manhattan':
                        similarity = cityblock(variant_vectors[class_dict[(ref_aa, var_class)][i]], variant_vectors[class_dict[(ref_aa, var_class)][j]])
                    elif similarity_type == 'minkowski':
                        similarity = minkowski(variant_vectors[class_dict[(ref_aa, var_class)][i]], variant_vectors[class_dict[(ref_aa, var_class)][j]])
                    elif similarity_type == 'cosine':
                        similarity = calculate_cosine_similarity(variant_vectors[class_dict[(ref_aa, var_class)][i]], variant_vectors[class_dict[(ref_aa, var_class)][j]])
                row.append(similarity)

            writer.writerow(row)

    print(f'{similarity_type.capitalize()} similarities for {ref_aa}_{var_class} have been calculated and saved to {output_file}.')
