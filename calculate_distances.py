import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock, minkowski, braycurtis, canberra, chebyshev
from scipy.stats import ks_2samp, kendalltau, spearmanr, pearsonr
from scipy.special import kl_div
import argparse
import pandas as pd
from collections import defaultdict

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def calculate_wasserstein_distance(p, q):
    """
    Calculate Wasserstein distance between two probability distributions (histograms).

    Parameters:
        p (list): Probability distribution 1 (list of values).
        q (list): Probability distribution 2 (list of values).

    Returns:
        float: Wasserstein distance between the two distributions.
    """
    # Ensure the input lists have the same length
    if len(p) != len(q):
        raise ValueError("Input lists must have the same length")

    # Sort the input lists
    p = sorted(p)
    q = sorted(q)

    # Calculate the Wasserstein distance
    wasserstein_dist = 0.0
    cumulative_p = 0.0
    cumulative_q = 0.0

    for pi, qi in zip(p, q):
        wasserstein_dist += abs(cumulative_p - cumulative_q)
        cumulative_p += pi
        cumulative_q += qi

    return wasserstein_dist

def calculate_pearson_correlation(p, q):
    return (1-abs(pearsonr(p, q)[0]))

def calculate_spearman_correlation(p, q):
    return (1-abs(spearmanr(p, q)[0]))

# Function to calculate Bhattacharyya Distance
def calculate_bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(np.multiply(p, q))))

# Function to calculate KL Divergence
def calculate_kl_divergence(p, q):
    p = np.array(p)  # Convert p to a NumPy array
    q = np.array(q)  # Convert q to a NumPy array

    p = np.abs(p)
    q = np.abs(q)
    
    # Normalize the vectors to make sure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    if len(p) != len(q):
        raise ValueError("p and q must have the same length")

    epsilon = 1e-10  # A very small positive value to avoid division by zero

    return np.sum(kl_div(p + epsilon, q + epsilon))



def calculate_jeffrey_divergence(p, q):
    p = np.array(p)  # Convert p to a NumPy array
    q = np.array(q)  # Convert q to a NumPy array

    p = np.abs(p)
    q = np.abs(q)
    
    # Normalize the vectors to make sure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    if len(p) != len(q):
        raise ValueError("p and q must have the same length")

    epsilon = 1e-10  # A very small positive value to avoid division by zero

    kl_div_p = kl_div(p + epsilon, q + epsilon)  # Add epsilon to both p and q
    kl_div_q = kl_div(q + epsilon, p + epsilon)  # Add epsilon to both q and p

    # Calculate the mean of the individual divergences
    jeffrey_divergence = 0.5 * (kl_div_p + kl_div_q).mean()

    return jeffrey_divergence



# Function to calculate Chi-Squared Distance with epsilon to avoid division by zero
def calculate_chi_squared_distance(p, q):
    p = np.array(p)
    q = np.array(q)
    offset = abs(min(min(p), min(q)))
    p += offset
    q += offset

    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape.")

    epsilon = 1e-10  # A very small positive value to avoid division by zero
    return 0.5 * np.sum(np.square(p - q) / (p + q + epsilon))



# Function to calculate Kolmogorov-Smirnov Statistic
def calculate_ks_statistic(p, q):
    return ks_2samp(p, q).statistic


# Function to calculate Kendall Tau Rank Correlation
def calculate_kendall_tau_correlation(p, q):
    return kendalltau(p, q).correlation

# Function to calculate Jensen-Shannon Divergence
def calculate_jensen_shannon_divergence(p, q):
    p = np.array(p)  # Convert p to a NumPy array
    q = np.array(q)  # Convert q to a NumPy array

    p = np.abs(p)
    q = np.abs(q)
    
    # Normalize the vectors to make sure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    if len(p) != len(q):
        raise ValueError("p and q must have the same length")

    epsilon = 1e-10  # A very small positive value to avoid division by zero

    m = np.array([0.5 * (pi + qi) for pi, qi in zip(p, q)])

    return 0.5 * (kl_div(p + epsilon, m + epsilon) + kl_div(q + epsilon, m + epsilon)).mean()


# Function to calculate Hellinger Distance for vectors with negative values
def calculate_hellinger_distance(p, q):
    p = np.abs(p)
    q = np.abs(q)
    
    # Normalize the vectors to make sure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sqrt(1 - np.sum(np.sqrt(np.multiply(p, q))))


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Calculate vector similarity between vectors in a TSV file.')
parser.add_argument('input_file', help='TSV with ESM-1v scores for each variant')
parser.add_argument('input_file2', help='Filtered variants TSV file')
parser.add_argument('output_file', help='Output CSV file to save vector similarities')
parser.add_argument('--type', choices=['euclidean', 'manhattan', 'cosine', 'braycurtis', 'canberra', 'chebyshev', 'wasserstein', 'bhattacharyya', 'kl_divergence', 'jeffrey_divergence', 'chi_squared', 'ks_statistic', 'kendall_tau', 'jensen_shannon', 'hellinger', 'spearman', 'pearson'], default='cosine', help='Specify the type of vector similarity metric (default: cosine)')
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
output_file = args.output_file
similarity_type = args.type


class_dict = defaultdict(list)
ref_aa_dict = defaultdict(list)

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
        class_dict[var_class].append(unique_id)
        ref_aa_dict[unique_id].append(ref_aa)

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

# Create the header row for the output TSV file
header_row = [''] + variants

# Calculate vector similarity and write to the output TSV file
with open(output_file, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')

    # Write the header row
    writer.writerow(header_row)

    for i in range(len(variants)):
        row = [variants[i]]
        for j in range(len(variants)):
            if i == j or variants[j] not in class_dict['P'] or ref_aa_dict[variants[i]] != ref_aa_dict[variants[j]]:
                similarity = ''
            else:
                if similarity_type == 'euclidean':
                    similarity = euclidean(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'manhattan':
                    similarity = cityblock(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'minkowski':
                    similarity = minkowski(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'cosine':
                    similarity = calculate_cosine_similarity(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'braycurtis':
                    similarity = braycurtis(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'canberra':
                    similarity = canberra(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'chebyshev':
                    similarity = chebyshev(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'wasserstein':
                    similarity = calculate_wasserstein_distance(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'bhattacharyya':
                    similarity = calculate_bhattacharyya_distance(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'kl_divergence':
                    similarity = calculate_kl_divergence(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'jeffrey_divergence':
                    similarity = calculate_jeffrey_divergence(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'chi_squared':
                    similarity = calculate_chi_squared_distance(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'ks_statistic':
                    similarity = calculate_ks_statistic(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'kendall_tau':
                    similarity = calculate_kendall_tau_correlation(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'jensen_shannon':
                    similarity = calculate_jensen_shannon_divergence(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'hellinger':
                    similarity = calculate_hellinger_distance(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'spearman':
                    similarity = calculate_spearman_correlation(variant_vectors[variants[i]], variant_vectors[variants[j]])
                elif similarity_type == 'pearson':
                    similarity = calculate_pearson_correlation(variant_vectors[variants[i]], variant_vectors[variants[j]])
                else:
                    print(f'Unknown similarity type: {similarity_type}')
                    similarity = ''
            row.append(similarity)

        
        writer.writerow(row)

print(f'{similarity_type.capitalize()} similarities have been calculated and saved to {output_file}.')
