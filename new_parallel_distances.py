import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock, minkowski, braycurtis, canberra, chebyshev
from scipy.stats import ks_2samp, kendalltau, spearmanr, pearsonr
from scipy.sparse import csr_matrix

amino_acid_columns = None
similarity_matrix = None
column_lookup = None
non_B_rows = None

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


def calculate_similarity(vector1, vector2, similarity_type):
    similarity_functions = {
        'euclidean': euclidean,
        'manhattan': cityblock,
        'minkowski': minkowski,
        'cosine': calculate_cosine_similarity,
        'braycurtis': braycurtis,
        'canberra': canberra,
        'chebyshev': chebyshev,
        'wasserstein': calculate_wasserstein_distance,
        'bhattacharyya': calculate_bhattacharyya_distance,
        'kl_divergence': calculate_kl_divergence,
        'jeffrey_divergence': calculate_jeffrey_divergence,
        'chi_squared': calculate_chi_squared_distance,
        'ks_statistic': calculate_ks_statistic,
        'kendall_tau': calculate_kendall_tau_correlation,
        'jensen_shannon': calculate_jensen_shannon_divergence,
        'hellinger': calculate_hellinger_distance,
        'spearman': calculate_spearman_correlation,
        'pearson': calculate_pearson_correlation,
    }
    similarity_function = similarity_functions.get(similarity_type)
    if similarity_function is not None:
        similarity = similarity_function(vector1, vector2)
    else:
        print(f'Unknown similarity type: {similarity_type}')
        similarity = ''
    return similarity

def filter_non_B_rows(df, mode, idx):
    class_column = [col for col in df.columns if col.endswith('class')][0]
    non_B_rows = df[df[class_column] != 'B']

    if mode == 'same_ref':
        ref_aa = df.iloc[idx]['variant'][0]
        non_B_rows_out = non_B_rows[non_B_rows['variant'].str[0] == ref_aa]
    elif mode == 'same_sub':
        ref_aa = df.iloc[idx]['variant'][0]
        sub_aa = df.iloc[idx]['variant'][-1]
        non_B_rows_out = non_B_rows[(non_B_rows['variant'].str[0] == ref_aa) & (non_B_rows['variant'].str[-1] == sub_aa)]
    else:
        non_B_rows_out = non_B_rows
    return non_B_rows_out


def process_chunk(chunk, amino_acid_columns, similarity_matrix, column_lookup, non_B_rows, mode, similarity_type):
    data_matrix = chunk[amino_acid_columns].values

    for idx in range(len(chunk)):
        if mode == 'same_ref' or mode == 'same_sub':
            non_B_rows_out = filter_non_B_rows(chunk, mode, idx)
            non_B_data_matrix = non_B_rows_out[amino_acid_columns].values
        else:
            non_B_data_matrix = non_B_rows[amino_acid_columns].values

        target_vector = data_matrix[idx, :]

        if len(non_B_data_matrix) > 0:
            for i, row in enumerate(non_B_data_matrix):
                if not (np.isclose(target_vector, row, atol=1e-5).all() or (np.all(target_vector == 0) or np.all(row == 0))):
                    if mode == 'same_ref' or mode == 'same_sub':
                        unique_id = non_B_rows_out.iloc[i]['uniprot_id'] + '_' + non_B_rows_out.iloc[i]['variant']
                    else:
                        unique_id = non_B_rows.iloc[i]['uniprot_id'] + '_' + non_B_rows.iloc[i]['variant']
                    try:
                        score = calculate_similarity(target_vector, row, similarity_type)
                    except ValueError as e:
                        print(f"Error calculating similarity for {unique_id}: {e}")
                        score = 0.0

                    column_idx = [k for k, v in column_lookup.items() if v == unique_id][0]
                    similarity_matrix[idx, column_idx] = score
                elif np.isclose(target_vector, row).all() == True:
                    similarity_matrix[idx, i] = 0.0
        else:
            print(f"Warning: Skipping row {idx} as non_B_data_matrix is empty.")

    # Return the modified values
    return amino_acid_columns, similarity_matrix, column_lookup, non_B_rows_out

def main(input_file, output_file, mode, similarity_type, chunksize):
    # Read input TSV into a DataFrame with chunksize
    df_chunk_iter = pd.read_csv(input_file, sep='\t', header=0, chunksize=chunksize)

    # Read the first chunk to get amino acid columns and initialize similarity matrix
    first_chunk = next(df_chunk_iter)
    amino_acid_columns = first_chunk.columns[:-3]

    # Initialize a matrix to store the similarities
    similarity_matrix = csr_matrix((len(first_chunk), 2 * chunksize))  # Adjust size if needed

    # Create a DataFrame for the output matrix
    output_df = pd.DataFrame(similarity_matrix.toarray(), index=first_chunk['uniprot_id'] + '_' + first_chunk['variant'])

    # Create a dictionary to map column indices to unique identifiers
    column_lookup = {i: first_chunk.iloc[i]['uniprot_id'] + '_' + first_chunk.iloc[i]['variant'] for i in range(len(first_chunk))}

    # Process the first chunk
    amino_acid_columns, similarity_matrix, column_lookup, non_B_rows = process_chunk(first_chunk, amino_acid_columns, similarity_matrix, column_lookup, non_B_rows, mode, similarity_type)

    # Process the remaining chunks
    for chunk in df_chunk_iter:
        amino_acid_columns, similarity_matrix, column_lookup, non_B_rows = process_chunk(chunk, amino_acid_columns, similarity_matrix, column_lookup, non_B_rows, mode, similarity_type)

    # Save the output to a file
    output_df.to_csv(output_file, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate similarity matrix.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file')
    parser.add_argument('output_file', type=str, help='Path to the output TSV file')
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'same_ref', 'same_sub'],
                        help='Mode for calculating similarity')
    parser.add_argument('--similarity_type', choices=['euclidean', 'manhattan', 'cosine', 'braycurtis', 'canberra', 'chebyshev', 'wasserstein', 'bhattacharyya', 'kl_divergence', 'jeffrey_divergence', 'chi_squared', 'ks_statistic', 'kendall_tau', 'jensen_shannon', 'hellinger', 'spearman', 'pearson'], default='cosine', help='Specify the type of vector similarity metric (default: cosine)')
    parser.add_argument('--chunksize', type=int, default=1000, help='Chunk size for processing the input dataframe')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.mode, args.similarity_type, args.chunksize)