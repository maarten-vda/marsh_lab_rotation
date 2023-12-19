import argparse
import pandas as pd
import matplotlib
matplotlib.use('agg')  # Set the backend to 'agg' for remote server use
import matplotlib.pyplot as plt

def main(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    data = pd.read_csv(input_file, header=None, names=['Column1', 'Column2', 'Column3'])

    # Combine the data for 'P' and 'B'
    P_data = data[data['Column3'] == 'P']
    B_data = data[data['Column3'] == 'B']

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create normalized histograms for Column2 for P and B on the same subplot
    create_normalized_histogram(P_data, 'Pathogenic', ax, 'blue', 0.5)
    create_normalized_histogram(B_data, 'Benign', ax, 'red', 0.5)

    # Set the title and labels for the subplot
    ax.set_title('Normalized Distribution of closest distance to a Pathogenic Variant')
    ax.set_xlabel('Closest Distance to a Pathogenic Variant')
    ax.set_ylabel('Normalized Frequency')
    ax.legend(['P', 'B'])

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()  # Close the plot to release resources

def create_normalized_histogram(data, label, ax, color, alpha):
    # Create a normalized histogram for Column2 on a specified subplot
    total_data_points = len(data)
    ax.hist(data['Column2'], bins=200, label=label, color=color, alpha=alpha, density=True, stacked=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split and plot data from a TSV file')
    parser.add_argument('input_file', help='Input TSV file')
    parser.add_argument('output_file', help='Output file to save the combined plot')

    args = parser.parse_args()
    main(args.input_file, args.output_file)
