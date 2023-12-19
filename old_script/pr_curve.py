import argparse
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive mode
plt.switch_backend('agg')

def plot_precision_recall_curve(input_file, output_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)

    # Extract the scores and true labels
    scores = data.iloc[:, 1].values
    true_labels = data.iloc[:, 2].apply(lambda x: 1 if x == 'P' else 0).values

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    # Save Precision-Recall curve as a PNG image
    plt.savefig(output_file, format='png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Precision-Recall curve from a CSV file and save as PNG.')
    parser.add_argument('input_file', type=str, help='Input CSV file')
    parser.add_argument('output_file', type=str, help='Output PNG file')
    args = parser.parse_args()

    plot_precision_recall_curve(args.input_file, args.output_file)
