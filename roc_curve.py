import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive mode
plt.switch_backend('agg')

def plot_roc_curve(input_file, output_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)

    # Extract the scores and true labels
    scores = data.iloc[:, 1].values
    true_labels = data.iloc[:, 2].apply(lambda x: 1 if x == 'P' else 0).values

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save ROC curve as a PNG image
    plt.savefig(output_file, format='png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ROC curve from a CSV file and save as PNG.')
    parser.add_argument('input_file', type=str, help='Input CSV file')
    parser.add_argument('output_file', type=str, help='Output PNG file')
    args = parser.parse_args()

    plot_roc_curve(args.input_file, args.output_file)
