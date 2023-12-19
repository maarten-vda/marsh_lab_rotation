import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import math
import ast
from sklearn.preprocessing import MinMaxScaler


# Use Agg backend for non-interactive mode
plt.switch_backend('agg')

def plot_roc_curve(input_file, input_file2, output_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    esm_data = pd.read_csv(input_file2, delimiter='\t')

    # Extract the scores and true labels
    scores = data.iloc[:, 1].values
    true_labels = data.iloc[:, 2].apply(lambda x: 1 if x == 'P' else 0).values
    uniprot_id = data.iloc[:, 0].values
    truth_dict = dict(zip(uniprot_id, true_labels))

    esm_scores = esm_data.iloc[:, -20:].values
    esm_scores = np.nan_to_num(esm_scores)
    esm_scores_normalised = MinMaxScaler().fit_transform(-1 * esm_scores)
    esm_unique_id = esm_data.iloc[:, 0:2].values
    ids_list = []
    for i in range(len(esm_unique_id)):
        ids_list.append(str(esm_unique_id[i][0]) + '_' + str(esm_unique_id[i][1]))
    scores_dict = dict(zip(ids_list, esm_scores_normalised))
    scores_truth_dict = {}
    for i in ids_list:
        if i in truth_dict:
            scores_truth_dict[str(list(scores_dict[i]))] = truth_dict[i]
    

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)


    esm_roc_dict = {}
    for i in scores_truth_dict:
        try:
            for j in eval(i):
                if j != 0.0:           
                    esm_roc_dict[j] = scores_truth_dict[i]
        except Exception as e:
            print(f"Error processing {i}:", e)
            continue
        


    fpr_esm, tpr_esm, _ = roc_curve(np.array(list(esm_roc_dict.values())), np.array(list(esm_roc_dict.keys())))
    roc_auc_esm = auc(fpr_esm, tpr_esm)


    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Nearest Distance (n=1) ROC (AUC = {roc_auc:.2f})') 
    plt.plot(fpr_esm, tpr_esm, color='green', lw=2, label=f'ESM-1v ROC (AUC = {roc_auc_esm:.2f})')  # New line for ESM ROC curve
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
    parser.add_argument('input_file2', type=str, help='Input CSV file')
    parser.add_argument('output_file', type=str, help='Output PNG file')
    args = parser.parse_args()

    plot_roc_curve(args.input_file, args.input_file2, args.output_file)
