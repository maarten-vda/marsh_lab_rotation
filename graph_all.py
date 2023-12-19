import matplotlib
matplotlib.use('Agg')
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def process_tsv1(tsv1_path):
    tsv1_df = pd.read_csv(tsv1_path, sep='\t', index_col=0)
    scores = tsv1_df.iloc[:, 0].to_dict()
    return scores

def process_tsv2(tsv2_path):
    tsv2_df = pd.read_csv(tsv2_path, sep='\t')
    tsv2_df['identifier'] = tsv2_df['uniprot_id'] + '_' + tsv2_df['variant']
    tsv2_df.set_index('identifier', inplace=True)
    return tsv2_df

def extract_score(row, variant):
    amino_acid = variant[-1]
    amino_acid_index = ord(amino_acid) - ord('A')  # Assuming amino acids are represented by letters
    return row.iloc[amino_acid_index]

def plot_roc_curve(ax, y_true, y_scores, label):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    except ValueError:
        print(f"Skipping plot due to ValueError: {y_true}, {y_scores}")
        return
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label=f'ROC curve {label} (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc='lower right')

def plot_pr_curve(ax, y_true, y_scores, label):
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
    avg_precision = average_precision_score(y_true, y_scores)

    ax.plot(recall, precision, lw=2, label=f'Precision-Recall curve {label} (area = {avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')




def main():
    parser = argparse.ArgumentParser(description='Generate ROC curve from two input TSVs.')
    parser.add_argument('tsv1_path', help='Path to the first TSV file')
    parser.add_argument('tsv2_path', help='Path to the second TSV file')
    parser.add_argument('output_path', help='Path to save the ROC curve plot')
    parser.add_argument('--mode', choices=['roc', 'pr'], default='roc', help='Mode for curve generation (roc or pr)')

    args = parser.parse_args()

    scores = process_tsv1(args.tsv1_path)
    tsv2_df = process_tsv2(args.tsv2_path)

    distances_dict = {1: [], 0: []}
    alphamissense_dict = {1: [], 0: []}

    for identifier, score in scores.items():
        uniprot_id, variant = identifier.split('_')
        row = tsv2_df.loc[identifier]

        if isinstance(row, pd.DataFrame):
            # Extract the first row as a Series and use the header as the index
            row = row.iloc[0]
            row.index = tuple(row.index)

        if any(pd.isnull(row[col]) for col in row.keys()):
            print(f"Skipping row due to NaN values:\n{row}")
            continue

        substitution = row['variant'][-1]

        # Skip rows with stop codons
        if substitution == '*':
            print(f"Skipping row with stop codon: {row}")
            continue

        # Check if the column exists before accessing it
        if substitution not in row.index:
            print(f"Skipping row due to missing column: {substitution}")
            continue

        score_value = row[substitution]

        if score != 0 and not pd.isnull(score_value):
            if row['class'] == 'LP' or row['class'] == 'P':
                distances_dict[1].append(score)
            elif row['class'] == 'LB' or row['class'] == 'B':
                distances_dict[0].append(score)
            else:
                print(f"Skipping row due to invalid class: {row}")
        else:
            print(f"Skipping row due to invalid score: {row}")

    for _, row in tsv2_df.iterrows():
        substitution = row['variant'][-1]

        # Skip rows with stop codons
        if substitution == '*':
            print(f"Skipping row with stop codon: {row}")
            continue

        score = row[substitution]

        if score != 0 and not pd.isnull(score):
            if row['class'] == 'LP' or row['class'] == 'P':
                alphamissense_dict[1].append(score)
            elif row['class'] == 'LB' or row['class'] == 'B':
                alphamissense_dict[0].append(score)
            else:
                print(f"Skipping row due to invalid class: {row}")
        else:
            print(f"Skipping row due to invalid score: {row}")


    distances_labels = [1] * len(distances_dict[1]) + [0] * len(distances_dict[0])
    alphamissense_labels = [1] * len(alphamissense_dict[1]) + [0] * len(alphamissense_dict[0])
    distances_scores = np.nan_to_num(distances_dict[1] + distances_dict[0])
    alphamissense_scores = np.nan_to_num(alphamissense_dict[1] + alphamissense_dict[0])

    plot_roc_curve(np.array(distances_labels), np.array(distances_scores), args.output_path, label='Original')
    plot_roc_curve(np.array(alphamissense_labels), np.array(alphamissense_scores), args.output_path, label='Alphamissense')


    with plt.style.context('seaborn-whitegrid'):
        fig, ax = plt.subplots()

        if args.mode == 'roc':
            # Plot ROC curve for the 'Original' dataset
            plot_roc_curve(ax, np.array(distances_labels), np.array(distances_scores), label='Original')

            # Plot ROC curve for the 'Alphamissense' dataset
            plot_roc_curve(ax, np.array(alphamissense_labels), np.array(alphamissense_scores), label='Alphamissense')

            # Save the figure to the output path
            plt.savefig(args.output_path)
            plt.close()
        elif args.mode == 'pr':
            # Plot Precision-Recall curve for the 'Original' dataset
            plot_pr_curve(ax, np.array(distances_labels), np.array(distances_scores), label='Original')

            # Plot Precision-Recall curve for the 'Alphamissense' dataset
            plot_pr_curve(ax, np.array(alphamissense_labels), np.array(alphamissense_scores), label='Alphamissense')

            # Save the figure to the output path
            plt.savefig(args.output_path)
            plt.close()
        else:
            print(f"Invalid mode: {args.mode}. Please choose either 'roc' or 'pr'.")

if __name__ == "__main__":
    main()
