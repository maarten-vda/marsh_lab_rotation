#!/usr/bin/env python
import argparse
import pandas as pd

def main(input_file, output_file, comparison_type):
    # Read the input TSV file into a DataFrame
    df = pd.read_csv(input_file, sep='\t')

    # Create a dictionary to store the mapping of unique identifiers to row indices
    unique_identifiers = {}
    substitutions = {}
    variant_effect = {}
    amino_acid_dict = {}


    for index, row in df.iterrows():
        unique_identifier = row['uniprot_id'] + '_' + ''.join(filter(str.isdigit, row['variant']))
        native_aa = row['variant'][0]
        unique_identifiers[unique_identifier] = index
        substitutions[unique_identifier] = native_aa
        variant_effect[unique_identifier] = row['class']
        for aa in "ARNDCQEGHILKMFPSTWVY":
            amino_acid_dict[aa] = []  # Initialize an empty list for each amino acid

        ##loop through substitutions dictionary and append unique_identifiers with same native_aa
        for unique_identifier, native_aa in substitutions.items():
            amino_acid_dict[native_aa].append(unique_identifier)

    # Create an empty square matrix DataFrame
    matrix = pd.DataFrame('', columns=unique_identifiers.keys(), index=unique_identifiers.keys())

    if comparison_type == "same_substitution":
        for native_aa1, id_list1 in amino_acid_dict.items():
            for native_aa2, id_list2 in amino_acid_dict.items():
                if native_aa1 == native_aa2:
                    for unique_id1 in id_list1:
                        for unique_id2 in id_list2:
                            class1 = variant_effect[unique_id1]
                            class2 = variant_effect[unique_id2]
                            if not (class1 == "B" and class2 == "B"):
                                if variant_effect[unique_id1] == variant_effect[unique_id2]:
                                    matrix.at[unique_id1, unique_id2] = 1
                                else:
                                    matrix.at[unique_id1, unique_id2] = 0
                            else:
                                continue

    if comparison_type == "only_pathogenic":
        for unique_id1, index1 in unique_identifiers.items():
            for unique_id2, index2 in unique_identifiers.items():
                class1 = df.loc[index1, 'class']
                class2 = df.loc[index2, 'class']
                if not (class1 == "B" and class2 == "B"):
                    if class1 == class2:
                        matrix.at[unique_id1, unique_id2] = 1
                    elif class1 != class2:
                        matrix.at[unique_id1, unique_id2] = 0

    if comparison_type == "full":
        # Fill the matrix with 1 or 0 based on the "class" column for rows with the same final character
        for unique_id1, index1 in unique_identifiers.items():
            for unique_id2, index2 in unique_identifiers.items():
                class1 = df.loc[index1, 'class']
                class2 = df.loc[index2, 'class']
                if class1 == class2:
                    matrix.at[unique_id1, unique_id2] = 1
                elif class1 != class2:
                    matrix.at[unique_id1, unique_id2] = 0

    # Output the square matrix to a TSV file
    matrix.to_csv(output_file, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a square matrix based on 'class' column")
    parser.add_argument("input_file", help="Input TSV file")
    parser.add_argument("output_file", help="Output TSV file for the square matrix")
    parser.add_argument("--type", choices=["full", "same_substitution", "only_pathogenic"], default="full", help="Comparison type")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.type)
