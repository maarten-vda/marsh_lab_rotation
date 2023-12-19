#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt

def create_graph(input_file, output_file):
    x_values = []
    y_values = []

    with open(input_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                x_values.append(float(parts[0]))
                y_values.append(float(parts[1]))

    # Create and customize the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="Correct Predictions")
    plt.title("Threshold vs. Correct Predictions")
    plt.xlabel("Threshold")
    plt.ylabel("Correct Predictions")
    plt.grid(True)

    # Save the graph as an image file (e.g., PNG)
    plt.savefig(output_file)


    print(f"Graph saved as '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a graph from a TXT file")
    parser.add_argument("input_file", help="Input TXT file")
    parser.add_argument("output_file", help="Output image file")

    args = parser.parse_args()
    create_graph(args.input_file, args.output_file)
