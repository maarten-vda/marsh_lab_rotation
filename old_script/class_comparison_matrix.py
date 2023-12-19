import pandas as pd
import argparse
import multiprocessing

def compare_classes(df, start, end):
    scores = []
    for i in range(start, end):
        row1 = df.iloc[i]
        row_scores = []
        for j in range(len(df)):
            row2 = df.iloc[j]
            if row1['class'] == row2['class']:
                row_scores.append(1)
            else:
                row_scores.append(0)
        scores.append(row_scores)
    return scores

def main(input_file, output_file, num_processes):
    df = pd.read_csv(input_file, delimiter='\t')
    num_rows = len(df)
    chunk_size = num_rows // num_processes
    processes = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []

        for i in range(num_processes):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_processes - 1 else num_rows
            results.append(pool.apply_async(compare_classes, (df, start, end)))

        for result in results:
            scores = result.get()
            for row in scores:
                with open(output_file, 'a') as f:
                    f.write('\t'.join(map(str, row)) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 'class' values and create a square matrix of 1s and 0s")
    parser.add_argument("input_file", help="Input TSV file")
    parser.add_argument("output_file", help="Output TSV file")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use for parallelization")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.num_processes)
