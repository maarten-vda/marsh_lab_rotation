import argparse
import csv
import requests
import time

MAX_RETRIES = 3

def get_uniprot_info(accession_id):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            url = f"https://www.uniprot.org/uniprot/{accession_id}.json"
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception if the request was not successful

            data = response.json()

            name = data['proteinDescription']['recommendedName']['fullName']['value']
            cellular_component = []
            molecular_function = []
            biological_process = []
            for i in data['uniProtKBCrossReferences']:
                if i['database'] == 'GO':
                    annotation = i['properties'][0]['value']
                    if annotation.startswith('C:'):
                        cellular_component.append(annotation)
                    elif annotation.startswith('F:'):
                        molecular_function.append(annotation)
                    elif annotation.startswith('P:'):
                        biological_process.append(annotation)
            return name, cellular_component, molecular_function, biological_process

        except requests.exceptions.RequestException as e:
            # Handle HTTP request errors
            print(f"Error while fetching data for accession {accession_id}: {e}")
            retries += 1
            if retries < MAX_RETRIES:
                print(f"Retrying ({retries}/{MAX_RETRIES})...")
                time.sleep(2)  # Wait for a while before retrying
            else:
                print(f"Max retries reached for accession {accession_id}. Skipping.")
                return None, None, None

def main(input_csv, output_csv):
    with open(input_csv, 'r') as input_file, open(output_csv, 'w', newline='') as output_file:
        input_reader = csv.reader(input_file)
        output_writer = csv.writer(output_file)

        for row in input_reader:
            protein1_id, esm_1v, pathogenic, protein2_id = row
            protein1_uniprot_id = protein1_id.split('_')[0]
            protein2_uniprot_id = protein2_id.split('_')[0]

            protein1_name, protein1_c, protein1_f, protein1_p = get_uniprot_info(protein1_uniprot_id)
            protein2_name, protein2_c, protein2_f, protein2_p = get_uniprot_info(protein2_uniprot_id)
            common_cellular_component = set(protein1_c).intersection(protein2_c)
            common_molecular_function = set(protein1_f).intersection(protein2_f)
            common_biological_process = set(protein1_p).intersection(protein2_p)

            if protein1_name is not None and protein2_name is not None:
                output_row = [protein1_uniprot_id, protein1_name, pathogenic, protein1_c, protein1_f, protein1_p,
                              protein2_uniprot_id, protein2_name, protein2_c, protein2_f, protein2_p,
                              esm_1v, common_cellular_component, common_molecular_function, common_biological_process]
                output_writer.writerow(output_row)
                output_file.flush()  # Explicitly flush the output file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and save UniProt information")
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    args = parser.parse_args()

    main(args.input_csv, args.output_csv)
