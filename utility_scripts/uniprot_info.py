import requests

accession_id = "Q86YT5"  # Replace with your UniProt Accession ID
url = f"https://www.uniprot.org/uniprot/{accession_id}.json"

response = requests.get(url)
data = response.json()

name = "Name:", data['proteinDescription']['recommendedName']['fullName']['value']
keywords = "Keywords:", data['keywords']


# You can further process the 'domains' data to identify domains at a specific residue.
