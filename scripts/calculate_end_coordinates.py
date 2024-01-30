import pandas as pd
import requests

def calculate_end_coordinates(df):
    """
    This function calls the UCSC Genome Browser API to extract the end coordinate based on insertions or deletions
    """
    print("Calling UCSC Genome Browser API to extract end coordinates. This may take some time...")
    df['end'] = 0
    for index, row in df.iterrows():
        start_pos = row['start']
        ref = row['ref']
        alt = row['alt']
        if row['ref'] != '-' and row['alt'] != '-':
            df.at[index, 'end'] = start_pos
        elif row['ref'] == '-':                 # insertion
            url = """https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={};start={};end={}""".format(row['chr'], start_pos - 1 , start_pos)
            # print(url)          
            response = requests.get(url)        # this API is 0 based, while my dataset is 1 based
            seq = response.json()
            df.at[index, 'end'] = start_pos
            df.at[index, 'start'] = start_pos + 1
            df.at[index, 'ref'] = seq['dna']
            df.at[index, 'alt'] = seq['dna'] + alt
        elif row['alt'] == '-':                 # deletion
            url = """https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={};start={};end={}""".format(row['chr'], start_pos - 1 - 1 , start_pos - 1)
            # print(url)          
            response = requests.get(url)        # this API is 0 based, while my dataset is 1 based
            seq = response.json()
            df.at[index, 'start'] = start_pos - 1
            df.at[index, 'end'] = start_pos
            df.at[index, 'ref'] = seq['dna'] + ref
            df.at[index, 'alt'] = seq['dna']
    print("Extraction complete!")
    return df