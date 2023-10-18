import pandas as pd
import requests

def read_ICGC_TCGA_data():
    """
    This function reads data provided by PCAWG for driver mutations and filters it to retain only non-coding driver mutations.
    Data source: https://dcc.icgc.org/releases/PCAWG/driver_mutations
    """
    icgc = pd.read_csv('data/TableS3_panorama_driver_mutations_ICGC_samples.controlled.tsv', sep='\t')
    icgc = icgc[icgc['category'] == 'noncoding']    #filter for noncoding mutations
    icgc['data_source'] = 'ICGC'
    tcga = pd.read_csv('data/TableS3_panorama_driver_mutations_TCGA_samples.controlled.tsv', sep='\t')  # this file is not publicly available and hence is not provided in this repository
    tcga = tcga[tcga['category'] == 'noncoding']
    tcga['data_source'] = 'TCGA'
    df = pd.concat([icgc, tcga]).drop(['Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15'], axis = 1).reset_index(drop=True)
    df.drop_duplicates(subset=['chr', 'pos', 'ref', 'alt'], inplace = True)
    df.reset_index(drop=True, inplace=True)
    df = df[['chr', 'pos', 'ref', 'alt', 'data_source']]
    df.rename({'pos': 'start'}, axis = 1, inplace = True)
    df['start'] = df['start'].apply(lambda x: int(x))
    df['driver'] = 1
    return df

def read_COSMIC_data():
    """
    This function reads non-coding mutations data provided by COSMIC
    Data source: https://cancer.sanger.ac.uk/cosmic/download
    File name: Cosmic_NonCodingVariants_Vcf_v98_GRCh37.tar
    This file contains a huge amount of non-coding mutations.
    Using BCFTools in a WSL-2 environment, 599 mutations were randomly selected and saved as a VCF file called negative_samples.vcf.
    These mutations were fisrt confirmed to not overlap with our positive set, and then made to be used as our negative set.
    """
    df = pd.read_csv('data/negative_samples.vcf', sep='\t', header=None)
    df.columns=['chr', 'start', 'ref', 'alt']
    df['data_source'] = 'COSMIC'
    df = df[['chr', 'start', 'ref', 'alt', 'data_source']]
    df['start'] = df['start'].apply(lambda x: int(x))
    df['driver'] = 0
    return df

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

positive_set = read_ICGC_TCGA_data()
negative_set = read_COSMIC_data()

raw_data = pd.concat([positive_set, negative_set])
raw_data.reset_index(inplace=True, drop=True)

raw_data['id'] = 'mut' + raw_data.index.astype(str) # adding a unique identifier to each mutation

raw_data = calculate_end_coordinates(raw_data)

raw_data.loc[raw_data['data_source'] == 'TCGA', 'chr'] = '*'        # removing controlled access information
raw_data.loc[raw_data['data_source'] == 'TCGA', 'start'] = '*'
raw_data.loc[raw_data['data_source'] == 'TCGA', 'end'] = '*'

raw_data = raw_data[['id', 'chr', 'start', 'end', 'ref', 'alt', 'driver', 'data_source']] # re-ordering columns for better visibility
raw_data.to_csv('data/dataset.csv', index = False)