import pandas as pd
import numpy as np
import os
import sys
import yaml

with open("./configuration.yaml", "r") as yml_file:
    config = yaml.load(yml_file, yaml.Loader)

sys.path.insert(1, config['SCRIPTS_FOLDER'])
from assembly_converter import convert_assembly_hg19_to_hg38

def calculate_total_interactions(metadata):
    """This function calculates the total number of interactions present in ChIA-PET data for CTCF and POLR2A separately"""
    ctcf_files = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell')) & (metadata['Experiment target'].str.contains('CTCF'))]['File accession'])
    polr_files = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell')) & (metadata['Experiment target'].str.contains('POLR2A'))]['File accession'])

    sum_ctcf = 0
    for i in ctcf_files:
        f = pd.read_csv('Long_Range_Interactions/ChIA-PET data/' + i + '.bedpe', sep = '\t',  header = None)
        sum_ctcf = sum_ctcf + len(f)

    sum_polr2a = 0
    for i in polr_files:
        f = pd.read_csv('Long_Range_Interactions/ChIA-PET data/' + i + '.bedpe', sep = '\t',  header = None)
        sum_polr2a = sum_polr2a + len(f)
    return sum_ctcf, sum_polr2a

def add_interactions(metadata, all_files, sum_ctcf, sum_polr2a):
    """This function adds together all CTCF and POLR2A interactions and chains (separately)
    and then normalizes them using sum of total interactions"""
    files_to_keep = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell'))]['File accession'])

    df = pd.read_csv('Long_Range_Interactions/Results/processed_'+files_to_keep[0]+'.csv') # using an arbitrary file as starting dataframe
    df = df[['chr', 'start', 'end', 'start_hg19', 'driver']]
    df[['CTCF_interactions', 'POLR2A_interactions']] = 0    # making its columns 0

    for file in all_files:
        if file.replace('processed_', '').replace('.csv', '') in files_to_keep:
            f = pd.read_csv('Long_Range_Interactions/Results/' + file)
            df[['CTCF_interactions', 'POLR2A_interactions']] = df[['CTCF_interactions', 'POLR2A_interactions']].add(f[['CTCF_interactions', 'POLR2A_interactions']])

    df['CTCF_interactions'] = df['CTCF_interactions']/sum_ctcf
    # df['CTCF_chains'] = df['CTCF_chains']/sum_ctcf
    df['POLR2A_interactions'] = df['POLR2A_interactions']/sum_polr2a
    # df['POLR2A_chains'] = df['POLR2A_chains']/sum_polr2a
    return df

# def long_range_interactions_results():
#     all_files = os.listdir('Results')
#     metadata = pd.read_csv('ChIA-PET data/metadata.tsv', sep='\t')
#     df = pd.read_csv('../data/dataset_uncensored.csv')

#     sum_ctcf, sum_polr2a = calculate_total_interactions(metadata)
#     combined_df = add_interactions(metadata, all_files, sum_ctcf, sum_polr2a)

#     df = df.merge(combined_df[['chr', 'start_hg19', 'driver', 'CTCF_interactions', 'POLR2A_interactions']], left_on=['chr', 'start', 'driver'], right_on=['chr', 'start_hg19', 'driver'], how='right')
#     df.drop('start_hg19', inplace=True, axis=1)
#     return df

def calculate_interactions(df):
    all_files = os.listdir('Long_Range_Interactions/Results')
    metadata = pd.read_csv('Long_Range_Interactions/ChIA-PET data/metadata.tsv', sep='\t')
    print("Adding all interactions together...")
    sum_ctcf, sum_polr2a = calculate_total_interactions(metadata)

    print("Finding total interactions for each mutation...")
    combined_df = add_interactions(metadata, all_files, sum_ctcf, sum_polr2a)
    combined_df.drop_duplicates(inplace=True)
    combined_df.drop(['start', 'end'], inplace=True, axis = 1)
    combined_df.rename({"start_hg19": "start"}, inplace=True, axis=1)
    retdf = df.merge(combined_df, how='left', left_on=['chr', 'start', 'driver'], right_on=['chr', 'start', 'driver'])
    retdf = retdf.drop_duplicates(subset=['id']).reset_index(drop=True)
    print("Done!")
    return retdf