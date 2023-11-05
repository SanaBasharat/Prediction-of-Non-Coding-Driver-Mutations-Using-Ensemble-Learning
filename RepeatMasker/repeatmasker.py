import pandas as pd
import numpy as np
import requests
import json

def call_repeatmasker(df):
    """This function calls the USCS Genome Browser API with the Repeat Masker track to retrieve annotations"""
    res = pd.DataFrame()
    for index, row in df.iterrows():
        print(index)
        if row['temp_start'] >= row['end']:
            url = """https://api.genome.ucsc.edu/getData/track?genome=hg19;track=rmsk;chrom=chr{};start={};end={}""".format(row['chr'], row['start'], row['start'] + 1)
        else:
            url = """https://api.genome.ucsc.edu/getData/track?genome=hg19;track=rmsk;chrom=chr{};start={};end={}""".format(row['chr'], row['start'], row['end'])
        response = requests.get(url)
        res = pd.concat([res, pd.DataFrame(response.json())])
    return res

def post_processing(res, df):
    """Post processing of the data received from Repeat Masker"""
    res = res[['start', 'end', 'chrom', 'rmsk', 'itemsReturned']]
    res['chrom'] = res['chrom'].apply(lambda x: x.replace('chr', ''))
    
    json_struct = json.loads(res.to_json(orient="records"))
    df_flat = pd.io.json.json_normalize(json_struct)
    df_flat.columns = df_flat.columns.str.replace("rmsk.", "")
    df_flat = df_flat.drop_duplicates(keep='first')
    df_flat.reset_index(inplace = True, drop = True)
    df_flat.drop(['id', 'end'], inplace=True, axis=1)
    df_merge = pd.merge(df, df_flat, how='left', left_on=['chr', 'start'], right_on=['chrom', 'start'])
    df_merge = df_merge[~df_merge['chrom'].isna()].reset_index(drop=True,inplace=False)
    # df_merge['start'] = df_merge['start'].apply(lambda x: x+1)
    df_merge.drop_duplicates().reset_index(drop=True)
    return df_merge

def main():
    # df = pd.read_csv('../data/dataset_uncensored.csv')    # TRAIN DATASET
    df = pd.read_csv('../data/test_data_final.csv')         # TEST DATASET
    rm_result = call_repeatmasker(df)
    final_df = post_processing(rm_result, df)
    # final_df.to_pickle('../data/RepeatMasker/repeat_masker.pickle', compression='infer', protocol=5, storage_options=None)
    final_df.to_pickle('../data/RepeatMasker/repeat_masker_test.pickle', compression='infer', protocol=5, storage_options=None)