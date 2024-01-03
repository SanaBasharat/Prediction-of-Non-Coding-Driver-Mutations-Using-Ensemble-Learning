import pandas as pd
import numpy as np
import requests
import json
import time

def call_api(df):
    """This function calls the USCS Genome Browser API with the Repeat Masker track to retrieve annotations"""
    res = pd.DataFrame()
    df['temp_start'] = df['start']
    len_df = len(df)
    for index, row in df.iterrows():
        if row['temp_start'] >= row['end']:
            url = """https://api.genome.ucsc.edu/getData/track?genome=hg19;track=rmsk;chrom=chr{};start={};end={}""".format(row['chr'], row['start'], row['start'] + 1)
        else:
            url = """https://api.genome.ucsc.edu/getData/track?genome=hg19;track=rmsk;chrom=chr{};start={};end={}""".format(row['chr'], row['start'], row['end'])
        response = requests.get(url)
        resdf = pd.DataFrame(response.json())
        print("Working on index", index, '/', len_df, "...", len(resdf), "rows returned")
        res = pd.concat([res, resdf])
    return res

def post_processing(res, df):
    """Post processing of the data received from Repeat Masker"""
    df_merge = pd.DataFrame()
    if len(res) > 0:
        res = res[['start', 'end', 'chrom', 'rmsk', 'itemsReturned']]
        res['chrom'] = res['chrom'].apply(lambda x: x.replace('chr', ''))
        
        json_struct = json.loads(res.to_json(orient="records"))
        df_flat = pd.io.json.json_normalize(json_struct)
        df_flat.columns = df_flat.columns.str.replace("rmsk.", "")
        df_flat = df_flat.drop_duplicates(keep='first')
        df_flat.reset_index(inplace = True, drop = True)
        df_flat.drop(['id', 'end'], inplace=True, axis=1, errors='ignore')
        df_merge = pd.merge(df, df_flat, how='left', left_on=['chr', 'start'], right_on=['chrom', 'start'])
        df_merge = df_merge[~df_merge['chrom'].isna()].reset_index(drop=True,inplace=False)
        # df_merge['start'] = df_merge['start'].apply(lambda x: x+1)
        df_merge.drop_duplicates().reset_index(drop=True)
    return df_merge

def call_repeatmasker(df):
    start_time = time.perf_counter()
    print("Calling RepeatMasker...")
    rm_result = call_api(df)
    rpt_masker = post_processing(rm_result, df)
    rpt_masker = rpt_masker[['chrom', 'start', 'end', 'repClass', 'driver']] #driver
    rpt_masker = pd.concat([rpt_masker.drop('repClass', axis = 1), rpt_masker['repClass'].str.get_dummies().drop(['Low_complexity', 'Satellite', 'snRNA'], axis = 1, errors='ignore')], axis = 1)
    rpt_masker = rpt_masker.drop_duplicates(keep='first').reset_index(drop = True)
    df = df.merge(rpt_masker, how='left', left_on=['chr','start', 'end', 'driver'], right_on=['chrom','start', 'end', 'driver'])
    df.drop(['chrom', 'temp_start'], inplace=True, axis=1, errors='ignore')
    df.fillna(0, inplace = True)
    if 'DNA' in df.columns:
        df = df.astype({'DNA': 'int'})
    else:
        df['DNA'] = 0
    if 'LINE' in df.columns:
        df = df.astype({'LINE': 'int'})
    else:
        df['LINE'] = 0
    if 'LTR' in df.columns:
        df = df.astype({'LTR': 'int'})
    else:
        df['LTR'] = 0
    if 'SINE' in df.columns:
        df = df.astype({'SINE': 'int'})
    else:
        df['SINE'] = 0
    if 'Simple_repeat' in df.columns:
        df = df.astype({'Simple_repeat': 'int'})
    else:
        df['Simple_repeat'] = 0
    df = df.groupby(['chr', 'start', 'ref', 'alt']).max().reset_index()
    finish_time = time.perf_counter()
    print("Completed in ", finish_time-start_time,"seconds.")
    print("\n")
    return df