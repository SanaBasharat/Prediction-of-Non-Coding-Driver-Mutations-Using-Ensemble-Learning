import pandas as pd

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
    df = pd.concat([icgc, tcga]).drop(['Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15'], axis = 1, errors='ignore').reset_index(drop=True)
    df.drop_duplicates(subset=['chr', 'pos', 'ref', 'alt'], inplace = True)
    df.reset_index(drop=True, inplace=True)
    df = df[['chr', 'pos', 'ref', 'alt', 'data_source']]
    df.rename({'pos': 'start'}, axis = 1, inplace = True)
    df['start'] = df['start'].apply(lambda x: int(x))
    df['driver'] = 1
    return df