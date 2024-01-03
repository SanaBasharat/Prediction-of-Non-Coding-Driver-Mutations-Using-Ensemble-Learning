import pandas as pd
import requests
import io
import os
import numpy as np
import yaml

with open("configuration.yaml", "r") as yml_file:
    config = yaml.load(yml_file, yaml.Loader)

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

def read_COSMIC_data():
    """
    This function reads non-coding mutations data provided by COSMIC
    Data source: https://cancer.sanger.ac.uk/cosmic/download
    File name: Cosmic_NonCodingVariants_Vcf_v98_GRCh37.tar
    This file contains a huge amount of non-coding mutations.
    Using BCFTools in a WSL-2 environment, 599 mutations were randomly selected and saved as a VCF file called negative_samples.vcf.
    These mutations were fisrt confirmed to not overlap with our positive set, and then made to be used as our negative set.
    """
    df = pd.read_csv('data/nondriver_noncoding_mutations.vcf', sep='\t', header=None) #negative_samples.vcf
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

def repeat_masker(df, mode):
    filename = 'repeat_masker'
    if mode == 'test':
        filename = filename + '_test'
    rpt_masker = pd.read_pickle('data/RepeatMasker/' + filename + '.pickle')
    rpt_masker = rpt_masker[['chrom', 'start', 'end', 'repClass', 'driver']] #driver
    rpt_masker = pd.concat([rpt_masker.drop('repClass', axis = 1), rpt_masker['repClass'].str.get_dummies().drop(['Low_complexity', 'Satellite', 'snRNA'], axis = 1, errors='ignore')], axis = 1)
    rpt_masker = rpt_masker.drop_duplicates(keep='first').reset_index(drop = True)
    rpt_masker
    df = df.merge(rpt_masker, how='left', left_on=['chr','start', 'end', 'driver'], right_on=['chrom','start', 'end', 'driver'])
    df.drop(['chrom'], inplace=True, axis=1, errors='ignore')
    df.fillna(0, inplace = True)
    df = df.astype({'DNA': 'int', 'LINE': 'int', 'LTR': 'int', 'SINE': 'int', 'Simple_repeat': 'int'})
    df = df.groupby(['chr', 'start', 'ref', 'alt']).max().reset_index()
    return df

def COSMIC_CGC_interactions(df, mode):
    filename = 'cosmic_overlaps'
    if mode == 'test':
        filename = filename + '_test'
    gene_df = pd.read_pickle('data/COSMIC CGC data/' + filename + '.pickle')
    gene_df.drop_duplicates(subset=['chr', 'start_hg19'], inplace = True)
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('0kb', 'known_driver_gene')
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('100kb downstream', 'known_driver_gene_100kb_downstream')
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('100kb upstream', 'known_driver_gene_100kb_upstream')
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('10kb downstream', 'known_driver_gene_10kb_downstream')
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('10kb upstream', 'known_driver_gene_10kb_upstream')
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('2kb downstream', 'known_driver_gene_2kb_downstream')
    gene_df['interaction_info'] = gene_df['interaction_info'].replace('2kb upstream', 'known_driver_gene_2kb_upstream')
    df = df.merge(gene_df[['chr', 'start_hg19', 'interaction_gene', 'interaction_info']], how='left', left_on=['chr', 'start'], right_on=['chr', 'start_hg19'])
    df.drop(['start_hg19'], inplace=True, axis=1, errors='ignore')
    df = pd.concat([df.drop(['interaction_gene', 'interaction_info'], axis = 1, errors='ignore'), df['interaction_info'].str.get_dummies()], axis = 1)
    return df

def TF_binding_site_annotations(df, mode):
    filename = 'vep_loss_gain_data_0.001_3o'
    if mode == 'test':
        filename = filename + '_test'
    tf = pd.read_csv('data/' + filename + '.csv')
    tf['chr'] = tf['chr'].str.replace('chr', '')
    tf['start'] = tf['start'] + 1
    df = df.merge(tf[['chr', 'start', 'ref', 'alt', 'driver', 'TF_loss', 'TF_gain', 'TF_loss_diff', 'TF_gain_diff']], how='left',
                            left_on=['chr', 'start', 'ref', 'alt', 'driver'], right_on=['chr', 'start', 'ref', 'alt', 'driver'])
    fix_start = list(df[df['TF_loss'].isna()]['start'])

    for index, row in df.iterrows():
        if row['start'] in fix_start:
            if len(tf[(tf['start'] == row['start'] + 1) & (tf['chr'] == row['chr'])]) == 1:
                df.at[index, 'TF_loss'] = tf[(tf['start'] == row['start'] + 1) & (tf['chr'] == row['chr'])].iloc[0]['TF_loss']
                df.at[index, 'TF_gain'] = tf[(tf['start'] == row['start'] + 1) & (tf['chr'] == row['chr'])].iloc[0]['TF_gain']
                df.at[index, 'TF_loss_diff'] = tf[(tf['start'] == row['start'] + 1) & (tf['chr'] == row['chr'])].iloc[0]['TF_loss_diff']
                df.at[index, 'TF_gain_diff'] = tf[(tf['start'] == row['start'] + 1) & (tf['chr'] == row['chr'])].iloc[0]['TF_gain_diff']
            
    df['TF_loss'] = df['TF_loss'].fillna(0)
    df['TF_gain'] = df['TF_gain'].fillna(0)
    df['TF_loss_diff'] = df['TF_loss_diff'].fillna(0)
    df['TF_gain_diff'] = df['TF_gain_diff'].fillna(0)
    return df

def calculate_total_interactions(metadata):
    """This function calculates the total number of interactions present in ChIA-PET data for CTCF and POLR2A separately"""
    ctcf_files = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell')) & (metadata['Experiment target'].str.contains('CTCF'))]['File accession'])
    polr_files = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell')) & (metadata['Experiment target'].str.contains('POLR2A'))]['File accession'])

    sum_ctcf = 0
    for i in ctcf_files:
        f = pd.read_csv('Long Range Interactions/ChIA-PET data/' + i + '.bedpe', sep = '\t',  header = None)
        sum_ctcf = sum_ctcf + len(f)

    sum_polr2a = 0
    for i in polr_files:
        f = pd.read_csv('Long Range Interactions/ChIA-PET data/' + i + '.bedpe', sep = '\t',  header = None)
        sum_polr2a = sum_polr2a + len(f)
    return sum_ctcf, sum_polr2a

def add_interactions(metadata, all_files, sum_ctcf, sum_polr2a):
    """This function adds together all CTCF and POLR2A interactions and chains (separately)
    and then normalizes them using sum of total interactions"""
    files_to_keep = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell'))]['File accession'])

    df = pd.read_csv('Long Range Interactions/Results/processed_'+files_to_keep[0]+'.csv') # using an arbitrary file as starting dataframe
    df = df[['chr', 'start', 'end', 'pos_37', 'driver']]
    df[['CTCF_interactions', 'CTCF_chains', 'POLR2A_interactions', 'POLR2A_chains']] = 0    # making its columns 0

    for file in all_files:
        if file.replace('processed_', '').replace('.csv', '') in files_to_keep:
            f = pd.read_csv('Long Range Interactions/Results/' + file)
            df[['CTCF_interactions', 'CTCF_chains', 'POLR2A_interactions', 'POLR2A_chains']] = df[['CTCF_interactions', 'CTCF_chains', 'POLR2A_interactions', 'POLR2A_chains']].add(f[['CTCF_interactions', 'CTCF_chains', 'POLR2A_interactions', 'POLR2A_chains']])

    df['CTCF_interactions'] = df['CTCF_interactions']*100/sum_ctcf
    df['CTCF_chains'] = df['CTCF_chains']*100/sum_ctcf
    df['POLR2A_interactions'] = df['POLR2A_interactions']*100/sum_polr2a
    df['POLR2A_chains'] = df['POLR2A_chains']*100/sum_polr2a
    return df

def long_range_interactions_results(df, mode):
    all_files = os.listdir('Long Range Interactions/Results')
    metadata = pd.read_csv('Long Range Interactions/ChIA-PET data/metadata.tsv', sep='\t')
    # df = pd.read_csv('data/dataset_uncensored.csv')

    sum_ctcf, sum_polr2a = calculate_total_interactions(metadata)
    # combined_df = add_interactions(metadata, all_files, sum_ctcf, sum_polr2a)

    #___________________ #TODO: remove this
    filename = 'final_interactions_result'
    if mode == 'test':
        filename = filename + '_test'
    combined_df = pd.read_csv('data/' + filename + '.csv')
    combined_df['CTCF_interactions'] = combined_df['CTCF_interactions']*1000/sum_ctcf
    combined_df['CTCF_chains'] = combined_df['CTCF_chains']*1000/sum_ctcf
    combined_df['POLR2A_interactions'] = combined_df['POLR2A_interactions']*1000/sum_polr2a
    combined_df['POLR2A_chains'] = combined_df['POLR2A_chains']*1000/sum_polr2a
    #_____________________
    
    df = df.merge(combined_df[['chr', 'pos_37', 'driver', 'CTCF_interactions', 'CTCF_chains', 'POLR2A_interactions', 'POLR2A_chains']], left_on=['chr', 'start', 'driver'], right_on=['chr', 'pos_37', 'driver'], how='left')
    df.drop('pos_37', inplace=True, axis=1, errors='ignore')
    df.drop_duplicates(inplace=True)
    df['CTCF_interactions'] = df['CTCF_interactions'].fillna(0)
    df['CTCF_chains'] = df['CTCF_chains'].fillna(0)
    df['POLR2A_interactions'] = df['POLR2A_interactions'].fillna(0)
    df['POLR2A_chains'] = df['POLR2A_chains'].fillna(0)
    return df

def convert_to_vcf(df):
    # df['end'] = 0
    for index, row in df.iterrows():
        start_pos = row['start']
        ref = row['ref']
        alt = row['alt']
        if row['ref'] != '-' and row['alt'] != '-':
            # df.at[index, 'end'] = start_pos
            pass
        elif row['ref'] == '-':                 # insertion
            url = """https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={};start={};end={}""".format(row['chr'], start_pos - 1 , start_pos)
            # print(url)          # this API is 0 based, while my dataset is 1 based
            response = requests.get(url)
            seq = response.json()
            df.at[index, 'end'] = start_pos
            df.at[index, 'start'] = start_pos + 1
            df.at[index, 'ref'] = seq['dna']
            df.at[index, 'alt'] = seq['dna'] + alt
        elif row['alt'] == '-':                 # deletion
            url = """https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom={};start={};end={}""".format(row['chr'], start_pos - 1 - 1 , start_pos - 1)
            # print(url)          # this API is 0 based, while my dataset is 1 based
            response = requests.get(url)
            seq = response.json()
            # print(index, row['chr'], row['start'], row['ref'], row['alt'])
            # print(seq['dna'])
            df.at[index, 'start'] = start_pos - 1
            df.at[index, 'end'] = start_pos
            df.at[index, 'ref'] = seq['dna'].capitalize() + ref
            df.at[index, 'alt'] = seq['dna'].capitalize()
    return df

def create_vep_input(df, filename):
    if len(df[df['ref'] == '-']) > 0 or len(df[df['alt'] == '-']) > 0:
        print("File format detected that is other than VCF. Converting " + str(len(df[(df['ref'] == '-') | (df['alt'] == '-')])) + " records to VCF now...")
        print("This may take some time...")
        df_fix = df[(df['ref'] == '-') | (df['alt'] == '-')]
        ind_delete = df_fix.index
        df.drop(ind_delete, inplace=True)
        df_conv = convert_to_vcf(df_fix)
        df = pd.concat([df, df_conv])
        df.reset_index(drop=True, inplace=True)
    df['qual'] = '.'
    df['filter'] = '.'
    df['info'] = '.'
    df['format'] = '.'
    df = df[['chr', 'start', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format']]
    df.sort_values(by=['chr', 'start'], inplace=True)
    header = """##fileformat=VCFv4.1
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<>
##source=Test
##fileDate=20210728
##reference=hg19
##contig=<ID=chr1,length=249250621>
##contig=<ID=chr2,length=243199373>
##contig=<ID=chr3,length=198022430>
##contig=<ID=chr4,length=191154276>
##contig=<ID=chr5,length=180915260>
##contig=<ID=chr6,length=171115067>
##contig=<ID=chr7,length=159138663>
##contig=<ID=chr8,length=146364022>
##contig=<ID=chr9,length=141213431>
##contig=<ID=chr10,length=135534747>
##contig=<ID=chr11,length=135006516>
##contig=<ID=chr12,length=133851895>
##contig=<ID=chr13,length=115169878>
##contig=<ID=chr14,length=107349540>
##contig=<ID=chr15,length=102531392>
##contig=<ID=chr16,length=90354753>
##contig=<ID=chr17,length=81195210>
##contig=<ID=chr18,length=78077248>
##contig=<ID=chr19,length=59128983>
##contig=<ID=chr20,length=63025520>
##contig=<ID=chr21,length=48129895>
##contig=<ID=chr22,length=51304566>
##contig=<ID=chrX,length=155270560>
##INDIVIDUAL=<NAME=sample01, ID=sample010101>
#CHROM  POS ID  REF ALT QUAL    FILTER  INFO    FORMAT
"""
    with open(filename, 'w') as vcf:
        vcf.write(header)

    df.to_csv(filename, sep='\t', mode='a', index=False, header=False)
    return df

def read_vcf(path):
    """Reads a VCF format text file"""
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    df = pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})
    return df

def clean_and_preprocess(df):
    """This function pre-processes the VEP output, and flatten it to 1 row per mutation"""
    # drop columns with only one unique value
    cols_to_drop = []

    for col in df.columns:
        if len(df[col].unique()) == 1 and col not in config['COLUMNS_TRAINING']:
            cols_to_drop.append(col)

    df.drop(cols_to_drop, axis = 1, inplace = True, errors='ignore')

    df.replace('-', np.nan, inplace = True)

    # drop columns that are more than 50% null
    check = df.isnull().sum() / len(df) 
    cols = check[check > 0.5].index
    
    COLS_TO_DROP_INTUITION = ['Gene', 'Feature',
       'HGVSc', 'Existing_variation', 'DISTANCE', 'SYMBOL_SOURCE',
       'HGNC_ID', 'MaxEntScan_alt', 'MaxEntScan_diff', 'MaxEntScan_ref',
       'PHENOTYPES', 'AA', 'SOMATIC', 'PHENO', 'INTRON', 'SpliceAI_pred_SYMBOL', 'TSL']
    
    COLS_TO_KEEP = ['ada_score', 'rf_score', 'LOEUF', 'SpliceAI_pred_DP_AG', 'SpliceAI_pred_DP_AL',
       'SpliceAI_pred_DP_DG', 'SpliceAI_pred_DP_DL', 'SpliceAI_pred_DS_AG',
       'SpliceAI_pred_DS_AL', 'SpliceAI_pred_DS_DG', 'SpliceAI_pred_DS_DL',
        'CADD_PHRED', 'CADD_RAW', 'GO']

    COLS_TO_DROP = list(cols) + COLS_TO_DROP_INTUITION
    COLS_TO_DROP = [i for i in COLS_TO_DROP if i not in COLS_TO_KEEP]
    for col in COLS_TO_DROP:
        if col in df.columns and col not in config['COLUMNS_TRAINING']:
            df.drop(col, axis = 1, inplace = True, errors='ignore')

    df = df.fillna({'STRAND': 0,
                         'ada_score': 0, 'rf_score': 0,
                         'LOEUF': 0,
                        'SpliceAI_pred_DP_AG': 0, 'SpliceAI_pred_DP_AL': 0, 'SpliceAI_pred_DP_DG': 0,
                        'SpliceAI_pred_DP_DL': 0, 'SpliceAI_pred_DS_AG': 0, 'SpliceAI_pred_DS_AL': 0,
                        'SpliceAI_pred_DS_DG': 0, 'SpliceAI_pred_DS_DL': 0,
                        'CADD_PHRED': 0, 'CADD_RAW': 0})

    df = df.astype({'STRAND': 'int', 'ada_score': 'float', 'rf_score': 'float', 'LOEUF': 'float',
                        'SpliceAI_pred_DP_AG': 'float', 'SpliceAI_pred_DP_AL': 'float', 'SpliceAI_pred_DP_DG': 'float',
                        'SpliceAI_pred_DP_DL': 'float', 'SpliceAI_pred_DS_AG': 'float', 'SpliceAI_pred_DS_AL': 'float',
                        'SpliceAI_pred_DS_DG': 'float', 'SpliceAI_pred_DS_DL': 'float',
                        'CADD_PHRED': 'float', 'CADD_RAW': 'float'})
    
    df['Allele'] = df['Allele'].fillna('-')
    df['SYMBOL'] = df.groupby('#Uploaded_variation').SYMBOL.transform('first') # to fill in the null SYMBOLs for some variants
    
    df_dummies = pd.concat([df.drop(['Consequence', 'IMPACT', 'Feature_type', 'BIOTYPE'], axis = 1, inplace = False, errors='ignore'),
                    df['Consequence'].str.get_dummies(sep=","),
                    df['IMPACT'].str.get_dummies(),
                    df['Feature_type'].str.get_dummies(),
                    df['BIOTYPE'].str.get_dummies()], axis = 1)
    
    agg_dict = {c: 'max' for c in df_dummies.columns}
    agg_dict['ENSP'] = lambda x: ','.join(set(x.dropna()))
    agg_dict['UNIPARC'] = lambda x: ','.join(set(x.dropna()))
    agg_dict['GO'] = lambda x: ','.join(set(x.dropna()))
    
    df_grp = df_dummies.groupby(['#Uploaded_variation', 'Allele'], as_index=False).agg(agg_dict)
    df_grp.columns = df_grp.columns.get_level_values(0)
    
    df_grp['ENSP'] = [len(set(x.split(','))) for x in df_grp['ENSP']]
    df_grp['UNIPARC'] = [len(set(x.split(','))) for x in df_grp['UNIPARC']]
    df_grp['GO'] = [len(set(x.split(','))) for x in df_grp['GO']]

    df_grp['chr'] = df_grp['Location'].str.split(':').str[0]
    df_grp['start'] = df_grp['Location'].str.split(':').str[1].str.split('-').str[0]
    df_grp['end'] = df_grp['Location'].str.split(':').str[1].str.split('-').str[1]
    df_grp['start'] = pd.to_numeric(df_grp['start'])
    df_grp['end'] = pd.to_numeric(df_grp['end'])
    return df_grp

def make_dataset_uncensored():
    positive_set = read_ICGC_TCGA_data()
    negative_set = read_COSMIC_data()

    raw_data = pd.concat([positive_set, negative_set])
    raw_data.reset_index(inplace=True, drop=True)

    raw_data['id'] = 'mut' + raw_data.index.astype(str) # adding a unique identifier to each mutation

    raw_data = calculate_end_coordinates(raw_data)

    # raw_data.loc[raw_data['data_source'] == 'TCGA', 'chr'] = '*'        # removing controlled access information
    # raw_data.loc[raw_data['data_source'] == 'TCGA', 'start'] = '*'
    # raw_data.loc[raw_data['data_source'] == 'TCGA', 'end'] = '*'

    raw_data = raw_data[['id', 'chr', 'start', 'end', 'ref', 'alt', 'driver', 'data_source']] # re-ordering columns for better visibility
    raw_data.to_csv('data/dataset_uncensored.csv', index = False)

def make_dataset_censored():
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