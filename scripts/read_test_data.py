import pandas as pd

def read_test_data():
    df = pd.read_excel('data/Rheinbay 2020/41586_2020_1965_MOESM4_ESM.xlsx', engine='openpyxl', sheet_name='Table 5 Non-coding point mut dr')
    df['file_name'] = df['ID'].apply(lambda x: x.split('::')[0]+'.bed')
    df['file_name'] = df['file_name'].apply(lambda x: x.replace('mirna_', 'mirna.') if 'mirna_pre' in x or 'mirna_mat' in x else x)
    noncoding_elements = list(df[df['Final filter judgement'] == True]['ID'].unique())
    file_df = pd.read_csv('data/PCAWG_mutations_to_elements.icgc.public.txt', sep='\t')
    
    rheinbay_noncoding_drivers = file_df[file_df['reg_id'].isin(noncoding_elements)][['mut_chr', 'mut_pos1', 'mut_pos2', 'mut_ref', 'mut_alt', 'region_type', 'reg_id']].drop_duplicates().reset_index(drop=True)
    rheinbay_noncoding_drivers.rename({'mut_chr': 'chr', 'mut_pos1': 'start', 'mut_pos2': 'end', 'mut_ref': 'ref', 'mut_alt': 'alt'}, axis=1, inplace=True)
    rheinbay_noncoding_drivers['data_source'] = 'Rheinbay et al 2020'
    
    dr_nod = pd.read_excel('data\Dr. Nod\gkac1251_supplemental_files\SupplementaryTable4.xlsx', engine='openpyxl')
    dr_nod = dr_nod[['chr', 'pos0', 'pos1', 'ref', 'alt', 'tissue', 'geneSymbol']]
    dr_nod.rename({'pos0': 'start', 'pos1': 'end', 'tissue': 'region_type', 'geneSymbol': 'reg_id'}, inplace=True, axis=1)
    dr_nod['data_source'] = 'Dr.Nod 2023'
    
    test_data = pd.concat([rheinbay_noncoding_drivers, dr_nod])
    test_data.reset_index(drop=True, inplace=True)
    test_data['driver'] = 1
    test_data.drop_duplicates(subset=['chr', 'start', 'end', 'ref', 'alt'], inplace=True)
    test_data['chr'] = test_data['chr'].apply(lambda x: x.replace('chr', ''))
    test_data = test_data[['chr', 'start', 'end', 'ref', 'alt', 'driver', 'data_source']]
    return test_data