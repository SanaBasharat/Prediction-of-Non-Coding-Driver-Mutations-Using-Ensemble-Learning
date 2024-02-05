import pandas as pd
import requests

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