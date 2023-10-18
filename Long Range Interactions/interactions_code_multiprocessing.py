import pandas as pd
import numpy as np
import os
from anytree import Node, RenderTree, AsciiStyle, LevelGroupOrderIter, LevelOrderGroupIter, search
from intervaltree import Interval, IntervalTree
import multiprocessing
import time

def read_dataset():
    df_pos = pd.read_csv('df_grch38.bed', sep='\t', header=None)
    df_pos.columns = ['chr', 'start', 'end']

    orig = pd.read_csv('ICGC_TCGA_noncoding_data.csv')
    df_pos['pos_37'] = orig['pos']
    df_pos.drop_duplicates(keep='first', inplace=True)

    df_neg = pd.read_csv('negdf_grch38.bed', sep='\t', header=None)
    df_neg.columns = ['chr', 'start', 'end', 'old_pos', 'bed_format']
    df_neg['chr'] = df_neg['chr'].apply(lambda x: x.replace('chr', ''))
    df_neg['chr_old'] = df_neg['old_pos'].str.split(':').str[0].str.replace('chr', '')
    df_neg['start_old'] = df_neg['old_pos'].str.split(':').str[1].str.split('-').str[1]
    df_neg['end_old'] = df_neg['old_pos'].str.split(':').str[1].str.split('-').str[0]
    df_neg['start_old'] = pd.to_numeric(df_neg['start_old'])
    df_neg['end_old'] = pd.to_numeric(df_neg['end_old'])
    # metadata = pd.read_csv(r'D:\Sana\ChIA-PET\Files\metadata.tsv', sep='\t')

    df_pos = df_pos[['chr', 'start', 'end', 'pos_37']]
    df_pos['driver'] = 1
    df_pos.drop_duplicates(keep='first', inplace=True)

    df_neg = df_neg[['chr', 'start', 'end', 'start_old']]
    df_neg['driver'] = 0
    df_neg.rename(columns = {'start_old': 'pos_37'}, inplace = True)
    df_neg.drop_duplicates(keep='first', inplace=True)

    df = pd.concat([df_pos, df_neg])
    return df                              # final dataset including negative and positive set

def read_file(filename, chr):
    cp = pd.read_csv('D:/Sana/ChIA-PET/Files/' + filename, sep = '\t',  header = None)
    cp.columns = ['chr_A', 'start_A', 'end_A', 'chr_B', 'start_B', 'end_B', 'score']
    cp['chr_A'] = cp['chr_A'].map(lambda x: x.replace('chr', ''))
    cp['chr_B'] = cp['chr_B'].map(lambda x: x.replace('chr', ''))
    
    cp = cp[cp['chr_A'] == chr]
    return cp

def make_interval_tree(df):
    interactions_tree = IntervalTree()

    for index, row in df.iterrows():
        interactions_tree.add(Interval(row['start_A'], row['end_A'], tuple([row['chr_A'], row['start_B'], row['end_B']]))) # fotwards interaction
        interactions_tree.add(Interval(row['start_B'], row['end_B'], tuple([row['chr_A'], row['start_A'], row['end_A']]))) # backwards interaction
    
    return interactions_tree

def find_interactions(mutation, interactions_tree): # the mutation and interaction tree will be of the same chr
    df_tree = Node(name = 'artificial', checked = 0)
    
    found_overlap = False
    if len(interactions_tree.at(mutation)) > 0:
        found_overlap = True
        for i in interactions_tree.at(mutation):
            Node(Interval(i.data[1], i.data[2]), checked = 0, parent = df_tree)
            try:
                interactions_tree.remove(i)
            except:
                pass
            try:
                inv = Interval(i.data[1], i.data[2], tuple([i.data[0], i.begin, i.end]))
                interactions_tree.remove(inv)
            except:
                pass
       
    interactions = 0
    chains = 0
    
    if found_overlap:
        children_left = True
        while children_left is True:
            children_left = False
        
            for node in df_tree.leaves:
                if node.checked == 0:
                    node.checked = 1
                    found_list = list(interactions_tree.overlap(node.name))
                    for found_node in found_list:
                        children_left = True
                        Node(name = Interval(found_node.data[1], found_node.data[2]), checked = 0, parent = node)
                        try:
                            interactions_tree.remove(found_node)# only remove that node which satisfies both conditions of similarity
                        except:
                            pass
                            # ("Node was already removed")
                        try:    
                            interactions_tree.remove(Interval(found_node.data[1], found_node.data[2], tuple([found_node.data[0], found_node.begin, found_node.end])))
                        except:
                            pass
                            # print("Node was already removed")
        interactions = len(set([k.name for k in df_tree.descendants]))
        chains = sum(len(x) for x in [leaf for leaf in LevelGroupOrderIter(df_tree, filter_=lambda node: node.is_leaf and not node.is_root) if leaf])

    return interactions, chains
    
def fill_columns(metadata, file):
    if metadata[metadata['File accession'] == file.replace('.bedpe', '')]['Experiment target'].iloc[0].replace('-human', '') == 'CTCF':
        return 'CTCF_interactions', 'CTCF_chains'
    else:
        return 'POLR2A_interactions', 'POLR2A_chains'
    
def worker(filename, metadata, df):
    start_time = time.perf_counter()
    int_col, chains_col = fill_columns(metadata, filename)
    print("Entered file", filename)
    # old_chr = ''
    for index, row in df.iterrows():
        # print("working on ", filename, index, row['chr'], row['start'])
        cp = read_file(filename, row['chr'])
        # if row['chr'] != old_chr:
        tree = make_interval_tree(cp)
        interactions, chains = find_interactions(row['start'], tree)
        df.at[index, int_col] = row[int_col] + interactions
        df.at[index, chains_col] = row[chains_col] + chains
        # print("working on ", index, "with", interactions, chains)       
        # old_chr = row['chr']
    df.to_csv('D:/Sana/ChIA-PET/Results/processed_' + filename.replace('bedpe', 'csv'), index = False)
    finish_time = time.perf_counter()
    print("Elapsed time for" + filename + " in seconds:", finish_time-start_time)
    return df

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    done_files = os.listdir('D:\Sana\ChIA-PET\Results')
    done_files = [x.replace('processed_', '').replace('csv', 'bedpe') for x in done_files]
    all_files = os.listdir(r"D:\Sana\ChIA-PET\Files")
    all_files.remove('files.txt')
    all_files.remove('metadata.tsv')
    all_files = [x for x in all_files if x not in done_files]

    metadata = pd.read_csv(r'D:\Sana\ChIA-PET\Files\metadata.tsv', sep='\t')

    files_to_keep = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell'))]['File accession'])
    files_to_keep = [item + '.bedpe' for item in files_to_keep]
    all_files = [item for item in all_files if item in files_to_keep]
    print("FILES LEFT: ", len(all_files))
    
    df = read_dataset()
    df.sort_values('chr', inplace=True)
    df.reset_index(drop = True, inplace = True)

    df['CTCF_interactions'] = 0
    df['CTCF_chains'] = 0
    df['POLR2A_interactions'] = 0
    df['POLR2A_chains'] = 0

    jobs = []
    
    # start_time = time.perf_counter()
    for file in all_files:
    #     # worker(file, metadata, df)
        print("Starting file", file)
        jobs.append(pool.apply_async(worker, args=(file, metadata, df,)))
        # print(result.get(timeout=1))
    pool.close()
    pool.join()
    # finish_time = time.perf_counter()
    # print("Elapsed time during the whole program in seconds:", finish_time-start_time)