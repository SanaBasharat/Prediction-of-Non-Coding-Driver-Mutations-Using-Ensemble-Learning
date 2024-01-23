import pandas as pd
import numpy as np
import os
import multiprocessing
import time
from intervaltree import Interval, IntervalTree
import sys
import yaml
import pickle

with open("../configuration.yaml", "r") as yml_file:
    config = yaml.load(yml_file, yaml.Loader)

sys.path.insert(1, config['SCRIPTS_FOLDER'])
from assembly_converter import convert_assembly_hg19_to_hg38

def read_dataset(filename):
    print("Reading dataset...")
    df = pd.read_csv('../data/' + filename)
    # print("Converting assembly from hg19 to hg38...")
    # df = convert_assembly_hg19_to_hg38(df)
    df = df[['chr', 'start', 'end', 'start_hg19', 'driver']]
    # df = df[:5]
    return df

def read_files(filename):
    # read file
    interactions = pd.read_csv(f"ChIA-PET data/{filename}.bedpe", delimiter="\t",\
            names = ["CHR1", "start1", "end1", "CHR2", "start2", "end2", "p2"],\
            usecols=["CHR1", "start1", "end1", "CHR2", "start2", "end2", "p2"], 
            header=None, dtype=\
            {0: object, 1: int, 2: int, 3: object, 4: int, 5: int, 6: int})
    return interactions


def filter_bedpe_file(interactions, chromosome):
    # filter based on chromosome
    interactions = interactions[ interactions["CHR1"] == chromosome ].copy()
    interactions.drop(columns=["CHR1", "CHR2"], inplace=True)
    interactions.reset_index(drop=True, inplace=True)
    indexStop = interactions.index.stop
    
    # add positive index to intervals on left
    interactions.insert(2, "p1", list(map( lambda x: set([x]),range(1, 1 + indexStop)) ))
    
    # add negative index to intervals on right
    interactions["p2"] = list(map( lambda x: set([x]), range(-1, -1 -1 * indexStop, -1) ))
    
    # separate them into different arrays
    int1 = interactions.iloc[:, :3].values
    int2 = interactions.iloc[:, 3:].values
    
    # merge two array into one array, 0th index will be (0,0,0) array
    # In each sub array, the rightmost value will be its index as well (including minus index)
    result_int = np.concatenate(( int1, int2[::-1] ))
    return result_int, indexStop


def create_tree(data):
    # create IntervalTree from array
    GenomicTree = IntervalTree( Interval( *j ) \
            for j in data)
    
    return GenomicTree


def merge_direct_overlaps(tree, data_reducer = None):
    if data_reducer == None:
        data_reducer = lambda *x: None
    
    sorted_tree = sorted(tree)
    merged = [sorted_tree[0]]
    lt = len(sorted_tree)
    highest = sorted_tree[0][0] - 1 # var for checking if interval merged before
    
    i = -1
    for inv in sorted_tree:
        i += 1
        j = i + 1
        while j < lt and inv.end > sorted_tree[j].begin :
            upper = max(inv.end, sorted_tree[j].end)
            data = data_reducer(inv.data, sorted_tree[j].data)
            merged.append(Interval(inv.begin, upper, data))
            if inv.end > highest:
                highest = inv.end
            j += 1
        if inv.begin > highest:
            merged.append(inv)
    new_tree = IntervalTree(merged[1:])
    
    return new_tree


def tree_to_array(tree, edge_length):
    array = np.full( 2*edge_length + 1, set(), dtype=set)
    
    for i in tree:
        for j in i.data:
            tmp = array[j] | i.data
            array[j] = tmp.difference([j])
    
    return array


def build_tree(interactions, GenomicTree_at_mutation):
    
    initials = set()
    for i in GenomicTree_at_mutation:           # this intervaltree needs to be saved as well
        initials.add(int(*i.data))
    # if root has no descendants, there is no overlap at all
    if len(initials) == 0:
        #print("No matching")
        return 0
    
    # assing initial values to length values
    used_indices = set( initials )
    indices = set( [ -1 * i for i in used_indices ] )       # finding interactions
    [ used_indices.add(i) for i in indices ]
    values = set()
    while True:
        values = set.union( *interactions[[ *indices ]] )   # going to found interactions
        indices = set( [-1 * i for i in values] )           # finding interactions again
        indices = indices.difference(used_indices)          # removing the ones already found   # HERE, I think if difference more than 1, then it's a loop
        if not indices:
            break
        used_indices = used_indices | indices               # union of initial nodes plus found nodes
        
    return len(used_indices)


def reducer(old, new):
    return old.union(new)


def worker(filename, metadata, mutation_df, saved_trees, saved_arrays):
    print("Entered file", filename)
    filename = filename.replace(".bedpe", "")
    biotype = metadata[metadata["File accession"] == filename]["Experiment target"].values[0].\
            replace("-human", "") + "_interactions"
    # read bedpe file as interactions
    interactions = read_files(filename)
    for chromosome in np.intersect1d(interactions["CHR1"], "chr" + mutation_df["chr"]):
        print(chromosome)
        if 'tree_'+chromosome+'_'+filename+'.pkl' not in saved_trees:
            print("Tree not found for " + chromosome + " for file " + filename)
            # filter interactions dataframe to interactions array
            filtered_interactions, indexStop = filter_bedpe_file(interactions, chromosome)
            
            # create IntervalTree
            tree = create_tree(filtered_interactions)
            # SAVE THIS TREE AS WELL
            with open('Saved/Trees/tree_'+chromosome+'_'+filename+'.pkl','wb') as f:
                pickle.dump(tree, f)
        else:
            print("Loaded tree for " + chromosome + " for file " + filename)
            with open('Saved/Trees/tree_'+chromosome+'_'+filename+'.pkl','rb') as f:
                tree = pickle.load(f)

        if 'array_'+chromosome+'_'+filename+'.pkl' not in saved_arrays:
            print("Array not found for " + chromosome + " for file " + filename)
            # merge intervals which overlaps directly (not same as merge_overlaps())
            m_tree = merge_direct_overlaps(tree.copy(), data_reducer=reducer)
            # convert tree to array
            array = tree_to_array(m_tree, indexStop)
            # THIS IS WHERE WE STORE THE TREE, but add chr to the name as well
            with open('Saved/Arrays/array_'+chromosome+'_'+filename+'.pkl','wb') as f:
                pickle.dump(array, f)
        else:
            print("Loaded array for " + chromosome + " for file " + filename)
            with open('Saved/Arrays/array_'+chromosome+'_'+filename+'.pkl','rb') as f:
                array = pickle.load(f)

        for mutation in mutation_df[ mutation_df["chr"] == chromosome[3:] ]["start"]:
            interaction_number = build_tree(array, tree.at(mutation))
            if interaction_number > 0:
                #print(interaction_number)
                mutation_df.loc[ (mutation_df["chr"] == chromosome[3:]) & \
                        (mutation_df["start"] == mutation), biotype ] = interaction_number
        mutation_df.to_csv(f"Results/processed_{os.path.basename(filename)}.csv", index=False)

   
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=os.cpu_count())
    
    FILENAME = 'all_data_hg38.csv'#"test_data_final.csv"    # change this according to your dataset; make sure to include the file extension
    
    all_files = os.listdir('ChIA-PET data/')
    all_files.remove('files.txt')
    all_files.remove('metadata.tsv')
    done_files = os.listdir('Results')       # remove files which have already been processed and are present in Results folder
    done_files = [x.replace('processed_', '').replace('csv', 'bedpe') for x in done_files]
    all_files = [x for x in all_files if x not in done_files]

    metadata = pd.read_csv('ChIA-PET data/metadata.tsv', sep='\t')

    # Here we are discarding files that belong to treated samples
    files_to_keep = list(metadata[~(metadata['Biosample term name'].str.contains('positive')) & ~(metadata['Biosample term name'].str.contains('activated')) & ~(metadata['Biosample term name'].str.contains('T-cell'))]['File accession'])
    files_to_keep = [item + '.bedpe' for item in files_to_keep]
    all_files = [item for item in all_files if item in files_to_keep]
    
    # all_files = ['ENCFF110LPZ.bedpe']
    
    saved_trees = os.listdir('Saved/Trees/')
    saved_arrays = os.listdir('Saved/Arrays/')

    df = read_dataset(FILENAME)
    df.sort_values('chr', inplace=True)
    df.reset_index(drop = True, inplace = True)

    df['CTCF_interactions'] = 0
    df['POLR2A_interactions'] = 0

    jobs = []
    
    start_time = time.perf_counter()
    print("Files to work on:", len(all_files))
    print("THIS MAY TAKE SOME TIME, DEPENDING ON YOUR MACHINE.")
    for file in all_files:
        # worker(file, metadata, df)
        print("Starting file", file)
        jobs.append(pool.apply_async(worker, args=(file, metadata, df.copy(), saved_trees, saved_arrays,)))
        # print(result.get(timeout=1))
    pool.close()
    pool.join()
    finish_time = time.perf_counter()
    print("Finished!")
    print("Elapsed time during the whole program in seconds:", finish_time-start_time)
