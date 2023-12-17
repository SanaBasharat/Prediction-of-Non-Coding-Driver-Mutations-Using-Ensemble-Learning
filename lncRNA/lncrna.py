import pandas as pd
import numpy as np
import requests
from anytree import Node, LevelOrderGroupIter
from intervaltree import Interval, IntervalTree
import yaml
import sys

with open("../configuration.yaml", "r") as yml_file:
    config = yaml.load(yml_file, yaml.Loader)

sys.path.insert(1, config['SCRIPTS_FOLDER'])
from assembly_converter import convert_assembly_hg19_to_hg38

def read_data():
    df = pd.read_csv('../data/dataset_uncensored.csv')
    tst = pd.read_csv('../data/test_data_final.csv')
    tst = tst[['id', 'chr', 'start', 'end', 'ref', 'alt', 'driver', 'data_source']]
    df = pd.concat([df, tst]).reset_index(drop=True)
    return df

def read_lncrna_data():
    lncdf = pd.read_csv("../data/lncRNA/lncipedia_5_2_hg19.bed", sep="\t", header = None)
    lncdf.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    lncdf.rename({'chrom': 'chr', 'chromStart': 'start', 'chromEnd': 'end'}, inplace=True, axis=1)
    lncdf['chr'] = lncdf['chr'].apply(lambda x: x.replace('chr', ''))
    return lncdf

def find_overlaps(df, lncdf):
    """This function divides the regions around the lncRNA into categories
    and finds the overlap of the given mutations with these regions"""
    lnc_tree = IntervalTree()

    for index, row in lncdf.iterrows():
        lnc_tree.add(Interval(row['start'], row['end'], (row['chr'], row['name'], '0kb')))

        lnc_tree.add(Interval(row['start']-2000, row['start'], (row['chr'], row['name'], '2kb upstream')))
        lnc_tree.add(Interval(row['start']-10000, row['start']-2000, (row['chr'], row['name'], '10kb upstream')))
        lnc_tree.add(Interval(row['start']-100000, row['start']-10000, (row['chr'], row['name'], '100kb upstream')))

        lnc_tree.add(Interval(row['end'], row['end']+2000, (row['chr'], row['name'], '2kb downstream')))
        lnc_tree.add(Interval(row['end']+2000, row['end']+10000, (row['chr'], row['name'], '10kb downstream')))
        lnc_tree.add(Interval(row['end']+10000, row['end']+100000, (row['chr'], row['name'], '100kb downstream')))

    df_tree = []

    for index, row in df.iterrows():
        df_tree.append(Node(name = row['chr'] + ':' + str(row['start']) + '-' + str(row['end']), chr = row['chr'], start = row['start'], end = row['end']))

    df['overlaps'] = 0
    df['overlap_lncrna'] = np.nan
    df['overlap_info'] = np.nan

    for index in range(len(df_tree)):
        child_list = [node for node in LevelOrderGroupIter(df_tree[index])][0]
        for node in child_list:
            found_list = list(set(list(lnc_tree.overlap(node.start, node.end)) + list(lnc_tree.at(node.start)) + list(lnc_tree.at(node.end))))  # use its coordinates to find interactions in the IntervalTree
            listindex = 0
            while listindex < len(found_list) and found_list[listindex].data[0] != node.chr:
                listindex += 1
            if listindex < len(found_list):
                children_left = True
                found_interaction = found_list[listindex].data
                Node(name = found_interaction[0] + ':' +  str(found_list[listindex].begin) + '-' +  str(found_list[listindex].end), chr = found_interaction[0], start = found_list[listindex].begin, end = found_list[listindex].end, checked = 0, parent = node)
                df.at[index, 'overlaps'] = len(df_tree[index].descendants)
                df.at[index, 'overlap_lncrna'] = found_interaction[1]
                df.at[index, 'overlap_info'] = found_interaction[2]
    return df

def main():
    RESULT_FILENAME = 'lncrna_overlaps'
    df = read_data()
    lncdf = read_lncrna_data()
    df = find_overlaps(df, lncdf)
    df.to_pickle('../data/lncRNA/' + RESULT_FILENAME + '.pickle')