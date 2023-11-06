import pandas as pd
import numpy as np
from anytree import Node, LevelOrderGroupIter
from intervaltree import Interval, IntervalTree
import sys
import yaml

with open("../configuration.yaml", "r") as yml_file:
    config = yaml.load(yml_file, yaml.Loader)

sys.path.insert(1, config['SCRIPTS_FOLDER'])
from assembly_converter import convert_assembly_hg19_to_hg38

def read_and_convert_data():
    FILENAME = "test_data_final.csv"    # change this according to your dataset; make sure to include the file extension
    df = pd.read_csv('../data/' + FILENAME)
    print("Converting assembly from hg19 to hg38...")
    df = convert_assembly_hg19_to_hg38(df)
    df = df[['chr', 'start', 'end', 'start_hg19', 'driver']]
    return df

def read_COSMIC_data():
    cosmic = pd.read_csv('../data/COSMIC CGC data/COSMIC_driver_genes.csv')
    cosmic = cosmic[~cosmic['Genome Location'].str.contains(':-')]
    cosmic['chr'] = cosmic['Genome Location'].str.split(':').map(lambda x: x[0])
    cosmic['start'] = cosmic['Genome Location'].str.split(':').map(lambda x: x[1]).str.split('-').map(lambda x: int(x[0]))
    cosmic['end'] = cosmic['Genome Location'].str.split(':').map(lambda x: x[1]).str.split('-').map(lambda x: int(x[1]))
    return cosmic

def find_overlaps(df, cosmic):
    """This function divides the regions around the driver genes into categories
    and finds the overlap of the given mutations with these regions"""
    cgc_tree = IntervalTree()

    for index, row in cosmic.iterrows():
        cgc_tree.add(Interval(row['start'], row['end'], (row['chr'], row['Gene Symbol'], '0kb')))

        cgc_tree.add(Interval(row['start']-2000, row['start'], (row['chr'], row['Gene Symbol'], '2kb upstream')))
        cgc_tree.add(Interval(row['start']-10000, row['start']-2000, (row['chr'], row['Gene Symbol'], '10kb upstream')))
        cgc_tree.add(Interval(row['start']-100000, row['start']-10000, (row['chr'], row['Gene Symbol'], '100kb upstream')))

        cgc_tree.add(Interval(row['end'], row['end']+2000, (row['chr'], row['Gene Symbol'], '2kb downstream')))
        cgc_tree.add(Interval(row['end']+2000, row['end']+10000, (row['chr'], row['Gene Symbol'], '10kb downstream')))
        cgc_tree.add(Interval(row['end']+10000, row['end']+100000, (row['chr'], row['Gene Symbol'], '100kb downstream')))

    df_tree = []

    for index, row in df.iterrows():
        df_tree.append(Node(name = row['chr'] + ':' + str(row['start']) + '-' + str(row['end']), chr = row['chr'], start = row['start'], end = row['end']))

    df['interactions'] = 0
    df['interaction_gene'] = np.nan
    df['interaction_info'] = np.nan

    for index in range(len(df_tree)):
        child_list = [node for node in LevelOrderGroupIter(df_tree[index])][0]
        for node in child_list:
            found_list = list(set(list(cgc_tree.overlap(node.start, node.end)) + list(cgc_tree.at(node.start)) + list(cgc_tree.at(node.end))))  # use its coordinates to find interactions in the IntervalTree
            listindex = 0
            while listindex < len(found_list) and found_list[listindex].data[0] != node.chr:
                listindex += 1
            if listindex < len(found_list):
                children_left = True
                found_interaction = found_list[listindex].data
                Node(name = found_interaction[0] + ':' +  str(found_list[listindex].begin) + '-' +  str(found_list[listindex].end), chr = found_interaction[0], start = found_list[listindex].begin, end = found_list[listindex].end, checked = 0, parent = node)
                df.at[index, 'interactions'] = len(df_tree[index].descendants)
                df.at[index, 'interaction_gene'] = found_interaction[1]
                df.at[index, 'interaction_info'] = found_interaction[2]
    return df

def main():
    RESULT_FILENAME = "cosmic_overlaps_test"
    df = read_and_convert_data()
    cosmic = read_COSMIC_data()
    df = find_overlaps(df, cosmic)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_pickle('../data/COSMIC CGC data/' + RESULT_FILENAME + '.pickle')