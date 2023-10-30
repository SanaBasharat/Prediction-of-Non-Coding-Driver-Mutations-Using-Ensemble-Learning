import pandas as pd
import numpy as np
from anytree import Node, RenderTree, AsciiStyle, LevelGroupOrderIter, LevelOrderGroupIter, PreOrderIter
from intervaltree import Interval, IntervalTree

def check_overlaps(df, df_to_check):
    for index, row in df.iterrows():
        if row['start'] >= row['end']:  # TODO change to ==
            df.at[index, 'end'] = row['start'] + 1
    
    for index, row in df_to_check.iterrows():
        if row['start'] >= row['end']:
            df_to_check.at[index, 'end'] = row['start'] + 1

    origdf = IntervalTree()

    for index, row in df_to_check.iterrows():
        origdf.add(Interval(row['start'], row['end'], (row['chr'])))

    df_tree = []

    for index, row in df.iterrows():
        df_tree.append(Node(name = row['chr'] + ':' + str(row['start']) + '-' + str(row['end']), chr = row['chr'], start = row['start'], end = row['end']))

    df['found'] = 0

    for index in range(len(df_tree)):
        child_list = [node for node in LevelOrderGroupIter(df_tree[index])][0]
        for node in child_list:
            found_list = list(set(list(origdf.overlap(node.start, node.end)) + list(origdf.at(node.start)) + list(origdf.at(node.end))))  # use its coordinates to find interactions in the IntervalTree
            listindex = 0
            while listindex < len(found_list) and found_list[listindex].data[0] != node.chr:
                listindex += 1
            # print(len(found_list), listindex, node.chr, node.start, node.end)
            # print(found_list)
            if listindex < len(found_list):
                children_left = True
                found_interaction = found_list[listindex].data
                Node(name = found_interaction[0] + ':' +  str(found_list[listindex].begin) + '-' +  str(found_list[listindex].end), chr = found_interaction[0], start = found_list[listindex].begin, end = found_list[listindex].end, checked = 0, parent = node)
                # print("Added child to:", index)
                df.at[index, 'found'] = len(df_tree[index].descendants)
    if len(df[df['found'] > 0]) > 0:
        print("Found overlaps")
        return df[df['found'] > 0]
    else:
        print("No overlaps found")
        return False