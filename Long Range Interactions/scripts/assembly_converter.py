import pandas as pd
import numpy as np
from pyliftover import LiftOver

def convert_assembly_hg19_to_hg38(df):
    """
    This function uses the LiftOver package to convert the coordinates of a pandas dataframe from hg19 to hg38,
    while preserving the hg19 start position as a separate column
    """
    df['start_hg19'] = df['start']

    lift_over = LiftOver('hg19', 'hg38')
    indices_to_drop = []
    for index, row in df.iterrows():
        try:
            df.at[index, 'start'] = lift_over.convert_coordinate('chr'+row['chr'], row['start'])[0][1]
            df.at[index, 'end'] = lift_over.convert_coordinate('chr'+row['chr'], row['end'])[0][1]
        except:
            print("Failed to convert index", index, 'chr'+row['chr'], row['start'])
            print("Removing index", index)
            indices_to_drop.append(index)
    
    df = df.drop(indices_to_drop)
    df.reset_index(inplace=True,drop=True)
    return df

if __name__ == '__main__':
    convert_assembly_hg19_to_hg38()