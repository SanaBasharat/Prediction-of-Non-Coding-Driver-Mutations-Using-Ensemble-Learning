import pandas as pd
import numpy as np
import pickle
import gzip
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier
import os

COLUMNS_TRAINING = ['ada_score', 'rf_score',
                    'ENSP', 'UNIPARC', 'GO', 'SpliceAI_pred_DP_AG', 'SpliceAI_pred_DP_AL',
                    'SpliceAI_pred_DP_DG', 'SpliceAI_pred_DP_DL', 'SpliceAI_pred_DS_AG',
                    'SpliceAI_pred_DS_AL', 'SpliceAI_pred_DS_DG', 'SpliceAI_pred_DS_DL',
                    '3_prime_UTR_variant', '5_prime_UTR_variant',
                    'NMD_transcript_variant', 
                    'downstream_gene_variant',
                    'intergenic_variant', 'intron_variant',
                    'non_coding_transcript_exon_variant', 'non_coding_transcript_variant',
                    'regulatory_region_variant',
                    'splice_donor_variant', 'splice_polypyrimidine_tract_variant',
                    'upstream_gene_variant', 'MODIFIER', 'MotifFeature',
                    'RegulatoryFeature', 'Transcript', 'CTCF_binding_site',
                    'enhancer', 'nonsense_mediated_decay',
                    'processed_pseudogene', 'processed_transcript', 'promoter',
                    'protein_coding', 'retained_intron',
                    'unprocessed_pseudogene',
                    'CTCF_interactions', 'POLR2A_interactions', 'CTCF_loops', 'POLR2A_loops',
                    'DNA', 'LINE', 'LTR', 'SINE', 'Simple_repeat',
                    'known_driver_gene', 'known_driver_gene_100kb_downstream', 'known_driver_gene_100kb_upstream', 'known_driver_gene_10kb_downstream',
                    'known_driver_gene_10kb_upstream', 'known_driver_gene_2kb_downstream', 'known_driver_gene_2kb_upstream',
                    'splice_acceptor_variant', 'splice_donor_region_variant', 'splice_donor_5th_base_variant',
                    'missense_variant', 'synonymous_variant', 'stop_gained', 'stop_lost', 'splice_region_variant', 'inframe_insertion', 'frameshift_variant',
                    'TF_loss', 'TF_gain', 'TF_loss_diff', 'TF_gain_diff',
                    'known_lncrna', 'known_lncrna_100kb_downstream', 'known_lncrna_100kb_upstream', 'known_lncrna_10kb_downstream',
                    'known_lncrna_10kb_upstream', 'known_lncrna_2kb_downstream', 'known_lncrna_2kb_upstream'
                  ]

def sort_filter(x_train, x_test, y_train, y_test):
    """
    Final preparation steps to achieve a competent CV data:
    1) remove repeated data items in the test datasets
       removing duplicate sites in test dataset --but not in training-- as repeated data in training provide us with
       weight of evidence, whereas too many repeated data at testing can spoil our capacity to evaluate
    2) random sampling to get a balanced test set
    3) set training feature labels in a canonical order taken from configuration
    """

    # reset index

    x_test.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # remove duplicates from test set

    x_test['chr'] = x_test['chr'].astype(str)
    x_test['start'] = x_test['start'].astype(int)
    x_test = x_test.drop_duplicates(['start', 'alt'])
    test_index = x_test.index
    y_test = y_test.loc[y_test.index.intersection(test_index)]

    # balance test set

    total_index = set(x_test.index)
    if y_test.mean() <= 0.5:
        balance_index = y_test[y_test == 1].index.tolist()
    else:
        balance_index = y_test[y_test == 0].index.tolist()
    remaining_index = list(set(total_index) - set(balance_index))
    balance_index += list(np.random.choice(remaining_index, size=len(balance_index), replace=False))
    x_test = x_test.loc[x_test.index.intersection(balance_index)]
    y_test = y_test.loc[y_test.index.intersection(balance_index)]

    # feature labels in standard order
    avoid = ['chr', 'start', 'ref', 'alt']
    features = list(filter(lambda x: x not in avoid, x_train))
    # print(set(features))
    # assert (set(features) == set(COLUMNS_TRAINING))
    x_train = x_train[avoid + COLUMNS_TRAINING]
    x_test = x_test[avoid + COLUMNS_TRAINING]

    return x_train, x_test, y_train, y_test

def split_balanced(x_data, y_data, test_size=0.3):
    """Generate balanced train-test split"""

    one_index  = list(y_data[y_data == 1].index)
    zero_index = list(y_data[y_data == 0].index)

    # randomly select n_ones indices from zero_index
    zero_index = list(np.random.choice(zero_index, size=len(one_index), replace=False))

    x_data_sub = x_data.loc[one_index + zero_index, :]
    y_data_sub = y_data.loc[one_index + zero_index]

    # the random state should be fixed prior to this call
    x_train, x_test, y_train, y_test = train_test_split(x_data_sub, y_data_sub, test_size=test_size)
    return x_train, x_test, y_train, y_test

def get_cv_sets_balanced(x_data, y_data, n, size):
    """Generate several balanced train-test sets"""

    for _ in range(n):
        x_train, x_test, y_train, y_test = split_balanced(x_data, y_data, test_size=size)
        yield x_train, x_test, y_train, y_test

def prepare(data, nsplits=10, test_size=0.2):

    data.rename(columns={'response': 'label'}, inplace=True)

    # keep 'pos' and 'chr' to run position-based filtering
    avoid = ['cohort', 'gene', 'aachange', 'label', 'motif']  # include 'ref' 'alt'
    features = list(filter(lambda x: x not in avoid, data.columns))

    # regression datasets
    x_data = data[features]
    y_data = data['driver'] # label

    # cv_list output has tuples (x_train, x_test, y_train, y_test) as elements
    cv_list = get_cv_sets_balanced(x_data, y_data, nsplits, test_size)

    # filter test data to prevent site repetitions
    cv_list = [sort_filter(*arg) for arg in cv_list]

    return cv_list

def load_mutations(input_path):
    data = pd.read_csv(input_path)
    data = data.sample(frac=1).reset_index(drop=True)

    for col in data.columns[data.isna().any()].tolist():
        data[col].fillna(0, inplace=True)

    data['TF_binding_site_agg'] = np.logical_or(data['TF_binding_site'], data['TF_binding_site_variant']).astype(int)

    data['TF_loss_add'] = data['TF_binding_site_agg'] + data['TF_loss']
    data['TF_gain_add'] = data['TF_binding_site_agg'] + data['TF_gain']
    data['TF_loss_diff_add'] = data['TF_binding_site_agg'] + data['TF_loss_diff']
    data['TF_gain_diff_add'] = data['TF_binding_site_agg'] + data['TF_gain_diff']

    data['SpliceAI_pred_DP_AG'] = abs(data['SpliceAI_pred_DP_AG'])
    data['SpliceAI_pred_DP_AL'] = abs(data['SpliceAI_pred_DP_AL'])
    data['SpliceAI_pred_DP_DG'] = abs(data['SpliceAI_pred_DP_DG'])
    data['SpliceAI_pred_DP_DL'] = abs(data['SpliceAI_pred_DP_DL'])


    data_test = data[(data['data_source'] == 'Rheinbay et al 2020') | (data['data_source'] == 'Dr.Nod 2023')]
    len_test_data = len(data_test)
    data_test = pd.concat([data_test, data[data['data_source'] == 'COSMIC'].sample(n=len_test_data)])   # get an equal amount of negative data
    data = data.drop(data_test.index, inplace=False).reset_index(drop=True, inplace=False)
    data_test.reset_index(drop=True, inplace=True)
    return data

def generate(mutations, random_state=None, bootstrap_splits=50, cv_fraction=0.3):

    # fix random seed (if provided)
    np.random.seed(random_state)

    d_output = {}

    # for gene, data in mutations.groupby('gene'):
    cv_list = prepare(mutations.copy(), nsplits=bootstrap_splits, test_size=cv_fraction) # data.copy()
    d_output = cv_list

    return d_output

def generate_splits(output_file):
    mutations = load_mutations('truba/home/sbasharat/ChIA-PET/final_dataset.csv')

    mutations.drop(['chr_y', 'start_y', 'end_y'], inplace=True, axis=1)
    mutations.rename({'chr_x': 'chr', 'start_x': 'start', 'end_x': 'end'}, inplace=True, axis=1)

    d_output = generate(mutations,
                        random_state=104,
                        bootstrap_splits=50,
                        cv_fraction=0.3)

    with gzip.open(output_file, 'wb') as f:
        pickle.dump(d_output, f)
        
def train(values):
    """Returns: optimal classification threshold and trained XGB model"""
    x_train, x_test, y_train, y_test, split_number, seed = tuple(values)
    print("Train: ", split_number)
    XGB_PARAMS = config['XGB_PARAMS']
    params = XGB_PARAMS.copy()
    params['n_estimators'] = 20000  # set it high enough to allow "early stopping" events below
    params['base_score'] = y_train.mean()
    # params['n_jobs'] = 1
    params['seed'] = seed
    myclassifier = XGBClassifier(**params)

    # train with xgboost
    learning_curve_dict = {}
    myclassifier.fit(x_train, y_train,
                       eval_set=[(x_train, y_train), (x_test, y_test)],
                       callbacks=[
                           xgboost.callback.EvaluationMonitor(rank=0, period=1, show_stdv=False),
                       ],
                       verbose=False)

    params['n_estimators'] = myclassifier.model.best_iteration
    learning_curve_dict = {k: v['logloss'][:params['n_estimators']] for k, v in learning_curve_dict.items()}
    myclassifier.model.set_params(**params)

    return myclassifier, split_number, x_test, y_test, learning_curve_dict

if __name__ == '__main__':   
    file_cv = 'truba/home/sbasharat/ChIA-PET/splits.pkl'
    generate_splits(file_cv)
    
    dict_results = {
        'models': [], 'split_number': [], 'x_test': [], 'y_test': [], 'learning_curves': []
    }
    
    min_rows = 30   # Minimum number of rows to carry out training
    non_features = ['start', 'chr', 'ref', 'alt']
    cores = os.cpu_count()
    
    with Pool(cores) as p:
        with gzip.open(file_cv, 'rb') as f:
            split_cv = pickle.load(f)

        mean_size = np.nanmean([cv[0].shape[0] for cv in split_cv])
        if mean_size < min_rows:
            print("ERROR")
            print(min_rows)
            print(mean_size)

        list_cvs = []
        for i, x in enumerate(split_cv):
            print("Enumerate: ", i)
            x_list = list(x) + [i, np.random.randint(100000)]

            # filter out non-features, i.e., columns not used for training
            x_list[0] = x_list[0].drop(non_features, axis=1)
            x_list[1] = x_list[1].drop(non_features, axis=1)
            list_cvs.append(x_list)

        for model, split_number, x_test, y_test, learning_curve in p.imap(
                train, list_cvs):
            print("Train: ", split_number)
            dict_results['models'].append(model)
            dict_results['split_number'].append(split_number)
            dict_results['x_test'].append(x_test)
            dict_results['y_test'].append(y_test)
            dict_results['learning_curves'].append(learning_curve)

        with gzip.open('truba/home/sbasharat/ChIA-PET/models.pkl', 'wb') as f:
            pickle.dump(dict_results, f)
