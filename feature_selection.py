import pandas as pd
import numpy as np
import warnings
import yaml
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import xgboost
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# from sklearn import preprocessing

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

data = pd.read_csv('data/final_dataset.csv')

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
data_test = pd.concat([data_test, data[data['data_source'] == 'COSMIC'].sample(n=len_test_data)]).reset_index(drop=True)   # get an equal amount of negative data
data = data.drop(data_test.index, inplace=False).reset_index(drop=True, inplace=False)

XGB_PARAMS = {                                            # CODE SOURCE: containers_build\boostdm\config.py
        "objective": "binary:logistic",
        "reg_lambda": 1,
        "random_state": 42,
        "scale_pos_weight": 1,
        "subsample": 0.7,        # fraction of observations to be random samples for each tree
        "reg_alpha": 0,          # L1 regularization term on weight
        "max_delta_step": 0,    # positive value can help make the update step more conservative. generally not used
        "min_child_weight": 1,
        "learning_rate": 1e-03,
        "colsample_bylevel": 1.0,
        "gamma": 0,     # specifies the minimum loss reduction required to make a split. Makes the algorithm conservative
        "colsample_bytree": 1.0,        # fraction of columns to be random samples for each tree
        "booster": "gbtree",
        "max_depth": 4, # Used to control over-fitting as higher depth will allow the model to learn relations very specific to a particular sample
        "silent": 1,
        "seed": 21,
        # "eval_metric": 'logloss',
        # "early_stopping_rounds": 2000
        # "reg_lambda": 1,  # explore this further

}

BIASED_COLUMNS = ['chr', 'ref_x', 'IG_C_gene', 'IG_D_gene', 'IG_J_gene', 'IG_J_pseudogene']

COLUMNS_TRAINING = [x for x in COLUMNS_TRAINING if x not in BIASED_COLUMNS]

COLUMNS_SHAP = [f'my_shap_{x}' for x in COLUMNS_TRAINING]
# COLUMNS_TRAINING = COLUMNS_TRAINING[:10]

for col in list(set(COLUMNS_TRAINING) - set(data.columns)):
    data[col] = 0

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

data[COLUMNS_TRAINING] = min_max_scaler.fit_transform(data[COLUMNS_TRAINING])

for col in list(set(COLUMNS_TRAINING) - set(data_test.columns)):
    data_test[col] = 0

data_test[COLUMNS_TRAINING] = min_max_scaler.fit_transform(data_test[COLUMNS_TRAINING])

x_train, x_test, y_train, y_test = train_test_split(data[COLUMNS_TRAINING], data['driver'],
                                                    random_state=104, 
                                                    test_size=0.25, 
                                                    shuffle=True)

params = XGB_PARAMS.copy()                                          
params['n_estimators'] = 20000  # set it high enough to allow "early stopping" events below
params['base_score'] = y_train.mean()
params['silent'] = True
# params['n_jobs'] = 1
params['seed'] = 104
model = XGBClassifier(**params)

# Build step forward feature selection
sfs1 = sfs(model,
           k_features=(10, len(COLUMNS_TRAINING)),
           forward=True,
           floating=False,
           verbose=1,
           scoring='accuracy',
           cv=5,
           n_jobs=4)

sfs1 = sfs1.fit(x_train, y_train)

feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

x_train.iloc[:, feat_cols]