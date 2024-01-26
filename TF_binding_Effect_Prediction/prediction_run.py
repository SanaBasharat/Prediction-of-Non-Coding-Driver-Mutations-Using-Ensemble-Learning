import glob
import pandas as pd
import os
import pickle
import scipy.stats
import numpy as np
import time
from TF_binding_Effect_Prediction import biocoder
import warnings
warnings.filterwarnings("ignore")

""""
# Run this py file, after vep_preprocess.py (which prepares VEP file to appropriate format for prediction)
# You can directly obtain pre-computed files in /TF-binding Effect Prediction/TF_outputs in the github repo. 
# Double check input/output file locations according to your goal.
"""
def pred_vep(TF_param, ENCODE_ID,TF_name, vep_data):
    sequences = [biocoder.seqtoi(x) for x in vep_data['sequence']]
    altered_sequences = [biocoder.seqtoi(x) for x in vep_data['altered_seq']]

    ref = biocoder.nonr_olig_freq(sequences)
    mut = biocoder.nonr_olig_freq(altered_sequences)
    diff_count = (mut - ref).to_numpy()
    diff = np.dot(diff_count, TF_param[1])
    SE = np.sqrt(np.abs((np.dot(diff_count,  TF_param[-1]) * diff_count).sum(axis=1)))
    t = diff / SE
    p_val = scipy.stats.norm.sf(np.abs(t)) * 2
    pred_df = vep_data.assign(diff=diff, t=t, p_value=p_val)
    pred_df.to_csv(f"TF_binding_Effect_Prediction/TF_outputs/preds_eachTF/pred_{ENCODE_ID}_{TF_name}.csv", index=False)

# Aggregate all filtered TFs
def mean_update(current_mean, current_count, new_value):
    if new_value < 0:
        updated_sum = (-current_mean) * current_count + new_value
    else:
        updated_sum = current_mean * current_count + new_value
    updated_count = current_count + 1
    updated_mean = updated_sum / updated_count
    return abs(updated_mean)
def gain_or_loss(row,alpha):
    p_values = row["p_value"]
    diff = row['diff']
    ind = row["index"]
    if p_values <= alpha:
        if diff < 0: # if diff score is negative, then loss of TF exist
            vep_seq["TF_loss"][ind] += 1
            vep_seq["TF_loss_diff"][ind]=mean_update(vep_seq["TF_loss_diff"][ind],len(vep_seq["TF_loss_detail"][ind]), diff)
            vep_seq["TF_loss_detail"][ind].append([TF_name, diff, p_values])
        elif diff > 0: # if diff score is positive, then gain of TF exist
            vep_seq["TF_gain"][ind] += 1
            vep_seq["TF_gain_diff"][ind]=mean_update(vep_seq["TF_gain_diff"][ind],len(vep_seq["TF_gain_detail"][ind]), diff)
            vep_seq["TF_gain_detail"][ind].append([TF_name, diff, p_values])
def addcolumn_gain_loss(pred_vcfs,alpha): # Aggregate all TF-models
    global TF_name
    vep_seq["TF_loss"] = 0 # add columns
    vep_seq["TF_gain"] = 0
    vep_seq["TF_loss_diff"] = 0
    vep_seq["TF_gain_diff"] = 0
    vep_seq["TF_loss_detail"] = [ [] for _ in range(len(vep_seq))]
    vep_seq["TF_gain_detail"] = [ [] for _ in range(len(vep_seq))]
    for pred_vcf in pred_vcfs:
        TF_name = pred_vcf.split("_")[-1].split(".")[0]
        pred_sgd = pd.read_csv(pred_vcf)
        pred_sgd.reset_index(inplace=True)
        pred_sgd.apply(gain_or_loss,alpha=alpha,axis=1)
    vep_seq.to_csv(f"TF_binding_Effect_Prediction/TF_outputs/TFcombined_results/vep_loss_gain_data_{alpha}.csv",index=False)
    return vep_seq
def process_vep(param, ENCODE_ID, TF_name, vep_seq):
    start = time.perf_counter()
    print(f"\nAnalyzing VEP file | ENCODE_ID: {ENCODE_ID}, TF_name: {TF_name}")
    pred_vep(param, ENCODE_ID, TF_name, vep_seq)
    end = time.perf_counter()
    print(f"Analyzing VEP file finished in {round(end-start,3)} s | ENCODE_ID: {ENCODE_ID}, TF_name: {TF_name}")

def pred_run(vep_seq_file, p_val):
    global vep_seq
    if not os.path.exists("TF_binding_Effect_Prediction/TF_outputs/preds_eachTF"):  # the main output file
        os.makedirs("TF_binding_Effect_Prediction/TF_outputs/preds_eachTF")
    if not os.path.exists("TF_binding_Effect_Prediction/TF_outputs/TFcombined_results"):  # the main output file
        os.makedirs("TF_binding_Effect_Prediction/TF_outputs/TFcombined_results")

    # TF_outputs file is generated to store estimated paramaters and results
    params = glob.glob("TF_binding_Effect_Prediction/TF_outputs/params/*.pkl")
    # Listing pre-computed-pred files
    param_dict = {}  # store pre-computed parameters
    for param in params:  # 30 pre-computed parameters of models
        with open(param, "rb") as file:
            param_dict[param] = pickle.load(file)

    time_start = time.time()
    print("Prediction Starting!")
    # vep_seq = pd.read_csv(vep_seq_file)
    vep_seq = vep_seq_file.copy()
    for param_file, param_data in param_dict.items():
        ENCODE_ID = param_file.split("_")[-2]
        TF_name = param_file.split("_")[-1].split(".")[0]
        process_vep(param_data, ENCODE_ID, TF_name, vep_seq)

    time_end = time.time()
    print(f"\nPrediction Finished in {time_end-time_start}!")
    print("\nAnnotations are generating!")

    ## Instead of computing for all p-values, let's just compute for the one we want
    # p_vls = [0.05, 0.01, 0.001, 0.0001, 0.00001]
    # for x in p_vls:  # different thresholds of p-value
    #     pred_files = glob.glob(f"TF_outputs/preds_eachTF/pred_*.csv")
    #     addcolumn_gain_loss(pred_files, x)  # x : alpha value for statistical significance
    #     print(f"Predictions with annotations having {x} alpha threshold are completed!")

    pred_files = glob.glob(f"TF_binding_Effect_Prediction/TF_outputs/preds_eachTF/pred_*.csv")
    tf_output_return = addcolumn_gain_loss(pred_files, p_val)  # x : alpha value for statistical significance
    print(f"Predictions with annotations having {p_val} alpha threshold are completed!")
    return tf_output_return

