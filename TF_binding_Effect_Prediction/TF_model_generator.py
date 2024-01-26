import numpy as np
import pyfastx
import scipy.stats
from sklearn.linear_model import SGDRegressor
import pickle
import glob
import pandas as pd
import os
import argparse
import biocoder

"""
This py file generates 30 TF-models and contains vep/vcf file operation.
You can skip this file if you want to use pre-computed parameters which already are available in TF_outputs file on the github repo. 

Run following py file with suitable input in terminal
py TF_model.py -input /data/test_data_final

"""

if __name__ == "__main__": # python TF_model.py $filein
    parser = argparse.ArgumentParser(description='Generate predicitons of TF-binding perturbations for 30 TF models.')
    parser.add_argument('input', type=str, nargs=1, help="Input for the script, format: <VCF/VEP-path>")
    args = parser.parse_args()
    input_file = args.input[0] # vep or vcf format file location

pd.options.mode.chained_assignment = None # ignoring the warning

def peak_offset(sequence, peak, offset): # +30 / -30 peak offsets
    peak = peak - 1 # peak value exclusive
    left_offset = peak - offset
    right_offset = len(sequence) - (peak + offset)
    if (left_offset) < 0:
        start = 0
        end = (peak + offset + abs(left_offset))
    elif (right_offset) < 0:
        start = (peak - (offset + abs(right_offset)))
        end = (peak + offset)
    else:
        start = (left_offset)
        end = peak + offset
    return sequence[start: end]
# minimum-maximum feature scaling
def minmax(score):  # Min-max normalization
    if score.max() == 0:
        return score
    else:
        diff_range =  score.max() - score.min()
        minmax_norm = (score - score.min()) / diff_range
    return minmax_norm
# logarithmic scale for target values (scores in the freq table)
def log2trans(score):  # binary logarithmic scaling
    log_scores = np.log2(score)
    return log_scores
def print_full(x): # print binary-based sequences
    y = pd.DataFrame(x)
    y.index = [itoseq(x) for x in y.index]
    y["revcomp"] = [revcompstr(x) for x in y.index]
    return y
def TF_bednarrow_to_fasta(bed_file,genome,ENCODE_ID): # fasta conversion
    ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz")
    with open(bed_file, "r") as bed_data, open(f"outputs/{ENCODE_ID}_fasta.txt", "w") as output_file:
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][start:end]
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
        print(f"{ENCODE_ID} is converted to fasta format!")
def bedtotrainset(filebed,filefasta,ENCODE_ID,TF): # TF-ChIPseq bednarrowpeak files to trainset format(scores and seqs)

    with open(filefasta) as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">", "").replace("(+)", "") for x, y in enumerate(fasta) if x % 2 == 0]
    seqs = [y.rstrip().upper().replace("N", "") for x, y in enumerate(fasta) if x % 2 != 0]

    bed_data = pd.read_csv(filebed, sep="\t",
                           header=None)  # bed narrowpeak file to extract scores
    bed_data.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", "signalValue", "pValue", "qValue",
                        "peak"]
    chipseq_train = pd.DataFrame(
        {"region": locs, "sequence": seqs, "score": bed_data["signalValue"], "peak": bed_data["peak"]})
    chipseq_train["peak_seq"] = chipseq_train.apply(lambda x: peak_offset(x["sequence"], x["peak"], 30), axis=1)
    pbm_format = pd.DataFrame({0: chipseq_train["score"], 1: chipseq_train["peak_seq"]})
    pbm_format.sort_values(by=0,ascending=False,inplace=True)
    pbm_format.to_csv(f"outputs/ChIPseq_{ENCODE_ID}_{TF}.txt", header=None, index=False, sep="\t")
    return pbm_format
def read_chip(pbm_format,norm_method,kmer=6): # reading PBM input and apply transformation to scores and binary seq
    nonrev_list = biocoder.gen_nonreversed_kmer(kmer) # 2080 features (6-mer DNA)
    pbm_format.columns = ['score','sequence'] #score,sequence ..
    score = norm_method(pbm_format['score']) # log transformation for fluorescent signals
    seqbin = [seqtoi(x) for x in pbm_format['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = biocoder.nonr_olig_freq(seqbin,kmer) # feature vs sekans içeren count table oluşturur
    return pd.concat([score,oligfreq],axis=1) # output is frequency table corresponding to trainset
def get_cov_params(model,X,y):
    # Calculate sigma^2: σ^2=(Y−Xβ^)T(Y−Xβ^)/(n−p)
    y_hat = model.predict(X) # (predicted values)
    residuals = y - y_hat # lm_pbm.resid (true values - predicted values)
    e = np.array(residuals).reshape(residuals.shape[0], 1) # (Y−Xβ^)
    e_T = np.transpose(e) # (Y−Xβ^)'
    df_res = len(X) - len(nonrev_list) # degree of freedom # n - p # lm_pbm.df_resid
    SSE = np.dot(e_T, e) # the sum of squared errors / the sum of squared residuals : scalar value
    residual_variance =  SSE / (df_res) # lm_pbm.mse_resid (s^2) - MSE estimates sigma^2
    # The covariance matrix of the parameter estimates :
    #  2080*34589 X 34589*2080 --> 2080 * 2080
    scaled_cov_matrix = np.linalg.inv(np.dot(X.T, X)) * residual_variance
    return pd.DataFrame(scaled_cov_matrix)

#SGD method for training model:
def apply_sgd(df,scaler,lss,regularizer,ENCODE_ID): # Model training part
    X = df.drop('score',axis=1).apply(scaler,axis=0) # values of features
    y = df["score"] # target values
    sgd = SGDRegressor(loss= lss,alpha=0.0001, max_iter=1000, tol=1e-3, penalty=regularizer, eta0=0.1, random_state=333)
    sgd.fit(X, y)
    cov = get_cov_params(sgd, X, y)
    params = sgd.coef_
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    return sgd,params, print_motif,cov # [1] = coefficients , [-1] = covariance
# Prediction the effect between ref and mut sequences :

# Two option:
#1: by using bed inputs, genome assembly info, ENCODE data and TF name
def train_ChIP(bed_path, genome, ENCODE_ID,TF): # MODEL OUTPUT / provides estimate of weights / covariance matrix
    TF_bednarrow_to_fasta(bed_path, genome, ENCODE_ID) # write fasta file in your working directory
    pbm_format = bedtotrainset(bed_path, f"outputs/{ENCODE_ID}_fasta.txt", ENCODE_ID,TF) # trainset
    df_chip = read_chip(pbm_format,log2trans) # frequency table
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None,ENCODE_ID) # run model
    with open(f"parameters_{ENCODE_ID}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file) #
    return sgd_chip_none # 4 variables

# One of example bednarrowPeak file from ENCODE database among 30 TFs :
# bed_narrowpeak = glob.glob("TFs/GATA1/ENCFF853VZF.bed")
#     genome = "hg38" # for ChIP-seq assembly
#     encode_ID = "ENCFF853VZF"
#     TF = "GATA1"
#     train_ChIP(bed_narrowpeak,genome,encode_ID,TF)

#2: by using TF_trainset pre-constructed (users can find from data folder)
def save_params(TF_trainset): # if user want to save pre-computed paramaters

    pbm_format = pd.read_csv(TF_trainset, sep="\t", header=None)
    ENCODE_ID = TF_trainset.split("_")[2]
    TF_name = TF_trainset.split("_")[3].split(".")[0]
    df_chip = read_chip(pbm_format,log2trans) # frequency table
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None, ENCODE_ID)  # run model
    with open(f"TF_outputs/params/params_{ENCODE_ID}_{TF_name}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file) # pre-computed coefficients/covariance matrix of features
    print(f"TF_outputs/{ENCODE_ID}_{TF_name} is trained and saved!")
    param_dict[f"{ENCODE_ID}_{TF_name}"] = sgd_chip_none

#-------------------------------------------------------------------------------------------------------------------

# TF_outputs file is generated to store estimated paramaters and results
if not os.path.exists("TF_binding_Effect_Prediction/TF_outputs"): # the main output file
    os.makedirs("TF_binding_Effect_Prediction/TF_outputs")
if not os.path.exists("TF_binding_Effect_Prediction/TF_outputs/params"): # for parameters of 6-mer model features
    os.makedirs("TF_binding_Effect_Prediction/TF_outputs/params")

# Combine all functions into one main workflow:
trainset_file = "data/TF_trainsets/*.txt"
def param_estimates(trainset_file):
    global nonrev_list,param_dict
    nonrev_list = biocoder.gen_nonreversed_kmer(6) # 2080 features (6-mer DNA)
    TF_trainsets = glob.glob(trainset_file) # TF trainset format
    param_dict = {} # store pre-computed parameters for 30 TFs
    for TF in TF_trainsets:
        save_params(TF) # run models using SGD and only saves coefficients + covariance matrix of features
    return param_dict

# save the coefficients of features and covariance of the features (optional)
param_dict = param_estimates(trainset_file)
