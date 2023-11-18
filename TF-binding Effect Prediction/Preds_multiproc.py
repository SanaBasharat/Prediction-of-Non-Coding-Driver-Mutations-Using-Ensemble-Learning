import glob
import pandas as pd
import os
import pickle
import scipy.stats
import numpy as np
import multiprocessing
import time

""""
Run this py file, after TF_model.py ( which computes coefficients and covariance matrix of TF features )

or you can directly obtain pre-computed files in /TF-binding Effect Prediction/TF_outputs in the github repo. 

Double check input/output file locations according to your goal.

This file contains multiprocessing operation. You can arrange of it as you desire if you change the number of processor.  
"""
# all permutations are already reverse-deleted for feature engineering part
# all sequences are represented in binary for model input
nucleotides = {'A':0,'C':1,'G':2,'T':3}
numtonuc = {0:'A',1:'C',2:'G',3:'T'}
complement = {0:3,3:0,1:2,2:1}

def window(fseq, window_size):
    for i in range(len(fseq) - window_size + 1):
        yield fseq[i:i+window_size]

# return the first or the last number representation
def seqpos(kmer,last):
    return 1 <<  (1 + 2 * kmer) if last else 1 << 2 * kmer;

def seq_permutation(seqlen):
    return (range(seqpos(seqlen,False),seqpos(seqlen,True)))

def gen_nonreversed_kmer(k):
    nonrevk = list()
    for i in range(seqpos(k,False),seqpos(k,True)):
        if i <= revcomp(i):
            nonrevk.append(i)
    return nonrevk

def itoseq(seqint):
    if type(seqint) is not int:
        return seqint
    seq = ""
    mask = 3
    copy = int(seqint) # prevent changing the original value
    while(copy) != 1:
        seq = numtonuc[copy&mask] + seq
        copy >>= 2
        if copy == 0:
            print("Could not find the append-left on the input sequence")
            return 0
    return seq

def seqtoi(seq,gappos=0,gapsize=0):
    # due to various seqlengths, this project always needs append 1 to the left
    binrep = 1
    gaps = range(gappos,gappos+gapsize)
    for i in range(0,len(seq)):
        if i in gaps:
            continue
        binrep <<= 2
        binrep |= nucleotides[seq[i]]
    return binrep

def revcomp(seqbin):
    rev = 1
    mask = 3
    copy = int(seqbin)

    while copy != 1:
        rev <<= 2
        rev |= complement[copy&mask]
        copy >>= 2
        if copy == 0:
            print("Could not find the append-left on the input sequence")
            return 0
    return rev

def revcompstr(seq):
    rev = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join([rev[base] for base in reversed(seq)])

def insert_pos(seqint,base,pos): # pos is position from the right
    return ((seqint << 2) & ~(2**(2*pos+2)-1)) | ((seqint & 2**(2*pos)-1) | (nucleotides[base] << pos*2))
    #return (seqint << 2) | (seqint & 2**pos-1) & ~(3 << (pos*2)) | (nucleotides[base] << pos*2)

# this function already counts without its reverse complement,
# Input: panda list and kmer length
# Output: oligonucleotide count with reverse removed
def nonr_olig_freq(seqtbl,kmer,nonrev_list):
    # separator, since this is binary, the number is counted from the right
    rightseparator = kmer
    leftseparator = rightseparator
    olig_df =  {k: [0] * len(seqtbl) for k in nonrev_list} # use dictionary first to avoid slow indexing from panda data frame
    for i in range(0,len(seqtbl)): #22s for 3000
        mask = (4**kmer)-1
        cpy = int(seqtbl[i])
        while cpy > (4**kmer)-1:
            cur = cpy & mask
            right = cur & ((4**rightseparator)-1)
            left = (cur >> 2*leftseparator) << 2*rightseparator
            seqint = left | right

            r = (1<<(2*kmer))|seqint # append 1
            rc = revcomp(r)
            if r > rc:
                r = rc
            # 392secs with loc,434 secs with the regression. R time, 10secs for allocation, 3.97mins for linreg
            # with 'at', only 23secs! -- 254secs total for 6mer
            olig_df[r][i] += 1
            cpy >>= 2
    return pd.DataFrame(olig_df)
def mypredict(seq1, seq2, params,cov_matrix,TF_name):
    ref = nonr_olig_freq([seqtoi(seq1)],6,nonrev_list)  # from N
    mut = nonr_olig_freq([seqtoi(seq2)],6,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    print(f"Active Process {os.getpid()} : {TF_name} diff score (WTBeta-MUTBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict
def pred_vcf(param,ENCODE_ID,TF_name,vcf_seq):
    stat_df = vcf_seq.apply(lambda x: mypredict(x["sequence"],x["altered_seq"],param[1],param[-1],TF_name),
                                                   axis=1,result_type="expand")
    pred_df = pd.concat([vcf_seq,stat_df],axis=1)
    pred_df.to_csv(f"TF_outputs/preds_eachTF/pred_{ENCODE_ID}_{TF_name}.csv", index=False)
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
            vcf_seq["TF_loss"][ind] += 1
            vcf_seq["TF_loss_diff"][ind]=mean_update(vcf_seq["TF_loss_diff"][ind],len(vcf_seq["TF_loss_detail"][ind]), diff)
            vcf_seq["TF_loss_detail"][ind].append([TF_name, diff, p_values])
        elif diff > 0: # if diff score is positive, then gain of TF exist
            vcf_seq["TF_gain"][ind] += 1
            vcf_seq["TF_gain_diff"][ind]=mean_update(vcf_seq["TF_gain_diff"][ind],len(vcf_seq["TF_gain_detail"][ind]), diff)
            vcf_seq["TF_gain_detail"][ind].append([TF_name, diff, p_values])
def addcolumn_gain_loss(pred_vcfs,alpha): # Aggregate all TF-models
    global vcf_seq, TF_name
    # vcf_seq = pd.read_csv(vcf_seq_path)
    vcf_seq["TF_loss"] = 0 # add columns
    vcf_seq["TF_gain"] = 0
    vcf_seq["TF_loss_diff"] = 0
    vcf_seq["TF_gain_diff"] = 0
    vcf_seq["TF_loss_detail"] = [ [] for _ in range(len(vcf_seq))]
    vcf_seq["TF_gain_detail"] = [ [] for _ in range(len(vcf_seq))]
    for pred_vcf in pred_vcfs:
        TF_name = pred_vcf.split("_")[-1].split(".")[0]
        pred_sgd = pd.read_csv(pred_vcf)
        pred_sgd.reset_index(inplace=True)
        pred_sgd.apply(gain_or_loss,alpha=alpha,axis=1)
    vcf_seq.to_csv(f"TF_outputs/vep_loss_gain_data_{alpha}.csv",index=False)
def process_vcf(param, ENCODE_ID, TF_name, vcf_seq):
    start = time.perf_counter()
    print(f"Process {os.getpid()} is analyzing VCF file, ENCODE_ID: {ENCODE_ID}, TF_name: {TF_name}")
    pred_vcf(param, ENCODE_ID, TF_name, vcf_seq)
    print(f"Process {os.getpid()} finished analyzing VCF file:, ENCODE_ID: {ENCODE_ID}, TF_name: {TF_name}")
    end = time.perf_counter()
    print(f"Elapsed Time -----------:{end-start} seconds")

# TF_outputs file is generated to store estimated paramaters and results
if not os.path.exists("TF_outputs"):
    os.makedirs("TF_outputs")
if not os.path.exists("TF_outputs/params"):
    os.makedirs("TF_outputs/params")
if not os.path.exists("TF_outputs/preds_eachTF"):
    os.makedirs("TF_outputs/preds_eachTF")

nonrev_list = gen_nonreversed_kmer(6)  # 2080 features (6-mer DNA)

# Listing pre-computed-pred files
params = glob.glob("TF_outputs/params/*.pkl")
vcf_filename = "TF_outputs/VEP_seq.csv"
param_dict = {}  # store pre-computed parameters
for param in params:  # 30 pre-computed parameters of models
    with open(param, "rb") as file:
        param_dict[param] = pickle.load(file)

if __name__ == "__main__":

    with multiprocessing.Pool() as pool:
        print("Multiprocessing Starting")
        vcf_seq = pd.read_csv(vcf_filename)

        for param_file, param_data in param_dict.items():
            ENCODE_ID = param_file.split("_")[-2]
            TF_name = param_file.split("_")[-1].split(".")[0]
            pool.apply_async(process_vcf, args=(param_data, ENCODE_ID, TF_name, vcf_seq,))
        # Close the pool and wait for all tasks to complete
        pool.close()
        pool.join()

        print("Multiprocessing Finished!/n")

    print("Annotations are generating!")
    p_vls = [0.05, 0.01, 0.001, 0.0001, 0.00001]
    for x in p_vls:  # different thresholds of p-value
        pred_files = glob.glob(f"TF_outputs/preds_eachTF/pred_*.csv")
        addcolumn_gain_loss(pred_files, x)  # x : alpha value for statistical significance
        print(f"Predictions with annotations having {x} alpha threshold are completed!")
