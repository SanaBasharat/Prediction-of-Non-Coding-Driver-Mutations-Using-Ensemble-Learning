import numpy as np
import pyfastx
import scipy.stats
from sklearn.linear_model import SGDRegressor
import pickle
import glob
import pandas as pd
import os


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
    global nonrev_list
    nonrev_list = gen_nonreversed_kmer(kmer) # 2080 features (6-mer DNA)
    pbm_format.columns = ['score','sequence'] #score,sequence ..
    score = norm_method(pbm_format['score']) # log transformation for fluorescent signals
    seqbin = [seqtoi(x) for x in pbm_format['sequence']] #  PBM içindeki her bir sekansı binary gösterimine çevirir
    oligfreq = nonr_olig_freq(seqbin,kmer,nonrev_list) # feature vs sekans içeren count table oluşturur
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
def apply_sgd(df,scaler,lss,regularizer,ENCODE_ID):
    X = df.drop('score',axis=1).apply(scaler,axis=0) # values of features
    y = df["score"] # target values
    sgd = SGDRegressor(loss= lss,alpha=0.0001, max_iter=1000, tol=1e-3, penalty=regularizer, eta0=0.1, random_state=333)
    sgd.fit(X, y)
    cov = get_cov_params(sgd, X, y)
    params = sgd.coef_
    print_motif = pd.DataFrame({"Weights": sgd.coef_}, index=nonrev_list)  # for array-like output of OLS result
    print_motif = print_full(print_motif["Weights"].sort_values(ascending=False))
    # print_motif.to_csv(f"outputs/motifs_{ENCODE_ID}.csv")
    return sgd,params,print_motif,cov # [1] = coefficients [-1] = covariance
# Prediction the effect between ref and mut sequences :
def mypredict(seq1, seq2, params,cov_matrix):
    ref = nonr_olig_freq([seqtoi(seq1)],6,nonrev_list)  # from N
    mut = nonr_olig_freq([seqtoi(seq2)],6,nonrev_list)  # to N
    diff_count = mut - ref # c': count matrix
    diff = np.dot(diff_count, params) # c'Bhat --> the difference in binding affinity
    print("\ndiff score aka (wildBeta-mutatedBeta) (c'): ",diff)
    SE = np.sqrt((np.dot(diff_count,cov_matrix) * diff_count).sum(axis=1)) # 2080*2080 X 2080*1 = 2080*1 # a scalar value Standart error
    t =  diff/ SE # t-statistic : c'Bhat / sqrt(c'*covBhat*c)
    p_val = scipy.stats.norm.sf(abs(t))*2 # follows t-distribution
    statdict = {"diff":diff[0], "t":t[0], "p_value":p_val[0] }
    return statdict
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
# One of example bed file among 30 TFs :
# bed_narrowpeak = glob.glob("TFs/GATA1/ENCFF853VZF.bed")
#     genome = "hg38" # for ChIP-seq assembly
#     encode_ID = "ENCFF853VZF"
#     TF = "GATA1"
#     train_ChIP(bed_narrowpeak,genome,encode_ID,TF)

#2: by using TF_trainset constructed previously
def save_params(TF_trainset): # if user want to save pre-computed paramaters
    pbm_format = pd.read_csv(TF_trainset, sep="\t", header=None)
    ENCODE_ID = TF_trainset.split("_")[1]
    TF_name = TF_trainset.split("_")[2].split(".")[0]
    df_chip = read_chip(pbm_format,log2trans) # frequency table
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None, ENCODE_ID)  # run model
    with open(f"outputs/params_{ENCODE_ID}_{TF_name}.pkl", "wb") as file:
        pickle.dump(sgd_chip_none, file) # pre-computed coefficients/covariance matrix of features
    print(f"outputs/{ENCODE_ID}_{TF_name} is trained and saved!")
    return
def pred_vep(model,ENCODE_ID,vep_seq): # prediction part using pre-computed model paramaters
    stat_df = vep_seq.apply(lambda x: mypredict(x["sequence"],x["altered_seq"],model[1],model[-1]),
                                                   axis=1,result_type="expand")
    pred_df = pd.concat([vep_seq,stat_df],axis=1)
    pred_df.to_csv(f"outputs/pred_{ENCODE_ID}.csv", index=False)
    return pred_df
# second option to predict mutations, using fasta,bed and encode_id inputs (without using pre-computed parameters)
def pred_mutation(fasta_path, bed_path, ENCODE_ID,vep_seq): # prediction of VEP file
    pbm_format = bedtotrainset(fasta_path, bed_path, ENCODE_ID)
    df_chip = read_chip(pbm_format, log2trans)  # count table of trainset
    sgd_chip_none = apply_sgd(df_chip, minmax, "squared_error", None,ENCODE_ID)
    pred_sgd_chip_none = pred_vep(sgd_chip_none,ENCODE_ID,vep_seq)
    return pred_sgd_chip_none, sgd_chip_none

#----------------------------------------------------------------------------------------------

# VEP/VCF operations:

def altered_seq(df): # fetching -30/+30 sequences from the center mutation point
    center = 30
    dna_sequences = df["sequence"]
    allele_ref, allele_alt = df["ref"], df["alt"]

    if len(allele_ref) <= len(allele_alt):
        if allele_ref == "-":  #insertion (-/NNNNN)
            altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center+1) :]
        elif len(allele_ref) == len(allele_alt): # balanced point mutation or doublets+
            if allele_ref == "-": # point insertion (-/N)
                altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center + 1):]
            elif allele_alt == "-": # point deletion (N/-)
                altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
            else: # (N/N)
                altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1):]
        else: # unbalanced insertion (N/NNNN)
            altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1 + len(allele_ref)):]
    else:  # Deletion
        if allele_alt == "-": # deletion (NNN/-)
            altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
        else: # unbalanced deletion # (NNNN/N)
            altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + 1 + len(allele_ref)):]
    return altered_sequence

def vep_to_bed(vep_file):
    # VEP_raw data retrieved from Sana
    vep_data = pd.read_csv(vep_file)
    vep_data.drop_duplicates(inplace=True) # duplications available
    vep_data = vep_data.reset_index(drop=True)
    vep_data["chr"] = "chr" + vep_data["chr"] # adding chr format for MEME tool
    vep_data["start"] = vep_data["start"]-1 # bed format

    ind = vep_data[vep_data["chr"] == "chrUn_KI270742v1"].index[0] # discard alternative chr variant
    vep_data = vep_data.drop(index=ind).reset_index(drop=True)

    bed_format = pd.concat([vep_data["chr"], (vep_data["start"]-30),(vep_data["start"]+30)],axis=1) # getting -30+ sequences
    bed_format.columns = ["chr","start","end"]
    bed_format.to_csv("vep_to_bed.txt",header=False, index=False,sep="\t")
    print("VEP file is converted to bed format!")
    return  bed_format

def bed_to_seq(vep_file,bed_file,genome,id):
    ref_genome = pyfastx.Fasta(f"genome_assembly/{genome}.fa.gz")
    with open(bed_file, "r") as bed_data, open(f"{id}_fasta.txt", "w") as output_file: # bed to fasta format
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][start:end]
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
    # Fetching Sequences in 60 bps
    with open(f"{id}_fasta.txt") as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">","").replace("(+)","") for x,y in enumerate(fasta) if x%2 == 0]
    seqs = [y.rstrip().upper().replace("N","") for x,y in enumerate(fasta) if x%2 != 0]
    vep_60seq = pd.DataFrame({"region": locs, "sequence": seqs})
    vep_data = pd.read_csv(vep_file)
    vep_data["sequence"] = vep_60seq["sequence"]
    vep_data["altered_seq"] = vep_data.apply(altered_seq, axis=1)
    vep_data.to_csv("VEP_seq.csv", index=False)
    return vep_data
#-------------------------------------------------------------------------------------------------

# Aggregate all TF-models
def mean_update(current_mean, current_count, new_value):
    if new_value < 0:
        updated_sum = (-current_mean) * current_count + new_value
    else:
        updated_sum = current_mean * current_count + new_value
    updated_count = current_count + 1
    updated_mean = updated_sum / updated_count
    return abs(updated_mean)
def gain_or_loss(row,alpha):
    q_values = row['adj_pvalue']
    p_values = row["p_value"]
    diff = row['diff']
    ind = row["index"]
    if q_values <= alpha:
        if diff < 0: # if diff score is negative, then loss of TF exist
            vcf_seq["TF_loss"][ind] += 1
            vcf_seq["TF_loss_diff"][ind]=mean_update(vcf_seq["TF_loss_diff"][ind],len(vcf_seq["TF_loss_detail"][ind]), diff)
            vcf_seq["TF_loss_detail"][ind].append([TF_name, diff, p_values,q_values])
        elif diff > 0: # if diff score is positive, then gain of TF exist
            vcf_seq["TF_gain"][ind] += 1
            vcf_seq["TF_gain_diff"][ind]=mean_update(vcf_seq["TF_gain_diff"][ind],len(vcf_seq["TF_gain_detail"][ind]), diff)
            vcf_seq["TF_gain_detail"][ind].append([TF_name, diff, p_values,q_values])
def addcolumn_gain_loss(vcf_seq_path,pred_vcfs,alpha):
    global vcf_seq, TF_name
    vcf_seq = pd.read_csv(vcf_seq_path)
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
    vcf_seq.to_csv(f"vep_loss_gain_data_{alpha}.csv",index=False)

#-------------------------------------------------------------------------------------------------------------------

bed_format = vep_to_bed("final_dataset.csv") # convert bed_file
bed_to_seq("final_dataset.csv","vep_to_bed.txt","hg19","vep_noncoding") # convert seq-info file
vep_seq = pd.read_csv("VEP_seq.csv") # vep file including offset sequences

def main_flow(vep_seq):
    nonrev_list = gen_nonreversed_kmer(6) # 2080 features (6-mer DNA)
    # save the coefficients of features and covariance of the features (optional)
    TF_trainsets = glob.glob("TFs_trainset/*.txt") # TF trainset format
    for TF in TF_trainsets:
        save_params(TF) # run models using SGD and only saves coefficients/covariance matrix of features
    # Creating pre-computed-pred VCF files
    params = glob.glob("params/*.pkl") # list of location names
    param_dict = {} # store pre-computed parameters for 30 TFs
    for param in params: # I have 50 pre-computed parameters of models
        with open(param,"rb") as file:
            param_dict[param] = pickle.load(file)
    # for each TF models, generating diff score,t-value and p-value in VEP file
    for id,param in param_dict.items(): # generating pred_files
        ENCODE_ID = id.split("_")[1]
        TF_name = id.split("_")[2].split(".")[0]
        pred_vep(param, ENCODE_ID,TF_name,vcf_seq) # predictions of TF models
    p_vls = [0.05,0.01,0.001,0.0001,0.00001]
    for x in p_vls: # different thresholds of p-value
        folder = f"pred_VEP_{x}" # x : alpha value for statistical significance
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"pred_VEP_{x} folder is created!")
        pred_files = glob.glob(f"outputs/pred_*.csv")
        addcolumn_gain_loss(vep_seq, pred_files, x)
        print(f"predictions having {x} alpha value are completed!")

