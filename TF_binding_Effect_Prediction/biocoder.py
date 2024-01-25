import pandas as pd
import numpy as np

nucleotides = {'A':0,'C':1,'G':2,'T':3}
numtonuc = {0:'A',1:'C',2:'G',3:'T'}
complement = {0:3,3:0,1:2,2:1}

def seqpos(kmer,last):
    return 1 <<  (1 + 2 * kmer) if last else 1 << 2 * kmer;
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
def gen_nonreversed_kmer(k):
    nonrevk = list()
    for i in range(seqpos(k,False),seqpos(k,True)):
        if i <= revcomp(i):
            nonrevk.append(i)
    return nonrevk
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

def nonr_olig_freq(seqtbl, kmer=6):
    nonrev_list = gen_nonreversed_kmer(kmer)
    rightseparator = kmer
    leftseparator = rightseparator
    # Use numpy for faster array operations
    seqint_arr = np.array(seqtbl) # adjust dtypes if it is necessary
    olig_df = np.zeros((len(seqtbl), len(nonrev_list)))
    mask = (4 ** kmer) - 1
    for i, cpy in enumerate(seqint_arr):
        while cpy > mask:
            cur = cpy & mask
            right = cur & ((4 ** rightseparator) - 1)
            left = (cur >> (2 * leftseparator)) << (2 * rightseparator)
            seqint = left | right

            r = (1 << (2 * kmer)) | seqint  # append 1
            rc = revcomp(r)
            r = r if r < rc else rc  # Use conditional operator
            # Use numpy indexing for faster updates
            olig_df[i, nonrev_list.index(r)] += 1
            cpy >>= 2
    return pd.DataFrame(olig_df, columns=nonrev_list)

