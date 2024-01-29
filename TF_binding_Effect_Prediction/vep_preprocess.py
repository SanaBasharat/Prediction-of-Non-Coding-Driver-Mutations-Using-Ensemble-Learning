import pandas as pd
import os
import pyfastx

pd.options.mode.chained_assignment = None # ignoring the warning

# VEP/VCF operations:

def altered_seq(df): # fetching -30/+30 sequences from the center mutation point
    center = 30
    dna_sequences = df["sequence"]
    allele_ref, allele_alt = df["ref"], df["alt"]
    if len(allele_ref) <= len(allele_alt):
        if (allele_ref == "-" and len(allele_ref) > 1):  #insertion (-/NNNNN) # for VCF format (previous base including) #
            altered_sequence = dna_sequences[:(center+1)] + (allele_alt[1:]) + dna_sequences[(center+1) :]
        elif len(allele_ref) == len(allele_alt): # balanced point mutation or doublets+
            if allele_ref == "-": # point insertion (-/N)
                altered_sequence = dna_sequences[:(center+1)] + allele_alt + dna_sequences[(center + 1):]
            elif allele_alt == "-": # point deletion (N/-)
                altered_sequence = dna_sequences[:center] + dna_sequences[center + len(allele_ref):]
            else: # (N/N) # problem fixed
                altered_sequence = dna_sequences[:center] + allele_alt + dna_sequences[(center + len(allele_alt)):]
        else: # unbalanced insertion (N/NNNN) # [1:]
            altered_sequence = dna_sequences[:(center+1)] + allele_alt[1:] + dna_sequences[(center + 1 + len(allele_ref[1:])):]
    else:  # Deletion
        if allele_alt == "-": # deletion (NNN/-) # -1
            altered_sequence = dna_sequences[:(center+1)] + dna_sequences[((center+1) + len(allele_ref[1:])):]
        else: # unbalanced deletion # (NNNN/N) [1:]
            altered_sequence = dna_sequences[:(center+1)] + allele_alt[1:] + dna_sequences[(center + 1 + len(allele_ref[1:])):]
    return altered_sequence
def vep_to_bed(vep_data):
    # VEP_raw data retrieved from Sana
    # vep_data = pd.read_csv(vep_file)
    vep_data.drop_duplicates(inplace=True) # duplications available
    vep_data = vep_data.reset_index(drop=True)
    vep_data["chr"] = "chr" + vep_data["chr"] # adding chr format for MEME tool
    vep_data["start"][:865] = vep_data["start"][:865]-1 # bed format against vcf
    # ind = vep_data[vep_data["chr"] == "chrUn_KI270742v1"].index[0] # discard alternative chr variant
    # vep_data = vep_data.drop(index=ind).reset_index(drop=True)
    bed_format = pd.concat([vep_data["chr"], (vep_data["start"]-30),(vep_data["start"]+30)],axis=1) # getting -30+ sequences
    bed_format.columns = ["chr","start","end"]
    bed_format.to_csv("TF_binding_Effect_Prediction/TF_outputs/vep_to_bed.txt",header=False, index=False,sep="\t")
    return  bed_format
def bed_to_seq(vep_data,bed_file,genome,id):
    ref_genome = pyfastx.Fasta(f"data/genome_assembly/{genome}.fa.gz")
    with open(bed_file, "r") as bed_data, open(f"TF_binding_Effect_Prediction/TF_outputs/{id}_fasta.txt", "w") as output_file: # bed to fasta format
        for line in bed_data:
            chrom, start, end = line.strip().split("\t")[:3]
            start = int(start)
            end = int(end)
            sequence = ref_genome[chrom][start:end]
            output_file.write(f">{chrom}:{start}-{end}\n")
            output_file.write(str(sequence) + "\n")
    print("2- Fasta file is created!")
    # Fetching Sequences in 60 bps
    with open(f"TF_binding_Effect_Prediction/TF_outputs/{id}_fasta.txt") as f:  # fasta file to extract sequences
        fasta = f.readlines()
    locs = [y.split()[0].replace(">","").replace("(+)","") for x,y in enumerate(fasta) if x%2 == 0]
    seqs = [y.rstrip().upper().replace("N","") for x,y in enumerate(fasta) if x%2 != 0]
    vep_60seq = pd.DataFrame({"region": locs, "sequence": seqs})
    # vep_data = pd.read_csv(vep_file)
    vep_data["sequence"] = vep_60seq["sequence"]
    vep_data["altered_seq"] = vep_data.apply(altered_seq, axis=1)
    vep_data.to_csv("TF_binding_Effect_Prediction/TF_outputs/VEP_seq.csv", index=False)
    return vep_data
def vep_to_seq(data,genome_ref="hg19",id_name="vep_noncoding"):
    if not os.path.exists("TF_binding_Effect_Prediction/TF_outputs"):  # the main output file
        os.makedirs("TF_binding_Effect_Prediction/TF_outputs")
    bed_format = vep_to_bed(data)  # convert bed_file
    print("1- Bed file is created!")
    vep_seq = bed_to_seq(data,"TF_binding_Effect_Prediction/TF_outputs/vep_to_bed.txt",genome_ref,id_name) # convert seq-info file
    print("3- VEP file with sequences is created!")
    return vep_seq

