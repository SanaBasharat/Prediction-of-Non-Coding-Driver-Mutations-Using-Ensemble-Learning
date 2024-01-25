from TF_binding_Effect_Prediction.vep_preprocess import vep_to_seq
from TF_binding_Effect_Prediction.prediction_run import pred_run

# Mutation input file :
input_file = "data/test_data_final.csv"

# 3 parameters : input_filepath , genome assembly ("hg19" or "hg38"), an id name as you prefer
vep_to_seq(input_vep_filepath=input_file,genome_ref="hg19",id_name="vep_noncoding")

# 1 parameter : processed vep file path created from vep_process.
pred_run(vep_processed_filepath="TF_binding_Effect_Prediction/TF_outputs/VEP_seq.csv")

