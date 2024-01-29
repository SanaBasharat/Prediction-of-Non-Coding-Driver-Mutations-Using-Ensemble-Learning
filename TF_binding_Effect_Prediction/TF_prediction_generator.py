from TF_binding_Effect_Prediction.vep_preprocess import vep_to_seq
from TF_binding_Effect_Prediction.prediction_run import pred_run
import pandas as pd

def generate_TF_predictions(df, genome_ref, p_value):
    # 3 parameters : input data , genome assembly ("hg19" or "hg38"), an id name as you prefer
    vep_to_seq_file = vep_to_seq(df, genome_ref, id_name="vep_noncoding")

    # 1 parameter : processed vep file returned from vep_process.
    tf_output = pred_run(vep_to_seq_file, p_value)
    tf_output.drop(['TF_loss_detail', 'TF_gain_detail', 'sequence', 'altered_seq'], inplace=True, axis = 1)
    return tf_output