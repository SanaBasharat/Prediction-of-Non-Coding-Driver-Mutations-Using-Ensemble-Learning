Firstly, if you want to compute 30 TF models using SGD algorithm by yourself. ( you can obtain the necessary TF-train files from the/data folder on the GitHub Repo. )

You can skip this step if you want to use pre-computed parameter files in the /TF_outputs folder. (SKIP TF_model_generator.py)
*biocoder : it is an utility python file for conversion between DNA string and binary formation

1-) run vep_preprocess.py
    from TF_binding_Effect_Prediction.vep_preprocess import vep_to_seq
    input_file = "data/test_data_final.csv"
    # 3 parameters : input_filepath , genome assembly ("hg19" or "hg38"), an id name as you prefer
    vep_to_seq(input_vep_filepath=input_file,genome_ref="hg19",id_name="vep_noncoding")

2-) run the prediction_run.py file to obtain predictions of TF-binding effects (for different alpha threshold values).
    from TF_binding_Effect_Prediction.prediction_run import pred_run
    # 1 parameter : processed vep file path created from vep_process.
    pred_run(vep_processed_filepath="TF_binding_Effect_Prediction/TF_outputs/VEP_seq.csv")

Outputs :
TF_binding_Effect_Prediction folder :
a.you will be creating 3 vep files (bed.txt,fasta.txt, sequence.csv) after vep_preprocess.py
b.you will be creating 2 folders
    b1. "preds_eachTF" which include individually TF-based predictions
    b2. "TFcombined_results"  which include 5 aggregated csv files having different level of alpha threshold. You can choose one of them according to your conservative level.
