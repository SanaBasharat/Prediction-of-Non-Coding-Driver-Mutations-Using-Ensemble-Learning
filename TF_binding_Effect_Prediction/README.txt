Firstly, if you want to compute 30 TF models using SGD algorithm by yourself. ( you can obtain the necessary TF-train files from the/data folder on the GitHub Repo. )

You can skip this step if you want to use pre-computed parameter files in the /TF_outputs folder. (SKIP TF_model_generator.py)

1-) Run the TF_prediction_generator.py file to obtain predictions of TF-binding effects (for different alpha threshold values).
    from TF_binding_Effect_Prediction.TF_prediction_generator.py import generate_TF_predictions # import following module

    # 3 parameter : processed vep data frame created from vep_process, genome assembly, and preferred p-value):
    generate_TF_predictions(df, genome_ref, p_value)

**
    # 3 parameters : input_filepath , genome assembly ("hg19" or "hg38"), an id name as you prefer
    vep_to_seq(input_vep_filepath=input_file,genome_ref="hg19",id_name="vep_noncoding")
** 
    biocoder.py : it is a utility python file for conversion between DNA string and binary formation

Outputs :
TF_binding_Effect_Prediction folder :
a.you will be creating 3 vep files (bed.txt,fasta.txt, sequence.csv) after vep_preprocess.py
b.you will be creating 2 folders
    b1. "preds_eachTF" which include individually TF-based predictions
    b2. "TFcombined_results"  include 5 aggregated csv files with different alpha threshold levels. You can choose one of them according to your conservative level.
    b3. "tf_outpu" data frame from TF_prediction_generator.py 
