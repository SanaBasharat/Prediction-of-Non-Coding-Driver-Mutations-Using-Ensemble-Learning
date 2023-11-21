# Prediction-of-Non-Coding-Driver-Mutations-Using-Ensemble-Learning
This repository contains the code and data for our paper titled "Prediction of Non-coding Driver Mutations Using Ensemble Learning".

### Install Requirements
Using Conda:<br>
`conda env create -f environment.yml`<br>
or<br>
`conda create --name <env_name> --file requirements.txt`<br>

Using pip:<br>
`pip install -r requirements.txt`

### Important Note
All work done in this paper is using the GRCH37 assembly. In cases where GRCH38 assembly had to be used, the mutations were first converted, used and then joined back with the original data.<br><br>
Files containing data obtained from TCGA have been purposefully removed. Data used for validation/testing (which was obtained from open-access sources) is present.

### Abstract
Driver coding mutations are extensively studied and frequently detected by their deleterious amino acid changes that affect protein function. However, non-coding mutations need further analysis from various aspects and require experimental validation to delineate them as driver non-coding mutations. Here, we employ XGBoost (eXtreme Gradient Boosting) algorithm to predict driver non-coding mutations based on novel long-range interaction features and engineered transcription factor binding site features augmented with features from existing annotation and effect prediction tools. Regarding novel long-range interaction features, we capture the frequency and spread of interacting regions overlapping with the non-coding mutation of interest. For this purpose, we use self-balancing trees to find overlaps within chromatin loop files and store the interacting regions as separate tree structures. For Transcription Factor (TF) binding engineered features, we train 30 TF models utilizing the stochastic gradient descent (SGD) algorithm to predict changes in transcription factor binding affinity by giving more weight to the non-coding mutations located at known transcription factor binding sites. We also include features from existing annotation and effect prediction tools, some of which rely on deep learning methods, relating to splicing effect, number of associated protein products, variant consequences, biotypes, and others. For the known driver and non-driver non-coding mutations, the resulting aggregated dataset is trained with our gradient boosting model to predict driver non-coding mutations versus passenger non-coding mutations. We then use non-coding driver mutations found in other state-of-the-art studies, annotate them in a similar way, and pass them through our model in order to make a comparison. Furthermore, we elaborate on the results by using explainable AI methodologies. Our results show an above-average performance on the unseen test data and suggest that using our annotations and training the resulting data using gradient boosting trees, the classification between a non-coding driver versus passenger mutation is possible with relatively high degrees of accuracy.

### Authors:

Sana Basharat<br>
Department of Data Informatics,<br>
Graduate School Of Informatics<br>
Middle East Technical University<br>
Ankara 06800, Türkiye<br>
sana.basharat@metu.edu.tr

Ramal Hüseynov<br>
Department of Health Informatics,<br>
Graduate School Of Informatics<br>
Middle East Technical University<br>
Ankara 06800, Türkiye<br>
ramal.huseynov@metu.edu.tr

Hüseyin Hilmi Kılınç<br>
Department of Health Informatics,<br>
Graduate School Of Informatics<br>
Middle East Technical University<br>
Ankara 06800, Türkiye<br>
hilmi.kilinc@metu.edu.tr<br>

Burçak Otlu<br>
Department of Health Informatics,<br>
Graduate School Of Informatics<br>
Middle East Technical University<br>
Ankara 06800, Türkiye<br>
burcako@metu.edu.tr<br>
