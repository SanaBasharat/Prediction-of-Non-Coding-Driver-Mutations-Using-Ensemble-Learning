# Prediction-of-Non-Coding-Driver-Mutations-Using-Ensemble-Learning
This repository contains the code and data for our paper titled "Prediction of Non-coding Driver Mutations Using Ensemble Learning".

### Abstract
Driver coding mutations are extensively studied and frequently detected by their deleterious amino acid changes that affect protein function. However, non-coding mutations need further analysis from various aspects and require experimental validation to delineate them as driver non-coding mutations. Here, we employ XGBoost (eXtreme Gradient Boosting) algorithm to predict driver non-coding mutations based on novel long-range interaction features and engineered transcription factor binding site features augmented with features from existing annotation and effect prediction tools. Regarding novel long-range interaction features, we capture the frequency and spread of interacting regions overlapping with the non-coding mutation of interest. For this purpose, we use self-balancing trees to find overlaps within chromatin loop files and store the interacting regions as separate tree structures. For TF-binding engineered features, we train 30 TF models utilizing the stochastic gradient descent (SGD) algorithm to predict changes in transcription factor binding affinity by giving more weight to the non-coding mutations located at known transcription factor binding sites. We also include features from existing annotation and effect prediction tools, some of which rely on deep learning methods, relating to splicing effect, number of associated protein products, variant consequences, biotypes, and others. For the known driver and non-driver non-coding mutations, the resulting aggregated dataset is trained with our gradient boosting model to predict driver non-coding mutations versus passenger non-coding mutations. Furthermore, we elaborate on the results by using explainable AI methodologies.

### Authors:

Sana Basharat<br>
Department of Data Informatics,<br>
Graduate School Of Informatics<br>
Middle East Technical University<br>
Ankara 06800, Türkiye<br>
ramal.huseynov@metu.edu.tr

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
