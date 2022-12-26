# GCAP
An end-to-end deep learning framework for predicting the seriousness of adverse reactions to drugs. GCAP has two tasks, one is to predict whether the adverse reactions are serious to drugs, and the other is to identify the seriousness classes of adverse reactions to drugs.




# Requirements
* python == 3.6
* pytorch == 1.6
* Numpy == 1.16.2
* scikit-learn == 0.21.3
* dgl == 0.7.2
* dgllife == 0.2.8


# Files:

1.data

drug_side_association_matrix.pckl: the known drug-ADR interaction matrix.

drug_side_serverity_matrix.pkl: the serious drug-ADR interaction matrix.

drug_smiles.pkl: the smiles sequences of drugs.

final_sample.pkl: all known drug-ADR interactions and corresponding labels.

side_vector_level_123.pkl: Semantic feature vectors of ADRs.

If you want to view the value stored in the file, you can run the following command:

```bash
import pickle
import numpy as np
gii = open(‘data’ + '/' + 'drug_side_association.pkl', 'rb')
drug_side_association = pickle.load(gii)
```


2.Code

Network.py: This function contains the network framework of our entire model and is based on pytorch 1.6. The model includes multiple CNN and GCN layers.

Cross_validation.py: This function can test the predictive performance of our model under ten-fold cross-validation.


# Train and test folds
python cross_validation.py --rawdata_dir /Your path --num_epochs Your number --batch_size Your number

rawdata_dir: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

num_epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training and testing.

All files of Data and Code should be stored in the same folder to run the model.

Example:

```bash
python cross_validation.py --rawdata_dir /data --num_epochs 50 --batch_size 128
```
# Contact 
If you have any questions or suggestions with the code, please let us know. Contact Haochen Zhao at zhaohaochen@csu.edu.cn
