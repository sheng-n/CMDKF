# CMDKF Submitted to "Bioinformatics"

## 1. Overview
The code for paper "Causal metapath and domain knowledge fusion for predicting miRNA-drug-disease triplet resistance and sensitivity associations". The repository is organized as follows:

+ `data/MDR_data/MDS_data` contains the data in the paper:
  * `m_r_d_pos_pairs.txt` contains known miRNA-drug-disease triple resistance and sensitivity associations;
  * `miRNA_sequence.xlsx` contains miRNA ID, sequences;
  * `drug_smiles.xlsx` contains drug ID, smiles;
  * `disease_name.xlsx` contains disease ID;
  * `mi_kmer.txt` contains miRNA k-mer feature;
  * `drug_maccs.txt` contains drug maccs feature;
  * `dis_sim.txt` contains disease similarity;
  * `CV_data` contains 5-cv data;
  * `indepent_data` contains indepent data ;
    
+ `code/`
  * `data_splits.py`contains the 5-fold cv and independent test set splits;
  * `utils_.py`contains matrics, parameters;
  * `data_process.py` contains the preprocess of data;
  * `model.py` contains CMDKF's model layer;
  * `train.py` contains training and testing code;
  * `main.py` runs code;
  
## 2. Dependencies
* torch == 2.1.2+cu118
* torch-geometric == 2.4.0
* numpy == 1.24.4
* RDKit v2023.9.6

## 3. Quick Start
Here we provide a example:

1. Download and upzip our data and code files
2. Run "data_splits.py"
3. Run "main.py" 

## 5. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
