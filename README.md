# Data and Code for Paper "DrugDoctor"
Our model was implemented using PyTorch 1.9.1 with Python 3.8 and trained on an NVIDIA GeForce RTX 4090 GPU.
## Folder Specification
* `drugRec.yml`: the environment for running
* `data/` folder contains necessary data or scripts for generating data. 
  * `processing.py`: The python script responsible for generating voc_final.pkl, records_final.pkl and ddi_A_final.pkl.
  * `input/`
    * `drug-atc.csv`: drug to atc code mapping file.
    * `ndc2RXCUI.txt`: NDC to RXCUI mapping file.
    * `drugbank_drugs_info.csv`: drug information table downloaded from drugbank here https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing, which is used to map drug name to drug SMILES string.
    * `drug-DDI.csv`: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    * `RXCUI2atc4.csv`: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2atc_level4.csv.
  * `output/`
    * `atc3toSMILES.pkl`: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict 
    * `ddi_A_final.pkl`: ddi adjacency matrix
    * `ddi_mask_H.pkl`: H mask structure (This file is created by ddi_mask_H.py), used in Safedrug baseline 
    * `substructure_smiles.pkl`
    * `ehr_adj_final.pkl`: if two drugs appear in one set, then they are connected
    * `records_final.pkl`: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split
    * `voc_final.pkl`: diag/prod/med index to code dictionary
* `src/`
  * `main.py`: Train or evaluate our DrugDoctor Model.
  * `models.py`: our model
  * `util.py`
  * `data_loader.py`
## Quick start
### Step 1: Data Processing
> Note: The usage of MIMIC-III datasets requires certification, so it's illegal for us to provide the raw data here.
> Therefore, if you want to have access to MIMIC-III datasets, you have to obtain the certification first and then download it from https://physionet.org/content/mimiciii/1.4/.
* The need to download three csv file from the MIMIC-III dataset: PRESCRIPTIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES_ICD.csv.
* Then, processing the data to get a complete records_final.pkl:
```
python processing.py
```
### Step 2: Package Dependency
* create the conda environment through yml file:
```
conda env create -f drugRec.yml
```
### Step 3: Running 
```
python main.py
```
* Here is the argument:
```
usage: SafeDrug.py [-h] [--Test] [--model_name MODEL_NAME]
               [--resume_path RESUME_PATH] [--lr LR]
               [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]
optional arguments:
  --Test                    test mode
  --model_name MODEL_NAME   model name
  --resume_path RESUME_PATH resume path
  --lr LR                   learning rate
  --target_ddi TARGET_DDI   target ddi
  --batch_size              batch size 
  --emb_dim                 dimension size of embedding
  --kp KP                   coefficient of P signal
  --dim DIM                 dimension
  ```
## Acknowledgement
We sincerely thank these repositories [SafeDrug](https://github.com/ycq091044/SafeDrug), [MoleRec](https://github.com/yangnianzu0515/MoleRec) and [COGNet](https://github.com/BarryRun/COGNet) for their well-implemented pipeline upon which we build our codebase.
