
# Touche 2022

## Data your need

1) Chekpoints of ColBERT model.
All mentioned chekpoints are available at https://zenodo.org/record/7078839#.YyG3VmRBxhE

2) Collection of Touche passages in proper format.
In Touche/collections folder, touche21_psgs.tsv file

3) Collection of Touche queries in proper format
Touche/collections folder, queries_22.tsv

The main track of this stage is employing the ColBERT model to rank passages.

## Train

Employing Colbert consists of 4 stage:

- Create regular index
- Create faiss index
- Retrieve

**Before Retrieving**

You may have acsess to:
1) ColBert models checkpoints in ZENODO
2) Full ColBERT pipeline for TOUCHE

If you have only ColBERT model checkpoints (without archieve version of full ColBERT pipeline), as downloaded from ZENODO, you firstly should create **index** and **index faiss**.

CUDA_VISIBLE_DEVICES="2,3,4,5" python3 -m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --checkpoint /notebook/ColBERT/regular_checkpoints/folder_with_main_chekpoints/**edinburg_colbert.dnn** --collection /notebook/ColBERT/collections/**touche21_psgs.tsv** --index_root /notebook/ColBERT/indexes/ --index_name full_bert_mscmarco --root /notebook/ColBERT/experiments/ --experiment full_bert_msmarco
 
These steps take a significant time, so we archive the folder with checkpoints and pre-counted indexes. We also represent provided queries and documents collections in the “collections” folder in the current repository and “ColBERT/collections” folder in archived pipeline.
If you have archieved pipeline, you can reproduce result by the following command:

**Model, fine-tuned on the previous Touche data:**

Retrieve:

CUDA_VISIBLE_DEVICES="4, 5" python3 -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --nprobe 32 --partitions 32768 --faiss_depth 1024 --queries /../Touche22/queries_22.tsv --checkpoint /../ColBERT/experiments/MSMARCO-fn/train.py/msmarco.ft.l2/checkpoints/filetune_colbert_232.dnn --index_root /../ColBERT/indexes/MSMARCO.L2.32x200k_22 --index_name ​​pretrain_21 --partitions 32768 --root /../ColBERT/experiments/ --experiment MSMARCO-fn

**Model, pre-trained on MSMarco dataset:**

Retrieve:

CUDA_VISIBLE_DEVICES="4, 5" python3 -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --nprobe 32 --partitions 32768 --faiss_depth 1024 --queries /../Touche22/queries_22.tsv --checkpoint /../ColBERT/experiments/MSMARCO-psg_new/train.py/msmarco.psg.l2/checkpoints/colbert_trained_by_us_300000.dnn --index_root /../ColBERT/indexes/ --index_name MSMARCO.L2.32x200k_22 --partitions 32768 --root /../ColBERT/experiments/ --experiment MSMARCO-psg

**Model, based on checkpoint from Glasgow University:**

Retrieve:

CUDA_VISIBLE_DEVICES="4, 5" python3 -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --nprobe 32 --partitions 32768 --faiss_depth 1024 --queries /../Touche22/queries_22.tsv --checkpoint /../ColBERT/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/edinburg_colbert.dnn --index_root /../ColBERT/indexes/ --index_name MSMARCO.L2.32x200k_22_old --partitions 32768 --root /../ColBERT/experiments/ --experiment MSMARCO-psg

The post-processing of obtained result are in Test folder.

# Touche 2020
Code for reproducing of our submission to the CLEF-2020 shared task on argument retrieval.

This repository contains some approaches to information retrieval task.
Every approach corresponds to its own \*.py module.

| Method | Module |
| --- | --- |
| Baseline based on an inverted index output | baseline.py |
| LSTM ULMFiT | lstm_ulmfit.py |
| Transformer's attention | transformers.py |
| Bidirectional Encoder Representations from Transformer (BERT) | bert.py |
| Combination of Baseline, number of comparative sentences and comparativestructure extraction | baseline_cam_obj.py |
| Combination of ULMFit, number of comparative sentences and comparative structure extraction | lstm_ulmfit_cam_obj.py |

# Technical

The proper **Transformers**, **Fastai**, **Bert** libraries are in **requrements.txt**

# Bert-service for token distributed representation.

**bert.py**, **lstm_ulmfit.py**, **lstm_ulmfit_cam_obj.py**

To run some \*.py modules needed bert-service for bert token distributed representation on the restarted machine, we need first to start bert-service which in turn needs restored graph in TensorFlow. Graph restoring and service starting are reproduced in the bash script before the main module starts (**restore_tf_graph.py**, **run_bash_tira3.sh**).
The one of possible Bert weights are on https://drive.google.com/file/d/17VC9ErQ-xRrmyCUf8Oy5gFF-LcsnJosZ/view?usp=sharing



# Degree of comparativeness and comparative structure extraction

**baseline_cam_obj.py**, **lstm_ulmfit_cam_obj.py**

Copy 

- infersent.allnli.pickle(https://drive.google.com/file/d/1bmCg-jPWZ8M0X5RLWtZq0Xq7wYUqc_Og/view?usp=sharing) to "/touche/cam/data/" 

- model.pkl(https://drive.google.com/file/d/1--REAj57WzYC36H6ji2vl7NH5O8fej25/view?usp=sharing) to "touche/cam/src/Backend/data/" 

- glove.840B.300d.txt(https://yadi.sk/d/cgJPQ8RwpFgS-g) to "touche/cam/GloVe/"

- berttt.hdf5(https://drive.google.com/file/d/1kU_4pqWgJ29kAbhBoMbCrjGZDtQ0zLjY/view?usp=sharing) to "touche/external_pretrained_models"

# Transformers

Use *pytorch_transformers* library.

Model: 'bert-base-uncased'

The second output of the forward are the attention's weihgts.

Since the transformer can process 2 sentences are connected by a delimiter [SEP], for every response we create __["[CLS] " + query + " [SEP] " + answer_body + " [SEP]"]__.

Than we apply self-attention and explore the map of the obtained weights. We consider only non-diagonal parts of map are corresponded to weights of word according tokens from __another__ sentence. 

As we can see in **touche-explain_transformers.ipynb**, to count closeness of query and response we need to take into account the 3rd, 4th, 9th and 10 head of attention. The 3rd head highlights the similar words, another heads are responsible for closeness in meaning. 

The score is sum of weights in this head on non-diagonal place, excluding the weights corresponding to the special tokens ([CLS], [SEP], ...).

