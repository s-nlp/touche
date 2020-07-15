# Touche
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

