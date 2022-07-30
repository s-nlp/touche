import requests
#r = requests.post('http://10.30.99.211:8261/gpt_small', data = "What is better for deep learning Python or Matlab?")
#print (r.status_code)
import requests
from xml.dom import minidom
import sys
import argparse

# transformers
import pytorch_transformers
from pytorch_transformers import *

#numpy
import numpy as np

#torch
import torch, torch.nn as nn
import torch.nn.functional as F

from xml.dom import minidom
import re

from help_response import create_list_of_unigue_answers, cleanhtml, clean_punct, read_xml

import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def make_scores_transformers(model, query, answers_bodies):
    scores = []
    for ind, elem in enumerate(answers_bodies):
        sentences = ["[CLS] " + query + " [SEP] " + answers_bodies[ind] + " [SEP]"]
        #print(sentences[0])

        # Tokenize with BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        #print ("Tokenize the first sentence:")
        #print (tokenized_texts[0])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_texts[0])
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = (len(tokenizer.tokenize(query)) + 2)*[0] + (len(tokenizer.tokenize(answers_bodies[ind])) + 1)*[1]
        cls_idx = 0
        sep_idx = len(tokenizer.tokenize(query))
        end_idx = len(tokenizer.tokenize(query)) + len(tokenizer.tokenize(answers_bodies[ind])) + 3
        #print (len(tokenizer.tokenize(query)), len(tokenizer.tokenize(answers_bodies[ind])))
        #print (len(indexed_tokens), len(segments_ids))
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        answ = model(tokens_tensor, segments_tensors)

        attensions = answ[2][3][0][0].detach().numpy() + answ[2][9][0][0].detach().numpy()
        #probs[cls_idx:sep_idx, sep_idx:] = 0.0
        score = np.sum(attensions[cls_idx+1:sep_idx-1, sep_idx+1:end_idx -1]) + np.sum(attensions[sep_idx+1:end_idx-1, cls_idx+1:sep_idx-2])
        #print (score)
        scores.append(score)
    return scores

def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics.xml'):
    
    #transformer part
    config_class, model_class, tokenizer_class = BertConfig, BertModel, BertTokenizer
    config = config_class.from_pretrained('bert-base-uncased')
    config.output_attentions=True
    model = model_class.from_pretrained('bert-base-uncased', config=config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # load responses for all queries from file 
    my_response_list = create_list_of_unigue_answers(input_dir = input_dir, input_file = input_file)
    
    print ("len my responses list", len(my_response_list))
    #print (my_response_list)
    list_of_tuples = read_xml(input_dir + '/' + input_file)
    common_list = []

    
    print ("transformers", flush = True)
    for ind_q, elem in enumerate(list_of_tuples):
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'MethodAttentionFilterResponse'
        responses = list(zip(*my_response_list[str(ind_q + 1)]))
        
        scores0 = responses[0]
        #print ("0")
        docs = responses[1]
        #print ("1")
        titles = responses[2]
        #print ("2")
        answers_bodies = responses[3]
        #print ("3")
        # print (scores0, scores3, scores)
        qids = [qid]*len(scores0)
        Q0s = [Q0 for elem in scores0]
        tags = [tag for elem in scores0]
        
        scores = make_scores_transformers(my_rnn, query, answers_bodies)
        
        part_of_commom_list = list(zip(qids, Q0s, docs, scores, tags))
        part_of_commom_list = sorted(part_of_commom_list, key = lambda x: x[3], reverse = True) 

        qids, Q0s, docs, scores, tags = zip(*part_of_commom_list)

        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        common_list = common_list + part_of_commom_list

    with open('./from_tira/transformers.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--inp_dir', default='/notebook/touche/')
    parser.add_argument('--out_dir', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.out_dir, input_dir = args.inp_dir, input_file = args.inp_file)
    