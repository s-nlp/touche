import requests
#r = requests.post('http://10.30.99.211:8261/gpt_small', data = "What is better for deep learning Python or Matlab?")
#print (r.status_code)
import requests
from xml.dom import minidom
import sys
import argparse

#numpy
import numpy as np

#torch
import torch, torch.nn as nn
import torch.nn.functional as F

#custom request and preprocessing
from help_response import create_list_of_unigue_answers, cleanhtml, clean_punct, read_xml

from xml.dom import minidom
import re

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
#bc = BertClient()

import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def make_scores_1(query, answer_titles):
    query_emb = bc.encode([query])
    scores = [cosine_similarity(query_emb.reshape(1, -1), bc.encode([cleanhtml(answer_title)]).reshape(1, -1))[0][0] for answer_title in answer_titles]
    return scores
    
def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics.xml'):
    
    # load responses for all queries from file 
    my_response_list = create_list_of_unigue_answers(input_dir = input_dir, input_file = input_file)
    
    print ("len my responses list", len(my_response_list))
    #print (my_response_list)
    list_of_tuples = read_xml(input_dir + '/' + input_file)
    common_list = []

    
    print ("baseline", flush = True)
    for ind_q, elem in enumerate(list_of_tuples):
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'MyBaselineFilterResponse'
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
        part_of_commom_list = list(zip(qids, Q0s, docs, scores0, tags))
        part_of_commom_list = sorted(part_of_commom_list, key = lambda x: x[3], reverse = True) 

        qids, Q0s, docs, scores, tags = zip(*part_of_commom_list)

        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        common_list = common_list + part_of_commom_list

    with open('./from_tira/baseline.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--inp_dir', default='/notebook/touche/')
    parser.add_argument('--out_dir', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.out_dir, input_dir = args.inp_dir, input_file = args.inp_file)
    