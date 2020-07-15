import requests
from xml.dom import minidom
import sys
import argparse
import re

#custom request and preprocessing
from help_response import create_list_of_unigue_answers, cleanhtml, clean_punct, read_xml

#numpy
import numpy as np

#torch
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.models as Model

# BERT
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient()

#load fast ai model
from fastai.text.models import AWD_LSTM
from fastai.text.learner import get_language_model

import pandas as pd
import sys
sys.path.insert(0, "/notebook/touche/cam/src/Backend")
from utils.regex_service import find_pos_in_sentence

import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

def extract_objs_asp(model_for_extraction, input_string):
    model_for_extraction.from_string(input_string)
    obj1, obj2, predicates, aspects = model_for_extraction.get_params()
    model_for_extraction.clear_params()
    return (obj1.lower(), obj2.lower(), predicates, aspects)

import models1
extr = models1.extractor()


def prepare_sentence_DF(sentences, obj_a, obj_b):
    index = 0
    temp_list = []
    for sentence in sentences:
        pos_a = find_pos_in_sentence(obj_a, sentence)
        pos_b = find_pos_in_sentence(obj_b, sentence)
        if (pos_a == -1 or pos_b == -1):
            pos_a = 0
            pos_b = len(sentence) - 1
        if pos_a < pos_b:
            temp_list.append([obj_a, obj_b, sentence])
        else:
            temp_list.append([obj_b, obj_a, sentence])
        print (pos_a, pos_b)
        index += 1
    sentence_df = pd.DataFrame.from_records(temp_list, columns=['object_a', 'object_b', 'sentence'])

    return sentence_df

def count_score(text, nlu_tuple):
    (obj1, obj2, pred, asp) = nlu_tuple
    r = 1.0
    print (text)
    if (len(obj1) != 0 and len(obj2) != 0):
        print ("0")
        if (len(pred) != 0):
            pred = re.sub('[!#?,.:";]', '', pred[0])
            if (obj1 in text and obj2 in text and pred in text):
                r += 1.0
        if (len(asp) != 0):
            asp = re.sub('[!#?,.:";]', '', asp[0])
            if (obj1 in text and obj2 in text and asp in text):
                r += 1.0
        elif (obj1 in text and obj2 in text):
            r = 1.5
        elif (obj1 in text or obj2 in text):
            r = 1.2
    else:
        if (obj1) in text or (obj2) in text:
            r = 1.2
    return r

def make_scores_obj(query, answers):
    print ("make_scores_obj")
    (obj1, obj2, pred, asp) = extract_objs_asp(extr, query)
    print ("in make scores", obj1, obj2, pred, asp)
    scores_answers = [count_score(cleanhtml(answer), (obj1, obj2, pred, asp)) for answer in answers]
    return scores_answers

import nltk
nltk.download('punkt')

def make_scores_cam(query, titles, answers):
    print ("make_scores_cam")
    query = (cleanhtml(query))
    print (query, "/n")
    answers_clean = [(cleanhtml(answer)) for answer in answers]
    titles_clean = [(cleanhtml(title)) for title in titles]
    obj1, obj2, pred, asp = extract_objs_asp(extr, query)
    print ("obj1, obj2, pred, asp", obj1, obj2, pred, asp, "\n")
    number_of_comparative_sentences = []
    for ind, answer in enumerate(answers_clean):
        if (ind%10 == 0):
            print (ind)
        sentenсes = [titles[ind]] + nltk.tokenize.sent_tokenize(answer)
        dframe = prepare_sentence_DF(sentenсes, obj1, obj2)
        answ = classify_sentences(dframe, 'infersent')
        filt = (answ["BETTER"] >= 0.2) | (answ["WORSE"] >= 0.2)
        new_answ_df = answ.where(filt)
        new_answ_df = new_answ_df.dropna()
        number_of_comparative_sentences.append(len(new_answ_df))
    return number_of_comparative_sentences

import sys
sys.path.append("./cam/src/Backend/")
from ml_approach.classify import classify_sentences

import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def run_baseline(output_dir = '/notebook/touche/output_tira/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):    
    # clean answers from ChatNoir to avoid repeating
    my_response_list = create_list_of_unigue_answers(input_dir = input_dir, input_file = input_file)
    
    print ("len my responses list", len(my_response_list))
    
    #print (my_response_list)
    list_of_tuples = read_xml(input_dir + '/' + input_file)
    common_list = []
    scores_cam_list = []
    scores_obj_list = []
    print ("baseline cam obj", flush = True)    
    for ind_q, elem in enumerate(list_of_tuples):
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'Baseline_CAM_OBJ'
        responses = list(zip(*my_response_list[str(ind_q + 1)]))

        scores = responses[0]
        print ("ind_q", ind_q)
        docs = responses[1]
        #print ("1")
        titles = responses[2]
        #print ("2")
        answers_bodies = responses[3]
        print ("3")
        # print (scores0, scores3, scores)
        #scores = make_scores_lstm(my_rnn, query, answers_bodies)
        #save_obj(scores, "lstm scores")
        print ("4")
        qids = [qid]*len(scores)
        Q0s = [Q0 for elem in scores]
        queries = query*len(scores)
        tags = [tag for elem in scores]
        
        scores_obj = make_scores_obj(query, answers_bodies)
        scores_cam = make_scores_cam(query, titles, answers_bodies)
        scores_cam_list.append(scores_cam)
        scores_obj_list.append(scores_obj)
        
        multiplicat = [0.5*scores + scores_obj[ind] for ind, scores in enumerate(scores_cam)]
        new_scores = [multiplicat[ind]*score for ind, score in enumerate(scores[:20])]
        
        part_of_commom_list = list(zip(qids, Q0s, docs, new_scores, tags))
        part_of_commom_list = sorted(part_of_commom_list, key = lambda x: x[3], reverse = True) 

        qids, Q0s, docs, new_scores, tags = zip(*part_of_commom_list)

        ranks = range(1, len(new_scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, new_scores, tags))
        common_list = common_list + part_of_commom_list
    save_obj(scores_cam_list, "cam_scores")
    save_obj(scores_obj_list, "obj_scores")
        
    

    with open(output_dir + 'baseline_cam_obj.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche/')
    parser.add_argument('--o', default='/notebook/touche/output_tira/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)