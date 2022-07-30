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

from xml.dom import minidom
import re

#from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
#bc = BertClient()

import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def read_xml(filename):
    # convert file filename to list of tuples (number_of_topic, title_of_topic) 
    # input: filename string
    # output: list of corresponding tuples
    answer_list = []
    xmldoc = minidom.parse(filename)
    itemlist = xmldoc.getElementsByTagName('topics')
    print(len(itemlist))
    print(itemlist)
    topic_list = itemlist[0].getElementsByTagName('topic')
    print (len(topic_list))
    for topic in topic_list:
        tuple_for_add = tuple((topic.getElementsByTagName('number')[0].firstChild.nodeValue, topic.getElementsByTagName('title')[0].firstChild.nodeValue))
        answer_list.append(tuple_for_add)
    return answer_list

def make_a_search_request(query, size=1000):
    # return json
    # json will be processed further
    params = {
            "apikey": "0833a307-97d3-462a-99d9-27db400c70da",
            "query": query,
            "index": ["cw12"],
            "size": size,
            "pretty": True
        }
    response = requests.get(url = "http://www.chatnoir.eu/api/v1/_search", params = params)
    return response

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.lower()

def clean_punct(s):
    s = re.sub(r'[^\w\s]','',s)
    return s

def make_scores_1(query, answer_titles):
    query_emb = bc.encode([query])
    scores = [cosine_similarity(query_emb.reshape(1, -1), bc.encode([cleanhtml(answer_title)]).reshape(1, -1))[0][0] for answer_title in answer_titles]
    return scores
    
def make_request(query, size):
    response = make_a_search_request(query, size)
    try:
        getted_request = response.json()
    except:
        print ("exept0")
        #return getted_request

    n_request = 5
    while (n_request > 0 and 'results' not in getted_request):
        response = make_a_search_request(query, size)
        getted_request = response.json()
        n_request = n_request - 1
        print ("rerequest", query)
        print ("n_request", n_request)
    return getted_request

def request_more_unique_titles(query, request_size, set_of_existed_titles, number_of_needed_titles):
    number_of_try = 5
    additional_list = []
    while (number_of_try > 0 and len(additional_list) < number_of_needed_titles):
        number_of_try -= 1
        got_request = make_request(query, 10*request_size)
        if ('results' in got_request.keys()):
        #print ("len got_r", len(got_request['results']), "size of list", len(additional_list))
            scores0 = [elem['score'] for elem in got_request['results']]
            #print ("0")
            docs = [elem['trec_id'] for elem in got_request['results']]
            #print ("1")
            titles = [cleanhtml(elem['title']) for elem in got_request['results']]
            #print ("2", titles)
            answers_bodies = [cleanhtml(elem['snippet']) for elem in got_request['results']]
            for ind, title in enumerate(titles):
                #print ("title", title)
                #print ("set_of_existed_titles", set_of_existed_titles)
                #print (title not in set_of_existed_titles)
                if (title not in set_of_existed_titles):
                    set_of_existed_titles.add(title)
                    additional_list.append((scores0[ind], docs[ind], title, answers_bodies[ind]))
        print ("number of try ", number_of_try)
    return additional_list[:number_of_needed_titles]
    
def create_list_of_unigue_answers(input_dir = '/notebook/touche2021', input_file = 'topics-task-2-only-titles-2021.xml'):
    # return list 
    list_of_tuples = read_xml(input_dir + '/' + input_file)

    size = 100

    vocabulary_of_all_answers = {}

    for elem in list_of_tuples:
        set_of_titles = set()
        common_list = []
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        print ("qid ", qid, " query ", query)
        #tag = 'MethodAttention'
        getted_request = make_request(query, size)
        #print ("query", query)  
        if ('results' in getted_request.keys()):
            for answer_info in getted_request['results']:
                score = answer_info['score']
                trec_id = answer_info['trec_id']
                title = cleanhtml(answer_info['title'])
                answer_bodies = cleanhtml(answer_info['snippet'])
                if title not in set_of_titles:
                    set_of_titles.add(title)
                    common_list.append((score, trec_id, title, answer_bodies))
        #print ("common_list_before", common_list)
        if (len(common_list) < size):
            print ("len(common_list), size", len(common_list), size)
            add_list = request_more_unique_titles(query, size, set_of_titles, size - len(common_list))
            #print ("add liat!!", add_list)
        else:
            print ("else")
        #print ("common_list size", len(common_list), "add list size", len(add_list))
        print ("answers size", len(common_list + add_list))
        #print("\n")
        print ("qid ", qid, query)
        vocabulary_of_all_answers[qid] = common_list + add_list
    
    file_to_write = open("list_of_un_answ_2020.pcl", "wb")
    pickle.dump(vocabulary_of_all_answers, file_to_write)
    return vocabulary_of_all_answers

import pytorch_transformers
from pytorch_transformers import *

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

import pickle

with open('/notebook/touche/list_of_un_answ.pkl', 'rb') as f:
        my_response_list = pickle.load(f)

def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics.xml'):
    
    # load responses for all queries from file 
    #my_response_list = create_list_of_unigue_answers(input_dir = input_dir, input_file = input_file)
    
    print ("len my responses list", len(my_response_list))
    
    print ("input_dir + '/' + input_file", input_dir + input_file)
    list_of_tuples = read_xml(input_dir + input_file)
    common_list = []

    print ("baseline", flush = True)
    for ind_q, elem in enumerate(list_of_tuples):
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'MyBaselineFilterResponse'
        responses = list(zip(*my_response_list[str(ind_q + 1)]))
        
        print ("responses", responses)
        
        scores0 = responses[0]
        print ("ind_q !!", ind_q)
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
        print ("len(common_list)", len(common_list))

    with open('./baseline.qrels', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--inp_dir', default='/notebook/touche/')
    parser.add_argument('--out_dir', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.out_dir, input_dir = args.inp_dir, input_file = args.inp_file)
    