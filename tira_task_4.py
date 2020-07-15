import requests
#r = requests.post('http://10.30.99.211:8261/gpt_small', data = "What is better for deep learning Python or Matlab?")
#print (r.status_code)

# numpy
import numpy as np
import argparse

# torch
import torch, torch.nn as nn
import torch.nn.functional as F

# transformers
import pytorch_transformers
from pytorch_transformers import *

# read file
from xml.dom import minidom
import re

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

def make_a_search_request(query):
    # return json
    # json will be processed further
    params = {
            "apikey": "0833a307-97d3-462a-99d9-27db400c70da",
            "query": query,
            "index": ["cw12"],
            "size": 10,
            "pretty": True
        }
    response = requests.get(url = "http://www.chatnoir.eu/api/v1/_search", params = params)
    return response

def clean_punct(s):
    s = re.sub(r'[^\w\s]','',s)
    return s

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.lower()

def make_scores_transformers(query, answers_bodies, model, tokenizer):
    scores = []
    for ind, elem in enumerate(answers_bodies):
        sentences = ["[CLS] " + query + " [SEP] " + answers_bodies[ind] + " [SEP]"]
        #print(sentences[0])

        # Tokenize with BERT tokenizer
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
        score = np.sum(attensions[cls_idx:sep_idx, sep_idx:]) + np.sum(attensions[sep_idx:end_idx, cls_idx:sep_idx+2])
        #print (score)
        scores.append(score)
    return scores
    


def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):
    
    config_class, model_class, tokenizer_class = BertConfig, BertModel, BertTokenizer
    config = config_class.from_pretrained('bert-base-uncased')
    config.output_attentions=True
    model = model_class.from_pretrained('bert-base-uncased', config=config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    list_of_tuples = read_xml(input_dir + "/" + input_file)
    common_list = []
    
    with open(output_dir + 'run_example.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        
    print ("attention")    
    for elem in list_of_tuples[:5]:
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'MethodAttention'
        response = make_a_search_request(query)
        try:
            getted_request = response.json()
        except:
            print ("exept0")
            #return getted_request

        n_request = 100
        while (n_request > 0 and 'results' not in getted_request):
            response = make_a_search_request(query)
            getted_request = response.json()
            n_request == n_request - 1
            print ("rerequest", query)

        try:
            scores0 = [elem['score'] for elem in getted_request['results']]
            #print ("0")
            docs = [elem['trec_id'] for elem in getted_request['results']]
            #print ("1")
            titles = [elem['title'] for elem in getted_request['results']]
            #print ("2")
            answers_bodies = [cleanhtml(elem['snippet']) for elem in getted_request['results']]
            #print ("3")
            # print (scores0, scores3, scores)
        except:
            print ("exept1")
            #return getted_request
        scores = make_scores_transformers(query, answers_bodies, model, tokenizer)
        qids = qid*len(scores)
        Q0s = [Q0 for elem in scores]
        queries = query*len(scores)
        tags = [tag for elem in scores]
        part_of_commom_list = list(zip(qids, Q0s, docs, scores, tags))
        part_of_commom_list = sorted(part_of_commom_list, key = lambda x: x[3], reverse = True) 

        qids, Q0s, docs, scores, tags = zip(*part_of_commom_list)

        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        common_list = common_list + part_of_commom_list

    with open(output_dir + 'run_example4.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche/')
    parser.add_argument('--o', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)