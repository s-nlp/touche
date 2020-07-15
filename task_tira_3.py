import requests
import sys
import argparse

#numpy
import numpy as np

#torch
import torch, torch.nn as nn
import torch.nn.functional as F

#load fast ai model

from fastai.text.models import AWD_LSTM
from fastai.text.learner import get_language_model

awd_lstm_lm_config_custom = dict(emb_sz=768, n_hid=1152, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1, hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

m = get_language_model(arch=AWD_LSTM, vocab_sz = 60004, config = awd_lstm_lm_config_custom)
# state = torch.load('./wt103/fwd_wt103.h5')

from xml.dom import minidom

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient()

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

def make_a_search_request(query):
    # return json
    # json will be processed further
    params = {
            "apikey": "0833a307-97d3-462a-99d9-27db400c70da",
            "query": query,
            "index": ["cw12"],
            "size": 1000,
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

class over_AWD_LSTM(nn.Module):
    def __init__(self, extra_model, emb_size, ulm_fit_emb_size = 400):
        super(over_AWD_LSTM, self).__init__()
        self.modules = [module for module in extra_model.modules()]
        self.rnns = self.modules[0][0].rnns
        self.linear = nn.Linear(emb_size, ulm_fit_emb_size)
    def forward(self, input_tensor, from_embeddings = True):
        #print ("input_tensor shape 0 ", input_tensor.shape)
        #tsr = self.linear(input_tensor)
        #print ("input_tensor shape 1 ", tsr.shape)
        tsr1 = self.rnns[0](input_tensor)
        tsr2 = self.rnns[1](tsr1[0])
        tsr3 = self.rnns[2](tsr2[0]) # output, (h_n, c_n)
        return tsr3 # output, (h_n, c_n)

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient()

import re

def create_tensor(strng):
    #print ((clean_punct(cleanhtml(strng))).split())
    arr = bc.encode((clean_punct(cleanhtml(strng))).split())
    return torch.tensor(np.copy(arr))
    

def make_scores_lstm(my_rnn, query, answers, emb_size = 768, hidden_size = 64):
    print ((clean_punct(cleanhtml(query))).split())
    query_embs = bc.encode((clean_punct(cleanhtml(query))).split())
    print (query_embs)
    tensor_query_embs = torch.tensor(query_embs)
    print ("00")
    all_query_hidden, (last_query_hidden, last_query_state) = my_rnn(tensor_query_embs.unsqueeze(0))
    print ("010")
    #last_answers_hidden = [my_rnn(torch.tensor(bc.encode((clean_punct(cleanhtml(answer))).split())).unsqueeze(0))[1] for answer in answers]
    #last_answers_hidden = [my_rnn(create_tensor(answer).unsqueeze(0))[1][0] for answer in answers]
    last_answers_hidden = []
    for answer in answers:
        try:
            last_answers_hidden.append(my_rnn(create_tensor(answer).unsqueeze(0))[1][0])
            #print (len(last_answers_hidden))
            #print (my_rnn(create_tensor(answer).unsqueeze(0))[1][0].shape)
        except:
            print ("except")
            print (answer)
            print (create_tensor(answer).unsqueeze(0))
            print (my_rnn(create_tensor(answer).unsqueeze(0))[1][0])
    print ("02")
    aa = last_query_hidden.detach().numpy()
    #print ("aa shape", aa.shape)
    print ("03")
    bb = last_answers_hidden[0].detach().numpy()
    #print ("bb shape", bb.shape)
    print ("04")
    scores = [cosine_similarity(last_query_hidden.squeeze(0).detach().numpy(), last_answer_hidden.squeeze(0).detach().numpy())[0][0] for last_answer_hidden in last_answers_hidden]
    return scores
    
    
    # take title from seacrh output from clueweb
    # return a cosine similarity between bert embedding of sentences and article title
    
def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):
    list_of_tuples = read_xml(input_dir + input_file)
    common_list = []
    
    #config_class, model_class, tokenizer_class = BertConfig, BertModel, BertTokenizer
    #config = config_class.from_pretrained('bert-base-uncased')
    #config.output_attentions=True
    #model = model_class.from_pretrained('bert-base-uncased', config=config)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # load responses for all queries from file 
     # load responses for all queries from file 
    my_response_list = load_obj("list_of_un_answ")#create_list_of_unigue_answers(input_dir = input_dir, input_file = input_file)
    
    list_of_tuples = read_xml(input_dir + input_file)
    common_list = []
    
    awd_lstm_lm_config_custom = dict(emb_sz=768, n_hid=1152, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1, hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)
    m = get_language_model(arch=AWD_LSTM, vocab_sz = 60004, config = awd_lstm_lm_config_custom)
    
    my_rnn = over_AWD_LSTM(m, 768)
    print ("lstm ulm", flush = True)    
    for ind_q, elem in enumerate(list_of_tuples[:5]):
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'ULMFIT_LSTM'
        responses = list(zip(*my_response_list[str(ind_q + 1)]))

        scores0 = responses[0]
        print ("0")
        docs = responses[1]
        print ("1")
        titles = responses[2]
        print ("2")
        answers_bodies = responses[3]
        print ("3")
        
        scores = make_scores_lstm(my_rnn, query, answers_bodies)
        print ("4")
        qids = [qid]*len(scores)
        print ("5")
        Q0s = [Q0 for elem in scores]
        queries = query*len(scores)
        tags = [tag for elem in scores]
        part_of_commom_list = list(zip(qids, Q0s, docs, scores, tags))
        part_of_commom_list = sorted(part_of_commom_list, key = lambda x: x[3], reverse = True) 
        
        qids, Q0s, docs, scores, tags = zip(*part_of_commom_list)
        
        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        common_list = common_list + part_of_commom_list

    with open(output_dir + 'run.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche/')
    parser.add_argument('--o', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)