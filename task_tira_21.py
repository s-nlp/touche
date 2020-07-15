import requests
from xml.dom import minidom
import sys
import argparse

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient()

import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def make_scores_1(query, answer_titles):
    query_emb = bc.encode([query])
    scores = [cosine_similarity(query_emb.reshape(1, -1), bc.encode([cleanhtml(answer_title)]).reshape(1, -1))[0][0] for answer_title in answer_titles]
    return scores

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

def read_xml(filename):
    # convert file filename to list of tuples (number_of_topic, title_of_topic) 
    # input: filename string
    # output: list of corresponding tuples
    answer_list = []
    print (filename)
    xmldoc = minidom.parse(filename)
    itemlist = xmldoc.getElementsByTagName('topics')
    #print(len(itemlist))
    #print(itemlist)
    topic_list = itemlist[0].getElementsByTagName('topic')
    #print (len(topic_list))
    for topic in topic_list:
        tuple_for_add = tuple((topic.getElementsByTagName('number')[0].firstChild.nodeValue, topic.getElementsByTagName('title')[0].firstChild.nodeValue))
        answer_list.append(tuple_for_add)
    return answer_list

def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):
    list_of_tuples = read_xml(input_dir + input_file)
    common_list = []
    
    queries = []
        
    for elem in list_of_tuples:
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        queries.append(query)
        tag = 'myBaseline'
        response = make_a_search_request(query)
        a = response.json()
        try:
            scores = [elem['score'] for elem in a['results']]
            docs = [elem['trec_id'] for elem in a['results']]
            titles = [elem['title'] for elem in a['results']]
        except:
            return -8
            print (a)
        qids = qid*len(scores)
        Q0s = [Q0 for elem in scores]
        queries = query*len(scores)
        tags = [tag for elem in scores]
        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        print ("len of recieve part, len of common list", len(part_of_commom_list), len(common_list))
        common_list = common_list + part_of_commom_list
        
    print ("part 2")    
    for elem in list_of_tuples:
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'myBertSimilarity'
        response = make_a_search_request(query)
        a = response.json()
        try:
            docs = [elem['trec_id'] for elem in a['results']]
            titles = [elem['title'] for elem in a['results']]
            scores = make_scores_1(query, titles)
        except:
            print ("except")
            return titles
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
        common_list = common_list + part_of_commom_list

    with open(output_dir + 'run_example_q.txt', 'w') as fp:
        fp.write(queries)


if __name__ == "__main__":
    print ("dddd")
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche/')
    parser.add_argument('--o', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics-task-2-only-titles.xml')
    args = parser.parse_args()
    print ("dddd")
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)
    
    
    