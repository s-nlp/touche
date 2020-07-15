import requests
from xml.dom import minidom
import sys
import argparse

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

def read_xml(filename):
    # convert file filename to list of tuples (number_of_topic, title_of_topic) 
    # input: filename string
    # output: list of corresponding tuples
    answer_list = []
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
    for elem in list_of_tuples:
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'myBaseline'
        print ("query", query)
        response = make_a_search_request(query)
        a = response.json()
        try:
            scores = [elem['score'] for elem in a['results']]
            docs = [elem['trec_id'] for elem in a['results']]
            titles = [elem['title'] for elem in a['results']]
        except:
            print (a)
        qids = qid*len(scores)
        Q0s = [Q0 for elem in scores]
        queries = query*len(scores)
        tags = [tag for elem in scores]
        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        #print ("len of recieve part, len of common list", len(part_of_commom_list), len(common_list))
        common_list = common_list + part_of_commom_list

    with open(output_dir + 'run.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--inp_dir', default='/notebook/touche/')
    parser.add_argument('--out_dir', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.out_dir, input_dir = args.inp_dir, input_file = args.inp_file)
    
    
    