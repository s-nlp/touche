from xml.dom import minidom
import re
import requests

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

def clean_punct(s):
    s = re.sub(r'[^\w\s]','',s)
    return s

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?\>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.lower()

def make_request(query, size):
    response = make_a_search_request(query, size)
    try:
        getted_request = response.json()
    except:
        print ("exept0")
        #return getted_request

    n_request = 10
    while (n_request > 0 and 'results' not in getted_request):
        print (getted_request)
        response = make_a_search_request(query, size)
        getted_request = response.json()
        n_request = n_request - 1
        print ("rerequest", query)
    if ('results' in getted_request):
        return getted_request
    else:
        return -42
    
def request_more_unique_titles(query, request_size, set_of_existed_titles, number_of_needed_titles):
    number_of_try = 10
    additional_list = []
    while (number_of_try > 0 and len(additional_list) < number_of_needed_titles):
        got_request = make_request(query, 1000)
        number_of_try = number_of_try - 1
        #print ("len got_r", len(got_request['results']), "size of list", len(additional_list))
        if (got_request != 42):
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
        else:
            print ("in request_more_unique_titles len(additional_list)", len(additional_list))
            return additional_list[:number_of_needed_titles]
        print ("in request_more_unique_titles len(additional_list)", len(additional_list))
    return additional_list[:number_of_needed_titles]
    
def create_list_of_unigue_answers(input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml', list_of_tuples = None):
    # return list 
    if (list_of_tuples == None):
        list_of_tuples = read_xml(input_dir + input_file)

    size = 1000

    vocabulary_of_all_answers = {}
    try:
        for elem in list_of_tuples:
            print (len(vocabulary_of_all_answers))
            set_of_titles = set()
            common_list = []
            common_list_reserve = []
            qid = elem[0]
            Q0 = 'Q0'
            query = elem[1]
            #tag = 'MethodAttention'
            if (int(qid) > 63):
                getted_request = make_request(query, size)
                for answer_info in getted_request['results']:
                    score = answer_info['score']
                    trec_id = answer_info['trec_id']
                    title = cleanhtml(answer_info['title'])
                    answer_bodies = cleanhtml(answer_info['snippet'])
                    if title not in set_of_titles:
                        set_of_titles.add(title)
                        common_list.append((score, trec_id, title, answer_bodies))
                    else:
                        common_list_reserve.append((score, trec_id, title, answer_bodies))
                #print ("common_list_before", common_list)
                #print ("common_list size", len(common_list), "add list size", len(add_list))
                #print ("answer size", len(common_list + add_list))
                #print("\n")
                if (len(common_list) > 0):
                    vocabulary_of_all_answers[qid] = common_list
    except:
        return vocabulary_of_all_answers
    return vocabulary_of_all_answers