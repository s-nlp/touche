import os
import os.path
import sys
import spacy


sys.path.append('/cqas')
from src.factories.factory_tagger import TaggerFactory
from src.layers import layer_context_word_embeddings_bert


current_directory_path = os.path.dirname(os.path.realpath(__file__))

def create_sequence_from_sentence(str_sentences):
    return [str_sentence.lower().split() for str_sentence in str_sentences]

class extractor:
    def __init__(self, model_name = 'bertttt.hdf5', model_path = current_directory_path + '/external_pretrained_models/'):
        self.answ = "UNKNOWN ERROR"
        self.model_name = model_name
        self.model_path = model_path
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.aspects = ''
        try:
            self.model = TaggerFactory.load(self.model_path + self.model_name, 2)
            print ("extract_objects_predicates gpu", self.model.gpu)
            self.model.cuda(device=2)
            self.model.gpu = 2
        except:
            raise RuntimeError("Can't map to gpu. Maybe it is OOM")
        
        
    def from_string(self, input_sentence):
        self.input_str = input_sentence
        
    def get_objects_predicates(self, list_words, list_tags):
        obj_list = []
        pred_list = []
        asp_list = []
        for ind, elem in enumerate(list_tags):
            if elem == 'B-OBJ':
                obj_list.append(list_words[ind])
            if elem == 'B-PREDFULL':
                pred_list.append(list_words[ind])    
            if elem == 'B-ASP':
                asp_list.append(list_words[ind])
        return obj_list, pred_list, asp_list
    
    def extract_objects_predicates(self, input_sentence):
        words = create_sequence_from_sentence([input_sentence])   
        print ("words ", words)
        tags = self.model.predict_tags_from_words(words)
        print ("extract_objects_predicates tags", tags)
        objects, predicates, aspects = self.get_objects_predicates(words[0], tags[0])
        self.predicates = predicates
        self.aspects = aspects
        if len(objects) >= 2:
            self.first_object = objects[0]
            self.second_object = objects[1]
        else: # try to use spacy
            
            #print("We try to use spacy")
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(input_sentence)
            tokens = [token.text for token in doc]
            split_sent = words[0]
            #print ("split_sent", split_sent)
            #print ("tokens ", tokens)
            if 'or' in split_sent:
                comp_elem = 'or'
            elif 'vs' in split_sent:
                comp_elem = 'vs'
            elif 'vs.' in split_sent:
                comp_elem = 'vs.'
            elif 'and' in split_sent and 'between' in split_sent:
                comp_elem = 'and'
            else:
                self.answ = "We can't recognize two objects for compare"  
                return;
    
            if (comp_elem in tokens):
                or_index = tokens.index(comp_elem)
                if (len (doc.ents) >= 2):
                    for ent in doc.ents:
                        #print ("or index doc snet", or_index)
                        #print ("begin end ", ent.start, ent.end, ent.text)
                        if (ent.end == or_index):
                            #print ("obj1 spacy doc sent", ent.text)
                            self.first_object = ent.text
                        if (ent.start == or_index + 1):
                            #print ("obj2 spacy doc sent", ent.text)
                            self.second_object = ent.text

                else:
                    print ("or simple split_sent", or_index)
                    try:
                        obj1 = tokens[or_index - 1]
                        obj2 = tokens[or_index + 1]
                        #print (obj1, obj2)
                        self.first_object = obj1
                        self.second_object = obj2
                    except:
                        self.answ = "We can't recognize two objects for compare" 

            else:
                self.answ = "We can't recognize two objects for compare" 
                
    def get_params(self):
        print ("in extractor get params 0")
        try:
            self.extract_objects_predicates(self.input_str)
        except:
            raise RuntimeError("Can't map to gpu. Maybe it is OOM")
        return self.first_object.strip(".,!/?"), self.second_object.strip(".,!/?"), self.predicates, self.aspects
    
    def clear_params(self):
        self.first_object = ''
        self.second_object = ''
        self.predicates = ''
        self.aspects = ''