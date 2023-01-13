import copy
import pdb
import sys
import torch
import random
import numpy as np
import json
import progressbar
import ontology
import random
import argparse
from transformers import  AutoTokenizer
import config as cfg
from utils import make_label_key, dictionary_split

class VerifyData:
    def __init__(self, tokenizer, raw_dataset, data_type):

        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.raw_dataset = raw_dataset
        self.value_list = self.make_value_list(raw_dataset)
        self.data_type = data_type

        dial_id, turn_id, schema, question, answer = self.seperate_data(raw_dataset)
        assert len(question) == len(answer)

        # 원본 보호가 되는가?
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema

        
        self.o_target = copy.deepcopy(self.target) 
        self.o_question = copy.deepcopy(self.question) 
        self.o_turn_id = copy.deepcopy(self.turn_id)
        self.o_dial_id = copy.deepcopy(self.dial_id)
        self.o_schema = copy.deepcopy(self.schema)

    def reverse_update(self):
        self.target = copy.deepcopy(self.o_target)
        self.question = copy.deepcopy(self.o_question)
        self.turn_id = copy.deepcopy(self.o_turn_id)
        self.dial_id = copy.deepcopy(self.o_dial_id)
        self.schema = copy.deepcopy(self.o_schema)

    def update(self, data_path, pseudo_label):
        dataset = json.load(open(data_path , "r"))
        
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            turn_text = ""
            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO

                    if key in turn['belief']: 
                        belief_answer = turn['belief'][key]
                        if isinstance(belief_answer, list) : belief_answer= belief_answer[0] # in muptiple type, a == ['sunday',6]
                    else:
                        belief_answer = ontology.QA['NOT_MENTIONED']

                    q1 = f"verify QA context : {turn_text}, question : {ontology.QA[key]['description1']},Answer : {belief_answer}"
                    a1 = 'true'

                    q2 = f"verify QA context : {turn_text}, question : {ontology.QA[key]['description1']}, Answer : {self.find_different_answer(belief_answer)}"
                    a2 = 'false'
                    self.question.append(q1) 
                    self.target.append(a1)
                    self.schema.append(key)
                    self.dial_id.append(d_id)
                    self.turn_id.append(t_id)

                    if self.data_type != 'test':
                        self.question.append(q2)
                        self.target.append(a2)
                        self.schema.append(key)
                        self.dial_id.append(d_id)
                        self.turn_id.append(t_id)


                turn_text += cfg.SYSTEM_tk
                turn_text += turn['response']





    def __len__(self):
        return len(self.question)

    def make_value_list(self, dataset):
        values = []
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            for t_id, turn in enumerate(dialogue):
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    if key in turn['belief']: 
                        belief_answer = turn['belief'][key]
                        if isinstance(belief_answer, list) : belief_answer= belief_answer[0] # in muptiple type, a == ['sunday',6]
                    else:
                        belief_answer = ontology.QA['NOT_MENTIONED']

                    values.append(belief_answer)
        
        return list(set(values))
                    


    def find_different_answer(self, target):
        return random.choice(list(set(self.value_list) - set([target])))


    def seperate_data(self, dataset):
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        K =0 
        for d_id in dataset.keys():
            # K +=1
            # if K>10:
                # break
            dialogue = dataset[d_id]['log']
            turn_text = ""
            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO

                    if key in turn['belief']: 
                        belief_answer = turn['belief'][key]
                        if isinstance(belief_answer, list) : belief_answer= belief_answer[0] # in muptiple type, a == ['sunday',6]
                    else:
                        belief_answer = ontology.QA['NOT_MENTIONED']


                    q1 = f"verify QA context : {turn_text}, question : {ontology.QA[key]['description1']},Answer : {belief_answer}"
                    a1 = 'true'

                    q2 = f"verify QA context : {turn_text}, question : {ontology.QA[key]['description1']}, Answer : {self.find_different_answer(belief_answer)}"
                    a2 = 'false'
                    question.append(q1)
                    answer.append(a1)
                    schema.append(key)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                    if self.data_type != 'test':
                        question.append(q2)
                        answer.append(a2)
                        schema.append(key)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                turn_text += cfg.SYSTEM_tk
                turn_text += turn['response']

        return dial_id, turn_id, schema, question, answer 


    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized)> self.max_length:
                    idx = [m.start() for m in re.finditer(cfg.USER_tk, text)]
                    text = text[:idx[0]] + text[idx[1]:] # delete one turn
                else:
                    break
                
            examples.append(tokenized)
        return examples


    def __getitem__(self, index):
        target = self.target[index]
        question = self.question[index]
        turn_id = self.turn_id[index]
        dial_id = self.dial_id[index]
        schema = self.schema[index]
        return {"question": question, "target": target, "turn_id" : turn_id,"dial_id" : dial_id, "schema":schema }
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        input_source = [x["question"] for x in batch]
        target = [x["target"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        schema = [x["schema"] for x in batch]


        source = self.encode(input_source)
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        source = self.tokenizer.pad(source,padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length = self.max_length, \
        padding=True, return_tensors='pt', truncation = True)

        return {"input": source, "label": target, "turn_id" : turn_id,"dial_id" : dial_id, "schema":schema }


class DSTMultiWozData:
    def __init__(self,  tokenizer, data_path, data_type):
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length

        self.raw_dataset = json.load(open(data_path , "r"))

        turn_id, dial_id,  question, schema, answer, dial_text = self.seperate_data(self.raw_dataset)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer) == len(dial_text)
            
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.dial_text = dial_text

        self.o_target = copy.deepcopy(self.target)
        self.o_turn_id = copy.deepcopy(self.turn_id) 
        self.o_dial_id = copy.deepcopy(self.dial_id) 
        self.o_question = copy.deepcopy(self.question) 
        self.o_schema = copy.deepcopy(self.schema) 
        self.o_dial_text = copy.deepcopy(self.dial_text) 

    def __len__(self):
        return len(self.dial_id)

    
    def reverse_update(self):
        self.target = copy.deepcopy(self.o_target)
        self.question = copy.deepcopy(self.o_question)
        self.turn_id = copy.deepcopy(self.o_turn_id)
        self.dial_id = copy.deepcopy(self.o_dial_id)
        self.schema = copy.deepcopy(self.o_schema)
        self.dial_text = copy.deepcopy(self.o_dial_text)

    def update(self, data_path, pseudo_label):

        dataset = json.load(open(data_path , "r"))
        
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            turn_text = ""

            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    q = ontology.QA[key]['description1']
                    label_key = make_label_key(d_id, t_id, key)
                    if label_key in pseudo_label:

                        a = pseudo_label[label_key]

                        self.schema.append(key)
                        self.target.append(a)
                        self.question.append(q)
                        self.dial_id.append(d_id)
                        self.turn_id.append(t_id)
                        self.dial_text.append(turn_text)

                turn_text += cfg.SYSTEM_tk
                turn_text += turn['response']


    def get_data(self):
        return self.raw_dataset

    def seperate_data(self, dataset):
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        context = []
        dial_text = []
        
        for d_id in dataset:
            d_id = dial[0]['dial_id']
            dialogue = dataset[d_id]['log']
            turn_text = ""

            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    q = ontology.QA[key]['description1']
                    label_key = make_label_key(d_id, t_id, key)

                    if key in turn['belief']: 
                        a = turn['belief'][key]
                        if isinstance(a, list) : a= a[0] # in muptiple type, a == ['sunday',6]
                    else:
                        a = ontology.QA['NOT_MENTIONED']

                    schema.append(key)
                    answer.append(a)
                    question.append(q)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                    dial_text.append(turn_text)

                turn_text += cfg.SYSTEM_tk
                turn_text += turn['response']

        return turn_id, dial_id,  question, schema, answer, dial_text


    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized)> self.max_length:
                    idx = [m.start() for m in re.finditer(cfg.USER_tk, text)]
                    text = text[:idx[0]] + text[idx[1]:] # delete one turn
                else:
                    break
                
            examples.append(tokenized)
        return examples


    def __getitem__(self, index):
        target = self.target[index]
        turn_id = self.turn_id[index]
        dial_id = self.dial_id[index]
        question = self.question[index]
        schema = self.schema[index]
        dial_text = self.dial_text[index]
        
        return {"target": target,"turn_id" : turn_id,"question" : question, "dial_text" : dial_text,\
            "dial_id" : dial_id, "schema":schema }
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """


        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        target = [x["target"] for x in batch]
        dial_text = [x["dial_text"] for x in batch]


        input_source = [f"question: {q} context: {c}" for (q,c) in zip(question, dial_text, )]
        source = self.encode(input_source)
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        source = self.tokenizer.pad(source,padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length = self.max_length, \
        padding=True, return_tensors='pt', truncation = True)
        
        return {"input": source, "label": target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 't5-small')
    parser.add_argument('--all_train_data_path' , type = str, default='../woz-data/MultiWOZ_2.1/split0.01/train_data10.01.json')
    parser.add_argument('--labeled_data_path' , type = str, default='/home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_1.json')
    parser.add_argument('--base_trained', type = str, default = "t5-small", help =" pretrainned model from 🤗")


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = DSTMultiWozData(tokenizer,args.all_train_data_path, data_type = 'train', labeled_data_path = args.labeled_data_path)

    labeled_data = dataset.get_labeled_data()
    train_LD, dev_LD = dictionary_split(labeled_data)

    verify_train_dataset = VerifyData(tokenizer, args.labeled_data_path, train_LD)
    verify_dev_dataset = VerifyData(tokenizer, args.labeled_data_path, dev_LD)

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    verify_loader = torch.utils.data.DataLoader(dataset=verify_train_dataset, batch_size=16, collate_fn=verify_train_dataset.collate_fn)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['label']['input_ids'][i]))
            print()
        break
    
    for batch in verify_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['label']['input_ids'][i]))
            print()
        break
    