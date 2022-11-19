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
from transformers import T5Tokenizer
import config as cfg
from utils import make_label_key, dictionary_split

class VerifyData:
    def __init__(self, tokenizer, raw_path, labeled_data):

        self.labeled_data = labeled_data
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        raw_dataset = json.load(open(raw_path , "r"))
        question, answer = self.seperate_data(raw_dataset, self.labeled_data)
        assert len(question) == len(answer)

        self.target = answer
        self.question = question

    def __len__(self):
        return len(self.question)

    def set_labeled_data(labeled_data):
        self.labeled_data = labeled_data
        


    def find_different_answer(self, target):
        return random.choice(list(set(self.labeled_data.values()) - set([target])))


    def seperate_data(self, dataset, labeled_data):
        question = []
        answer = []

        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            turn_text = ""
            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    label_key = make_label_key(d_id, t_id, key)
                    if label_key in self.labeled_data:
                        
                        
                        q1 = f"verify QA context : {turn_text}, question : {ontology.QA[key]['description1']}, \
                        Answer : {labeled_data[label_key]}"
                        a1 = 'true'

                        q2 = f"verify QA context : {turn_text}, question : {ontology.QA[key]['description1']}, \
                        Answer : {self.find_different_answer(labeled_data[label_key])}"
                        a2 = 'false'


                        question.append(q1)
                        question.append(q2)

                        answer.append(a1)
                        answer.append(a2)

                turn_text += cfg.SYSTEM_tk
                turn_text += turn['response']

        return question, answer


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
        return {"target": target, "question" : question}
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        input_source = [x["question"] for x in batch]
        target = [x["target"] for x in batch]

        source = self.encode(input_source)
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        source = self.tokenizer.pad(source,padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length = self.max_length, \
        padding=True, return_tensors='pt', truncation = True)
        return {"input": source, "label": target}


class DSTMultiWozData:
    def __init__(self,  tokenizer, data_path, data_type, labeled_data_path = None):
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.labeled_data = None

        if data_type == 'train':
            labeled_dataset = json.load(open(labeled_data_path , "r"))
            self.labeled_data = self.init_make_labeled_data(labeled_dataset)

        raw_dataset = json.load(open(data_path , "r"))

        turn_id, dial_id,  question, schema, answer, dial_text = self.seperate_data(raw_dataset, self.labeled_data)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer) == len(dial_text)
            
        
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.dial_text = dial_text

    def add_to_labeled(self, new_data):
        dataset = new_data
        for key, value in dataset.items():
            self.labeled_data[key] = value

    def get_labeled_data(self):
        return self.labeled_data

    def init_make_labeled_data(self, labeled_dataset):
        labeled_data = {}
        dataset = labeled_dataset
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            for t_id, turn in enumerate(dialogue):
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    label_key = make_label_key(d_id, t_id, key)
                    if key in turn['belief']: 
                        label_value = turn['belief'][key]
                        if isinstance(label_value, list) : label_value= label_value[0] # in muptiple type, a == ['sunday',6]
                    else:
                        label_value = ontology.QA['NOT_MENTIONED']
                    labeled_data[label_key] = label_value
        
        return labeled_data



    def __len__(self):
        return len(self.dial_id)

    def seperate_data(self, dataset, labeled_data):
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        context = []
        dial_text = []
        
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            turn_text = ""

            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    q = ontology.QA[key]['description1']
                    label_key = make_label_key(d_id, t_id, key)
                    if self.data_type == 'train' and label_key in labeled_data:
                        a = labeled_data[label_key]
                    else:
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
    parser.add_argument('--train_data_path' , type = str, default='../woz-data/MultiWOZ_2.1/split0.01/labeled.json')
    parser.add_argument('--base_trained', type = str, default = "t5-small", help =" pretrainned model from 🤗")


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)

    dataset = DSTMultiWozData(tokenizer,args.all_train_data_path, data_type = 'train', labeled_data_path = args.train_data_path)

    labeled_data = dataset.get_labeled_data()
    train_LD, dev_LD = dictionary_split(labeled_data)

    verify_train_dataset = VerifyData(tokenizer, args.train_data_path, train_LD)
    verify_dev_dataset = VerifyData(tokenizer, args.train_data_path, dev_LD)

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
    