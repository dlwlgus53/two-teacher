
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
    def __init__(self, tokenizer, data_path, data_type,  short = 0, use_list_path = None):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.raw_dataset = json.load(open(data_path , "r"))
        self.value_list = self.make_value_list(self.raw_dataset)
        self.data_type = data_type
        self.short = short
        if use_list_path:
            use_list = json.load(open(use_list_path , "r"))
        else:
            use_list = None

        dial_id, turn_id, schema, question, answer = self.seperate_data(self.raw_dataset, use_list)
        assert len(question) == len(answer)

        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema

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


    def seperate_data(self, dataset, use_list = None):
        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        dial_num = 0
        S = 0
        for d_id in dataset.keys():
            S +=1
            if self.short == True and S > 100:
                break
            dialogue = dataset[d_id]['log']
            turn_text = ""
            if use_list and d_id.lower() not in use_list: # use list 가 있는데 현재 대화는 거기 없음
                continue
            dial_num +=1
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
        
        print(f"total dial num is {dial_num}")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 't5-base')
    parser.add_argument('--labeled_data_path' , type = str, default='../woz-data/MultiWOZ_2.1/labeled/0.1/labeled_1.json')
    parser.add_argument('--base_trained', type = str, default = "t5-base", help =" pretrainned model from 🤗")


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = VerifyData(tokenizer, args.labeled_data_path, 'train', short = 1) 


    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['label']['input_ids'][i]))
            print()
        pdb.set_trace()    
