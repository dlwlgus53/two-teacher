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
from collections import defaultdict


class DSTMultiWozAugData:
    def __init__(self,  tokenizer, data_path, data_type, short = 1):
        self.data_type = data_type
        self.short = short
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.raw_dataset = json.load(open(data_path , "r"))
        self.value_dict = self.make_value_dict(self.raw_dataset)
        turn_id, dial_id,  question, answer, = self.seperate_data(self.raw_dataset)
        assert len(turn_id) == len(dial_id) == len(question) == len(answer)
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question

    def __len__(self):
        return len(self.dial_id)
    
    def make_value_dict(self, dataset):
        values = defaultdict(list)
        for d_id in dataset.keys():
            dialogue = dataset[d_id]['log']
            for t_id, turn in enumerate(dialogue):
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    if key in turn['belief']: 
                        belief_answer = turn['belief'][key]
                        if isinstance(belief_answer, list) : belief_answer= belief_answer[0] # in muptiple type, a == ['sunday',6]
                    else:
                        belief_answer = ontology.QA['NOT_MENTIONED']
                    
                    values[key].append(belief_answer)
                    values[key] = list(set(values[key]))
        values['restaurant-time'].remove('11:30 | 12:30')
        values['hotel-pricerange'].remove('cheap|moderate')
        return values

    def get_data(self):
        return self.raw_dataset
    def aug_dst(self,dst):
        return [dst,dst,dst,dst]
    def seperate_data(self, dataset):
        question = []
        answer = []
        dial_id = []
        turn_id = []
        context = []
        S = 0
        for d_id in dataset.keys():
            S +=1
            if self.short == True and S > 10:
                break
            dialogue = dataset[d_id]['log']
            system = "" # should be changed
            for t_id, turn in enumerate(dialogue):
                dsts = self.aug_dst(turn['belief'])
                for dst in dsts:
                    q = f"make dialgoue. DST {dst} System : {system}"

                    question.append(q)
                    answer.append(turn['user']) # should be change
                    dial_id.append(d_id)
                    turn_id.append(t_id)

                system = turn['response'] 

        return turn_id, dial_id,  question,  answer


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
        
        return {"target": target,"turn_id" : turn_id,"question" : question,\
            "dial_id" : dial_id}
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """


        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        target = [x["target"] for x in batch]


        input_source = [f"{q}" for q in zip(question)]
        source = self.encode(input_source)
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        source = self.tokenizer.pad(source,padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length = self.max_length, \
        padding=True, return_tensors='pt', truncation = True)
        
        return {"input": source, "label": target,\
                "dial_id":dial_id, "turn_id":turn_id}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 't5-base')
    parser.add_argument('--labeled_data_path' , type = str, default='../woz-data/MultiWOZ_2.1/labeled/0.1/labeled_1.json')
    parser.add_argument('--base_trained', type = str, default = "t5-base", help =" pretrainned model from 🤗")


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = DSTMultiWozAugData(tokenizer,args.labeled_data_path, data_type = 'train', short = 1)

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['label']['input_ids'][i]))
            print()
        pdb.set_trace()
    