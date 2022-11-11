import pdb
import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import ontology
import random
from torch.nn.utils import rnn
import argparse
from transformers import T5Tokenizer
import config as cfg

class DSTMultiWozData:
    def __init__(self,  tokenizer, data_path, data_type):
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        raw_dataset = json.load(open(data_path , "r"))
        turn_id, dial_id,  question, schema, answer, dial_text = self.seperate_data(raw_dataset)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(schema) == len(answer) == len(dial_text)
            
        
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.dial_text = dial_text
        
    def __len__(self):
        return len(self.dial_id)

    def seperate_data(self, dataset):
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
    parser.add_argument('--data_path' ,  type = str, default='../woz-data/MultiWOZ_2.1/split0.01/train_data10.01.json')
    args = parser.parse_args()

    dataset = DSTMultiWozData(args.model_name, args.data_path, 'train')

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = dataset.tokenizer
    for batch in loader:
        for i in range(16):
            pdb.set_trace()
            print(t.decode(batch['input']['input_ids'][i]))
            print(t.decode(batch['label']['input_ids'][i]))
            print()
    