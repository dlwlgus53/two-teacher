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

class DSTMultiWozData:
    def __init__(self,  tokenizer, data_path, data_type, short = 0):
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.raw_dataset = json.load(open(data_path , "r"))
        self.short = short

        turn_id, dial_id,  question,  answer, dial_text = self.seperate_data(self.raw_dataset)

        assert len(turn_id) == len(dial_id) == len(question)\
            == len(answer) == len(dial_text)
        print(f'data number {data_type}, : {len(turn_id)}')        
        
        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.dial_text = dial_text

    def __len__(self):
        return len(self.dial_id)
    
    def get_data(self):
        return self.raw_dataset

    def seperate_data(self, dataset):
        question = []
        answer = []
        dial_id = []
        turn_id = []
        context = []
        dial_text = []
        S = 0
        for d_id in dataset.keys():
            S +=1
            if self.short == True and S > 300:
                break
            dialogue = dataset[d_id]['log']
            turn_text = ""
            for t_id, turn in enumerate(dialogue):
                turn_text += cfg.USER_tk
                turn_text += turn['user']
                q = "What is dialgoue state? "
                belief_state = str(turn['belief'])
                belief_state = belief_state.replace("'","").replace("{","").replace("}","")
                a= belief_state

                answer.append(a)
                question.append(q)
                dial_id.append(d_id)
                turn_id.append(t_id)
                dial_text.append(turn_text)
                
                turn_text += cfg.SYSTEM_tk
                turn_text += turn['response']

        return turn_id, dial_id, question,  answer, dial_text


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
        dial_text = self.dial_text[index]
        return {"target": target,"turn_id" : turn_id,"question" : question, "dial_text" : dial_text,\
            "dial_id" : dial_id,  }

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """

        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        target = [x["target"] for x in batch]
        dial_text = [x["dial_text"] for x in batch]


        input_source = [f"question: {q} context: {c}" for (q,c) in zip(question, dial_text, )]
        source = self.encode(input_source)
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        source = self.tokenizer.pad(source,padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length = self.max_length, \
        padding=True, return_tensors='pt', truncation = True)
        
        return {"input": source, "label": target,\
                 "dial_id":dial_id, "turn_id":turn_id}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 't5-small')
    parser.add_argument('--labeled_data_path' , type = str, default='../woz-data/MultiWOZ_2.1/labeled/0.1/labeled_1.json')
    parser.add_argument('--base_trained', type = str, default = "t5-base", help =" pretrainned model from 🤗")


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = DSTMultiWozData(tokenizer, args.labeled_data_path, 'train', short = 1) 


    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i], skip_special_tokens = True))
            print(t.decode(batch['label']['input_ids'][i], skip_special_tokens = True))
            print()
        pdb.set_trace()    