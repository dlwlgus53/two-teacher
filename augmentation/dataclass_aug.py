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
        if data_type == 'aug':
            turn_id, dial_id,  question, answer, = self.seperate_data_aug(self.raw_dataset)
            assert len(turn_id) == len(dial_id) == len(question) == len(answer)
        else: 
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
        for dial in dataset:

            for t_id, turn in enumerate(dial):
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    if key in turn['belief']: 
                        belief_answer = turn['belief'][key]
                        if isinstance(belief_answer, list) : belief_answer= belief_answer[0] # in muptiple type, a == ['sunday',6]
                        values[key].append(belief_answer)
                        values[key] = list(set(values[key]))

        try:
            values['restaurant-time'].remove('11:30 | 12:30')
        except ValueError:
            pass
        try:
            values['hotel-pricerange'].remove('cheap|moderate')
        except ValueError:
            pass
        return values

    def get_data(self):
        return self.raw_dataset


    def aug_dst(self,dst): # TODO 현재 turn의 belief state만 중심적으로 봐야한다.

        value_dict = self.value_dict
        def add(dst, value_dict):
            try:
                domain = random.choice(list(dst.keys())).split("-")[0]
                slot = random.choice([item for item in value_dict.keys() if item.startswith(domain) and item not in dst.keys()])
                value = random.choice(value_dict[slot])
                dst[slot] = value
            except IndexError as e:
                dst = dst

            return dst

        def delete(dst):
            try:
                slot = random.choice(list(dst.keys()))
                del dst[slot]
            except IndexError as e:
                dst = None
            return dst

        def replace(dst, value_dict):
            try:
                slot = random.choice(list(dst.keys()))
                choices = value_dict[slot].copy()
                if dst[slot] == '11:30 | 12:30' or dst[slot] == 'cheap|moderate':
                    pass
                else:
                    choices.remove(dst[slot])
                
                value = random.choice(choices)
                dst[slot] = value
            except IndexError as e:
                dst = None
            except:
                pdb.set_trace()
            return dst
        
        result = [dst, add(dst.copy(), value_dict), delete(dst.copy()), replace(dst.copy(), value_dict)]
        result = [i for i in  result if i is not None]

        return result
    def remove_unuse_domain(self,dst):
        new_dst = {}
        for key in dst:
            if key in ontology.QA['all-domain']:
                new_dst[key] = dst[key]
        return new_dst
    def seperate_data(self, dataset):
        question = []
        answer = []
        dial_id = []
        turn_id = []
        context = []
        dial_num = 0
        S = 0
        for dial in dataset:
            S +=1
            if self.short == True and S > 1000:
                break
            d_id = dial[0]['dial_id']
            system = "" # should be changed
            dial_num +=1
            for t_id, turn in enumerate(dial):
                turn['curr_belief'] =self.remove_unuse_domain(turn['curr_belief'])
                if len(turn['curr_belief']) == 0 :continue
                curr_dst = turn['curr_belief']
                curr_dst_str = str(curr_dst)
                curr_dst_str = curr_dst_str.replace("{","").replace("}","").replace(": ", ' is ').replace("'","")
                q = f"make dialgoue. DST : {curr_dst_str} System : {system}"

                question.append(q)
                answer.append(turn['user'].replace("<eos_u>","").replace("<sos_u>","") ) # should be change
                dial_id.append(d_id)
                turn_id.append(t_id)

                system = turn['resp'].replace("<eos_r>","").replace("<sos_r>","") 
                prev_dst = turn['belief']

        print(f"total dial num is {dial_num}")
        return turn_id, dial_id,  question,  answer


    def seperate_data_aug(self, dataset):
        question = []
        answer = []
        dial_id = []
        turn_id = []
        context = []
        dial_num  = 0
        S = 0
        for dial in dataset:
            S +=1
            if self.short == True and S > 300:
                break
            d_id = dial[0]['dial_id']
            system = ""
            dial_num +=1

            for t_id, turn in enumerate(dial):

                turn['curr_belief'] =self.remove_unuse_domain(turn['curr_belief'])
                curr_dst = turn['curr_belief']
                curr_dsts = self.aug_dst(curr_dst)

                for curr_dst_aug in curr_dsts:
                    curr_dst_aug_str = str(curr_dst_aug)
                    curr_dst_aug_str = curr_dst_aug_str.replace("{","").replace("}","").replace(": ", ' is ').replace("'","")
                    q = f"make dialgoue. DST : {curr_dst_aug_str} System : {system}"


                    question.append(q)
                    answer.append("_") # should be change
                    dial_id.append(d_id)
                    turn_id.append(t_id)

                system = turn['resp'].replace("<eos_r>","").replace("<sos_r>","") 
                prev_dst = turn['belief']

        print(f"total dial num is {dial_num}")
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

        input_source = question
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
    parser.add_argument('--base_trained', type = str, default = "t5-base", help =" pretrainned model from 🤗")
    parser.add_argument('--labeled_data_path' , type = str, default= '/home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_1.json')
    parser.add_argument('--test_data_path' , type = str, default= '/home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json')


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = DSTMultiWozAugData(tokenizer,args.labeled_data_path, data_type = 'train', short = 1)
    dataset = DSTMultiWozAugData(tokenizer,args.labeled_data_path, data_type = 'train', short = 1)

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i], skip_special_tokens = True))
            print(t.decode(batch['label']['input_ids'][i], skip_special_tokens = True))
            print()
        pdb.set_trace()
    