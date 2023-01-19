
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

all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class VerifyData:
    def __init__(self, tokenizer, data_path, data_type,  short = 0, use_list_path = None):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.raw_dataset = json.load(open(data_path , "r"))
        self.data_type = data_type
        self.short = short
        if use_list_path:
            use_list = json.load(open(use_list_path , "r"))
        else:
            use_list = None

        dial_id, turn_id, question, answer,belief, pseudo = self.seperate_data(self.raw_dataset, use_list)
        assert len(question) == len(answer)

        self.target = answer
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.belief = belief
        self.pseudo = pseudo

    def __len__(self):
        return len(self.question)

    def remove_unuse_domain(self,dst):
        new_dst = {}
        for key in dst:
            if key in ontology.all_domain:
                new_dst[key] = dst[key]
        return new_dst

    def make_bspn_nospecial(self,dict_bspn):
        ans =[] # un usual
        for domain_slot in ontology.all_domain:
            if domain_slot in dict_bspn:
                domain,slot = domain_slot.split("-")[0], domain_slot.split("-")[1]
                if ("[" + domain + ']') not in ans:
                    ans.append("[" + domain + ']')
                ans.append(slot)
                ans.append(dict_bspn[domain_slot])
        ans = ' '.join(ans)
        return ans 

    def make_value_dict(self, dataset):
        values = defaultdict(list)
        for dial in dataset:
            for t_id, turn in enumerate(dial):
                if 'belief' not in turn:
                        print(f"no belief {turn['dial_id']} {t_id}")
                        continue
                for key_idx, key in enumerate(ontology.all_domain): # TODO
                    
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

    def aug_dst(self,dst, value_dict, seed): # TODO 현재 turn의 belief state만 중심적으로 봐야한다.
        random.seed(seed)

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
        
        result = [add(dst.copy(), value_dict), delete(dst.copy()), replace(dst.copy(), value_dict)]
        result = [i for i in  result if i is not None]

        return result

    def seperate_data(self, dataset, use_list = None):
        value_dict = self.make_value_dict(dataset)
        question, pseudo = [], []
        answer = []
        dial_id = []
        turn_id = []
        belief = []
        dial_num = 0
        S = 0
        for dial in dataset:
            S +=1
            if self.short == True and S > 100:
                break
            turn_text = ""
            dial_num +=1
            d_id = dial[0]['dial_id']
            if 'belief' not in dial[0]:
                print(f"no belief {d_id}")
                continue
            for t_id, turn in enumerate(dial):
                turn_text += cfg.USER_tk
                turn_text += turn['user'].replace("<eos_u>","").replace("<sos_u>","")
                turn['belief'] =self.remove_unuse_domain(turn['belief'])
                
                belief_answer = self.make_bspn_nospecial(turn['belief'])

                q1 = f"verify the question and answer : context : {turn_text}, Answer : {belief_answer}"
                a1 = 'true'
                b1 = turn['belief']
                p = turn['pseudo']



                if self.data_type != 'label':
                    for neg_belief in self.aug_dst(turn['belief'], value_dict, t_id):
                        wrong_belief_answer = self.make_bspn_nospecial(neg_belief)
                        q2 = f"verify the question and answer : context : {turn_text}, Answer : {wrong_belief_answer}"
                        a2 = 'false'
                        b2  = neg_belief
                    
                        question.append(q2)
                        answer.append(a2)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                        belief.append(b2)
                        pseudo.append(p)

                        question.append(q1)
                        answer.append(a1)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                        belief.append(b1)
                        pseudo.append(p)
                else:
                    question.append(q1)
                    answer.append(a1)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                    belief.append(b1)
                    pseudo.append(p)

                turn_text += cfg.SYSTEM_tk
                turn_text += turn['resp'].replace("<eos_r>","").replace("<sos_r>","")
        
        print(f"total dial num is {dial_num}")
        return dial_id, turn_id, question, answer, belief, pseudo


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
        belief = self.belief[index]
        pseudo = self.pseudo[index]
        return {"question": question, "target": target, "turn_id" : turn_id,"dial_id" : dial_id, 'belief' : belief, 'pseudo' : pseudo}
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        input_source = [x["question"] for x in batch]
        target = [x["target"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        belief = [x["belief"] for x in batch]
        pseudo = [x["pseudo"] for x in batch]



        source = self.encode(input_source)
        source = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
        source = self.tokenizer.pad(source,padding=True)

        target = self.tokenizer.batch_encode_plus(target, max_length = self.max_length, \
        padding=True, return_tensors='pt', truncation = True)

        return {"input": source, "label": target, "turn_id" : turn_id,"dial_id" : dial_id, 'belief' : belief, 'pseudo' : pseudo}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 't5-small')
    parser.add_argument('--labeled_data_path' , type = str, default= '/home/jihyunlee/pptod/data/multiwoz/data/labeled/0.1/labeled_1.json')
    parser.add_argument('--test_data_path' , type = str, default= '/home/jihyunlee/pptod/data/multiwoz/data/multi-woz-fine-processed/multiwoz-fine-processed-test.json')
    
    parser.add_argument('--base_trained', type = str, default = "t5-small", help =" pretrainned model from 🤗")


    # /home/jihyunlee/woz-data/MultiWOZ_2.1/split0.01/labeled.json
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_trained)

    dataset = VerifyData(tokenizer, args.labeled_data_path, 'train', short = 1) 


    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
    t = dataset.tokenizer
    for batch in data_loader:
        for i in range(3):
            print(t.decode(batch['input']['input_ids'][i], skip_special_tokens = True) )
            print(t.decode(batch['label']['input_ids'][i]))
            print()
        pdb.set_trace()    
