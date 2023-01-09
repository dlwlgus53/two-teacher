import random
import ontology
import json, os
import pdb

'''
def make_label_key(dial_id, turn_id, slot):
    return f'[d]{dial_id}[t]{turn_id}[s]{slot}'
'''
def make_label_key(dial_id, turn_id):
    return f'[d]{dial_id}[t]{turn_id}'

def filter_data(data, filter):
    left = 0 
    filtered = {}
    for key in data.keys():
        if filter[key] == 'true':
            left +=1
            filtered[key] = data[key]
    return filtered

def merge_data(unlabeled_data_path, made_label): # 이걸 왜 하는거지?
    dataset = json.load(open(unlabeled_data_path , "r"))    
    for d_id in dataset.keys():
        dialogue = dataset[d_id]['log']
        turn_text = ""
        for t_id, turn in enumerate(dialogue):
            for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                label_key = make_label_key(d_id, t_id, key)
                if made_label[label_key] != ontology.QA['NOT_MENTIONED']:
                    dataset[d_id]['log'][t_id]['belief'][key] = made_label[label_key]
    return dataset

def dictionary_split(data, ratio = 0.8):
    keys = list(data.keys())
    random.shuffle(keys)

    train_keys = keys[:int(len(keys)*0.8)]
    dev_keys = keys[int(len(keys)*0.8):]

    train = {}
    dev = {}

    for key in train_keys:
        train[key] = data[key]

    for key in dev_keys:
        dev[key] = data[key]

    return train, dev
