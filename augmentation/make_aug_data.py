'''
시나리오 업데이트 한 데이터셋을 만든다.

그 데이터셋을 넣는다 근데 거기서 파라미터를 얼만큼 aug하는 지 알려줘야 한다. > randomly 진행해야 하네
> 이거를 여기서 하고 dataclass에서는 할 일이 아닌 것 같은데?

그걸로 만들어서

멀쩡한 데이터셋으로 나와서 다른 모델들이 학습이 가능하도록 해야한다


그럼 지금 할 일은? 일단 이상하더라도 working하는 augmentation 모델을 만들어 놓자
'''

import os
import pdb
import sys
import json
import torch
import random
import ontology
import argparse
from collections import OrderedDict
from logger_conf import CreateLogger

import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataclass_aug import DSTMultiWozAugData
from utils import filter_data, merge_data
from collections import defaultdict

parser = argparse.ArgumentParser()

# data setting
parser.add_argument('--labeled_data_path' , type = str)
# training setting
parser.add_argument('--short' ,  type = int, default=1)
parser.add_argument('--seed' ,  type = int, default=1)
parser.add_argument('--gpus', default=2, type=int,help='number of gpus per node')
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')

# model parameter
parser.add_argument('--base_trained', type = str, default = "t5-small", help =" pretrainned model from 🤗")
parser.add_argument('--aug_model_path', type = str,  help =" pretrainned model from 🤗")
parser.add_argument('--test_batch_size_per_gpu' , type = int, default=16)

# aug parmeter
parser.add_argument('--scenario_percent', type = float)

def init_experiment(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU

def load_trained(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.","") # [7:]remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def generate_user( aug_model, tokenizer, dst, system):
    curr_dst_str = str(dst)
    curr_dst_str = curr_dst_str.replace("{","").replace("}","").replace(": ", ' is ').replace("'","")
    q = f"make dialgoue. DST : {curr_dst_str} System : {system}"
    input_ids = tokenizer(q, return_tensors = "pt").input_ids
    generaetd = aug_model.generate(input_ids=input_ids.to('cuda'))
    return tokenizer.decode(generaetd[0].cpu(), skip_special_tokens = True)

def make_domain_turn_dict(dataset, short=0):
    values = defaultdict(list)
    S = 0
    for dial in dataset:
        S +=1
        if short == True and S > 1000:
            break
        if 'curr_belief' in dial[0] :        
            for t_id, turn in enumerate(dial):
                curr_dst = clean_belief_state(turn['curr_belief'])
                curr_domains = list(set([key.split("-")[0] for key in list(curr_dst.keys())]))
                for domain in curr_domains:
                    values[domain].append(turn.copy())
    return values

def clean_belief_state(belief):
    new_belief = {}

    for key, value in belief.items():
        if key.startswith("police") or key.startswith("hospital"):
            pass
        else:
            new_belief[key] = value
    
    return new_belief


def make_value_dict(dataset):
    values = defaultdict(list)
    for dial in dataset:
        for t_id, turn in enumerate(dial):
            if 'belief' in turn:
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    if key in clean_belief_state(turn['belief']): 
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


def aug_dst(dst, value_dict): # TODO 현재 turn의 belief state만 중심적으로 봐야한다.
    
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
    
    result = [ add(dst.copy(), value_dict), delete(dst.copy()), replace(dst.copy(), value_dict)]
    result = [i for i in  result if i is not None]
    result = [i for i in  result if len(i)!=0]

    return result

def update_scenario(dataset, change_rate,  aug_model, tokenizer, short =0):
    '''
    doamin_turn_dict; 
    '''
    domain_turn_dict = make_domain_turn_dict(dataset, short)
    value_dict= make_value_dict(dataset)

    changed_dataset = dataset.copy()
    S=0
    for dial in dataset:
        S +=1
        if short == True and S > 1000:
            break
        system = "" # should be changed
        stack_dial = []
        aug_cnt = 0
        d_id = dial[0]['dial_id']

        for t_id, turn in enumerate(dial):
            try:
                turn['belief'] = clean_belief_state(turn['belief'])
                turn['curr_belief'] = clean_belief_state(turn['curr_belief'])
                turn['prev_belief'] = clean_belief_state(turn['prev_belief'])
            except:
                turn['belief'] = clean_belief_state(turn['pred_belief'])
                turn['curr_belief'] = clean_belief_state(turn['curr_pred_belief'])
                turn['prev_belief'] = clean_belief_state(turn['prev_pred_belief'])

            if  len(dial)-1 != t_id and len(turn['belief']) !=0 and random.random()<change_rate:
                copy_stack = stack_dial.copy()
                prev_domains = list(set([key.split("-")[0] for key in list(turn['prev_belief'].keys())]))

                change_domain = random.choice(list(set(domain_turn_dict) - set(prev_domains)))

                new_turn = random.choice(domain_turn_dict[change_domain]).copy()
                new_turn['belief'] = clean_belief_state(new_turn['belief'])
                new_turn['curr_belief'] = clean_belief_state(new_turn['curr_belief'])
                new_turn['prev_belief'] = clean_belief_state(new_turn['prev_belief'])

                # add original
                new_turn_org = new_turn.copy()
                aug_cnt +=1
                new_turn_org['belief'] = dict(new_turn['curr_belief'], **turn['prev_belief'])
                new_turn_org['prev_belief'] = turn['prev_belief']
                new_turn_org['turn_num'] = t_id
                new_turn_org['dial_id'] = d_id+'_'+str(aug_cnt)
                copy_stack.append(new_turn_org)
                changed_dataset.append(copy_stack)

                # add changed
                dsts = aug_dst(new_turn['curr_belief'], value_dict)
                for dst in dsts:
                    copy_stack = stack_dial.copy()
                    aug_cnt +=1
                    new_turn_ch = new_turn.copy()
                    system = "" if t_id ==0 else stack_dial[-1]['resp']
                    new_turn_ch['curr_belief'] = dst
                    new_turn_ch['belief'] = dict(new_turn['curr_belief'], **turn['prev_belief'])
                    new_turn_ch['user'] = generate_user(aug_model, tokenizer, dst, system)
                    new_turn_ch['turn_num'] = t_id
                    new_turn_ch['prev_belief'] = turn['prev_belief']
                    new_turn_ch['prev_belief'] = turn['prev_belief']
                    new_turn_ch['dial_id'] = d_id+'_'+str(aug_cnt)

                    copy_stack.append(new_turn_ch)
                    changed_dataset.append(copy_stack)

            stack_dial.append(turn)

    return changed_dataset

    
if __name__ =="__main__":
    args = parser.parse_args()
    os.makedirs(f"./logs/{args.save_prefix}", exist_ok=True); os.makedirs("./out", exist_ok = True);
    os.makedirs("./out", exist_ok = True);
    
    init_experiment(args.seed)
    log_folder = f'logs/{args.save_prefix}/'
    logger = CreateLogger('main', os.path.join(log_folder,'info.log'))
    logger.info("------------------------START NEW DATA MAKING-----------------")

    logger.info(args)
    writer = SummaryWriter()

    aug_model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    aug_model = load_trained(aug_model, args.aug_model_path)
    aug_model = aug_model.cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    raw_dial = json.load(open(args.labeled_data_path , "r"))
    update_scenario_aug_data = update_scenario(raw_dial, args.scenario_percent,  aug_model, tokenizer, args.short)

    with open(f"../data/{args.save_prefix}_scenario.json", "w") as f:
        json.dump(update_scenario_aug_data, f, indent = 4)



    