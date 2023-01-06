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
parser.add_argument('--aug_percent' , type = float)

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


def make_domain_turn_dict(dataset, short=0):
    values = defaultdict(list)
    S = 0
    for d_id in dataset.keys():
        S +=1
        if short == True and S > 1000:
            break
        dialogue = dataset[d_id]['log']
        system = "" # should be changed
        for t_id, turn in enumerate(dialogue):
            dst = turn['belief']
            # find curr_dst
            if t_id == 0:
                curr_dst = dst
            else:
                dst_set = set(dst.items())
                prev_dst_set = set(prev_dst.items())
                curr_dst = dict(dst_set - prev_dst_set)
            curr_domain = list(set([key.split("-")[0] for key in list(curr_dst.keys())]))
            for domain in curr_domain:
                values[domain].append(turn.copy())

            prev_dst = turn['belief']
    return values



def update_scenario(dataset, change_rate, short =0):
    domain_turn_dict = make_domain_turn_dict(dataset, short)
    # 없는 경우는 바꾸지 않습니다.
    changed_dataset = dataset.copy()
    S=0
    for d_id in dataset.keys():
        S +=1
        if short == True and S > 1000:
            break
        dialogue = dataset[d_id]['log']
        system = "" # should be changed
        stack_dial = []
        aug_cnt = 0
        for t_id, turn in enumerate(dialogue):
            stack_dial.append(turn)
            if len(dialogue)-1 != t_id and len(turn['belief']) !=0 and random.random()<change_rate:
                copy_stack = stack_dial.copy()
                prev_domains = list(set([key.split("-")[0] for key in list(turn['prev_belief'].keys())]))
                change_domain = random.choice(list(set(domain_turn_dict) - set(prev_domains)))
                new_turn = random.choice(domain_turn_dict[change_domain]).copy()
                new_turn['belief'] = dict(new_turn['curr_belief'], **turn['prev_belief'])
                new_turn['turn_num'] = t_id
                
                copy_stack[t_id] = new_turn
                changed_dataset[d_id+'_'+str(aug_cnt)] ={'log' : copy_stack}
                aug_cnt +=1


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
    aug_model = nn.DataParallel(aug_model).cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    raw_dial = json.load(open(args.labeled_data_path , "r"))
    update_scenario_data = update_scenario(raw_dial, args.scenario_percent, args.short)
    with open(f"../data/{args.save_prefix}_scenario.json", "w") as f:
        json.dump(update_scenario_data, f, indent = 4)


    labeled_dataset = DSTMultiWozAugData(tokenizer, args.labeled_data_path, 'train' , short = args.short) 


    