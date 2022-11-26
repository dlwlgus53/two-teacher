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
from trainer import mwozTrainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataclass import DSTMultiWozData, VerifyData
from utils import filter_data, merge_data
from evaluate import acc_metric, jga_metric


parser = argparse.ArgumentParser()

# data setting
parser.add_argument('--unlabeled_data_path' , type = str, )
parser.add_argument('--labeled_data_path' , type = str)
parser.add_argument('--valid_data_path' , type = str)
parser.add_argument('--test_data_path' , type = str)

# training setting
parser.add_argument('--seed' ,  type = int, default=1)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--gpus', default=4, type=int,help='number of gpus per node')
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')
parser.add_argument('--patient', type = int, help = 'prefix for all savings', default = 3)

# model parameter
parser.add_argument('--base_trained', type = str, default = "t5-small", help =" pretrainned model from 🤗")
parser.add_argument('--teacher_model_path', type = str,  help =" pretrainned model from 🤗")
parser.add_argument('--V_teacher_model_path', type = str,  help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_path', type = str, help =" pretrainned model from 🤗")
parser.add_argument('--checkpoint_file' , type = str,  help = 'pretrainned model')
parser.add_argument('--batch_size_per_gpu' , type = int, default=4)
parser.add_argument('--test_batch_size_per_gpu' , type = int, default=16)


def init_experiment(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU


def load_trained(model, model_path = None, optimizer = None):
    if args.pretrained_path:
        state_dict = torch.load(args.pretrained_path)
    else:
        state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k.replace("module.","") # [7:]remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    if optimizer != None:
        opt_path = "./model/optimizer/" + args.pretrained_model[7:] #todo
        optimizer.load_state_dict(torch.load(opt_path))
    return model
    
         
    
def find_test_model(test_output_dir):
    import os
    fileData = []
    for fname in os.listdir(test_output_dir):
        if fname.startswith('epoch'):
            fileData.append(fname)
    fileData = sorted(fileData, key=lambda x : x[-9:])
    return fileData[0:args.test_num]

def evaluate():
    test_dataset =Dataset(args, args.test_path, 'test')
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False)
    
    
    test_output_dir = f"model/woz{args.save_prefix}/"
    test_model_paths = find_test_model(test_output_dir)
    for test_model_path in test_model_paths:
        state_dict = torch.load(test_output_dir + test_model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.","") # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        model =  GPT2LMHeadModel.from_pretrained(args.base_trained,  pad_token_id=args.tokenizer.eos_token_id).to('cuda')
        model.load_state_dict(new_state_dict)

        
        joint_goal_acc, slot_acc, F1, detail_wrong = test(args, model, loader, test_dataset)
        
        
        if args.detail_log:
            utils.dict_to_json(detail_wrong, f'{args.save_prefix}{test_model_path}.json')

def find_test_model(test_output_dir):
    import os
    from operator import itemgetter
    fileData = []
    for fname in os.listdir(test_output_dir):
        if fname.startswith('epoch'):
            fileData.append(fname)
    fileData = sorted(fileData, key=lambda x : x[-9:])
    return fileData[0]


if __name__ =="__main__":
    args = parser.parse_args()
    os.makedirs(f"./logs/{args.save_prefix}", exist_ok=True); os.makedirs("./out", exist_ok = True);
    os.makedirs("./out", exist_ok = True);
    os.makedirs(f"model/optimizer/{args.save_prefix}", exist_ok=True)
    os.makedirs(f"model/{args.save_prefix}",  exist_ok=True)
    
    init_experiment(args.seed)
    log_folder = f'logs/{args.save_prefix}/'
    logger = CreateLogger('main', os.path.join(log_folder,'info.log'))
    logger.info(args)
    writer = SummaryWriter()

    teacher = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    student = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    V_teacher = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    V_student = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    
    if args.checkpoint_file:
        model = load_traiend(model, args.checkpoint_file)

    teacher = nn.DataParallel(teacher).cuda()
    student = nn.DataParallel(student).cuda()
    V_teacher = nn.DataParallel(V_teacher).cuda()
    V_student = nn.DataParallel(V_student).cuda()

    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    
    labeled_dataset = DSTMultiWozData(tokenizer, args.labeled_data_path, 'train') # 라벨된것만 나오는거, 확실한거지?
    unlabeled_dataset =  DSTMultiWozData(tokenizer,args.unlabeled_data_path, 'train') # 전체 데이터셋

    valid_dataset = DSTMultiWozData(tokenizer,  args.valid_data_path, 'valid')
    test_dataset = DSTMultiWozData(tokenizer,  args.test_data_path, 'test')

    V_train_dataset = VerifyData(tokenizer, labeled_dataset.get_data(), 'train')
    V_valid_dataset = VerifyData(tokenizer, valid_dataset.get_data(), 'valid')

    optimizer_setting = {
        'warmup_init':False,
        'lr':1e-3, 
        'eps':(1e-30, 1e-3),
        'clip_threshold':1.0,
        'decay_rate':-0.8,
        'beta1':None,
        'weight_decay':0.0,
        'relative_step':False,
        'scale_parameter':False,
    }
    teacher_optimizer = Adafactor(teacher.parameters(), **optimizer_setting)
    student_optimizer = Adafactor(student.parameters(), **optimizer_setting)
    V_teacher_optimizer = Adafactor(V_teacher.parameters(), **optimizer_setting)
    V_student_optimizer = Adafactor(V_student.parameters(), **optimizer_setting)
    trainer_setting = {
        'train_batch_size' : args.batch_size_per_gpu * args.gpus ,
        'test_batch_size' : args.test_batch_size_per_gpu * args.gpus,
        'tokenizer' : tokenizer,        
        'log_folder' : log_folder,
        'save_prefix' : args.save_prefix,
        'max_epoch' : args.max_epoch,
    }


    teacher_trainer = mwozTrainer(
        model = teacher,
        valid_data = valid_dataset,
        test_data = test_dataset,
        optimizer = teacher_optimizer,
        logger_name = 'teacher',
        evaluate_fnc = jga_metric, 
        belief_type = True,
        **trainer_setting)
    
    student_trainer = mwozTrainer(
        model = student,
        valid_data = valid_dataset,
        test_data = test_dataset,
        optimizer = student_optimizer,
        logger_name = 'student',
        evaluate_fnc = jga_metric, 
        belief_type = True,
        **trainer_setting)


    V_teacher_trainer = mwozTrainer(
        model = V_teacher,
        valid_data = V_valid_dataset,
        test_data = V_valid_dataset,
        optimizer = V_teacher_optimizer,
        logger_name = 'V_teacher',
        evaluate_fnc = acc_metric, 
        belief_type = False,
        **trainer_setting)


    V_student_trainer = mwozTrainer(
        model = V_student,
        valid_data = V_valid_dataset,
        test_data = V_valid_dataset,
        optimizer = V_student_optimizer,
        logger_name = 'V_student',
        evaluate_fnc = acc_metric, 
        belief_type = False,
        **trainer_setting)

    teacher_trainer.work(train_data = labeled_dataset,  test = True, save = True, \
        train =False, model_path = args.teacher_model_path) 
    V_teacher_trainer.work(train_data = V_train_dataset, test = True, save = True) 


    for i in range(args.max_epoch):
        made_label = teacher_trainer.make_label(data = unlabeled_dataset)
        temp_dataset = merge_data(args.unlabeled_data_path, made_label)
        V_pseudo_dataset = VerifyData(tokenizer, temp_dataset, 'test')

        TFresult = V_teacher_trainer.make_label(data = V_pseudo_dataset)
        verified_result = filter_data(made_label, TFresult) #output : label_key
        logger.info (f"From {len(made_label)}, {len(verified_result)} is left")


        labeled_dataset.update(args.unlabeled_data_path, verified_result)
        V_train_dataset.update(args.unlabeled_data_path, verified_result) # TODO

        test_score =  student_trainer.work(train_data = labeled_dataset, test = True, save = True)
        teacher_trainer.work(train_data = labeled_dataset,  test = True) 

        V_student_trainer.work(train_data = V_train_dataset)

        logger.info(test_score)

        teacher_trainer.set_model(student_trainer.get_model())
        V_teacher_trainer.set_model(V_student_trainer.get_model())

        student_trainer.init_model(args.base_trained) # 이렇게 해도 업데이트가 잘 될까?
        V_teacher_trainer.init_model(args.base_trained)
        labeled_dataset.reverse_update()
        V_train_dataset.reverse_update()