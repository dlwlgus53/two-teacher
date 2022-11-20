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




parser = argparse.ArgumentParser()

# data setting
parser.add_argument('--all_train_data_path' , type = str, default = '../woz-data/MultiWOZ_2.1/train_data.json')
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
    verifier = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    
    if args.checkpoint_file:
        model = load_traiend(model, args.checkpoint_file)

    teacher = nn.DataParallel(teacher)
    student = nn.DataParallel(student)
    verifier = nn.DataParallel(verifier)
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    
    labeled_dataset = DSTMultiWozData(tokenizer, args.labeled_data_path, 'train') # 라벨된것만 나오는거, 확실한거지?
    all_train_dataset =  DSTMultiWozData(tokenizer,args.all_train_data_path, 'train') # 전체 데이터셋
    verify_dataset = VerifyData(tokenizer, labeled_dataset.get_data())
    valid_dataset = DSTMultiWozData(tokenizer,  args.valid_data_path, 'valid')
    test_dataset = DSTMultiWozData(tokenizer,  args.test_data_path, 'test')

    optimizer = Adafactor(teacher.parameters(),lr=1e-3, # 이게 선생님것만 해도 되나??
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
                    )

    teacher_trainer = mwozTrainer(
        model = teacher,
        valid_data = valid_dataset,
        test_data = test_dataset,
        train_batch_size = args.batch_size_per_gpu * args.gpus ,
        test_batch_size = args.test_batch_size_per_gpu * args.gpus,
        tokenizer = tokenizer,
        optimizer = optimizer,
        log_folder = log_folder,
        save_prefix = args.save_prefix,
        max_epoch = args.max_epoch,
        logger_name = 'teacher')
    
    student_trainer = mwozTrainer(
        model = student,
        valid_data = valid_dataset,
        test_data = test_dataset,
        train_batch_size = args.batch_size_per_gpu * args.gpus ,
        test_batch_size = args.test_batch_size_per_gpu * args.gpus,
        tokenizer = tokenizer,
        optimizer = optimizer,
        log_folder = log_folder,
        save_prefix = args.save_prefix,
        max_epoch = args.max_epoch,
        logger_name = 'student')



    verify_trainer = mwozTrainer(
        model = teacher,
        valid_data = valid_dataset,
        test_data = test_dataset,
        train_batch_size = args.batch_size_per_gpu * args.gpus ,
        test_batch_size = args.test_batch_size_per_gpu * args.gpus,
        tokenizer = tokenizer,
        optimizer = optimizer,
        log_folder = log_folder,
        save_prefix = args.save_prefix,
        max_epoch = args.max_epoch,
        logger_name = 'verifier')



    for i in range(10):
        teacher_trainer.work(train_data = labeled_dataset) # keeps chainign
        verify_trainer.work(train_data = labeled_dataset)
        made_label = teacher_trainer.make_label(data = all_train_dataset)
        verified_label = verify_the_label(made_label, verify)
        labeled_dataset.update(verified_label)
        JGA,loss = student_trainer.work(train_data = labeled_dataset)
        print(JGA, loss)
        teacher_trainer.set_model(student_trainer.get_model())
        
 
    # best_model_path =  find_test_model(f"model/{args.save_prefix}/")
    # best_model = load_trained(model, f"model/{args.save_prefix}/{best_model_path}")
    # pred_result = trainer.test(best_model, test_loader,  test_max_iter)
    # answer = json.load(open(args.test_data_path , "r"))
    # JGA, slot_acc, unseen_recall = trainer.evaluate(pred_result, answer, args.unseen_data_path)
    # logger.info(f"JGA : {JGA}, slot_acc : {slot_acc}, unseen_recall : {unseen_recall}")
    
    
    

