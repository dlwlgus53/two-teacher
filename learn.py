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

from dataclass import DSTMultiWozData

parser = argparse.ArgumentParser()

# data setting
parser.add_argument('--train_data_path' , type = str)
parser.add_argument('--unseen_data_path' , type = str)
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

    model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    
    if args.checkpoint_file:
        model = load_traiend(model, args.checkpoint_file)

    model = nn.DataParallel(model).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)

    train_dataset = DSTMultiWozData(tokenizer, args.train_data_path, 'train')
    valid_dataset = DSTMultiWozData(tokenizer,  args.valid_data_path, 'valid')
    test_dataset = DSTMultiWozData(tokenizer,  args.test_data_path, 'test')

    train_batch_size = args.batch_size_per_gpu * args.gpus    
    test_batch_size = args.test_batch_size_per_gpu * args.gpus

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size,\
     collate_fn=train_dataset.collate_fn)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=test_batch_size,\
     collate_fn=valid_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size,\
     collate_fn=test_dataset.collate_fn)

    optimizer = Adafactor(model.parameters(),lr=1e-3,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False,
                    )

    min_loss = float('inf')
    best_performance = {}
    train_max_iter = int(len(train_dataset) / train_batch_size)
    valid_max_iter = int(len(valid_dataset) / test_batch_size)
    test_max_iter = int(len(test_dataset) / test_batch_size)


    trainer = mwozTrainer(tokenizer, optimizer, log_folder)
    try_count = 0
    for epoch in range(args.max_epoch):
        try_count +=1
        trainer.train(model, train_loader,epoch, train_max_iter)
        loss = trainer.valid(model, valid_loader, epoch, valid_max_iter)
        if loss < min_loss:
            try_count =0
            min_loss = loss
            torch.save(model.state_dict(), f"model/{args.save_prefix}/epoch_{epoch}_loss_{loss:.4f}.pt")
            torch.save(optimizer.state_dict(), f"model/optimizer/{args.save_prefix}/epoch_{epoch}_loss_{loss:.4f}.pt")
        if try_count > args.patient:
            logger.info(f"Early stop in Epoch {epoch}")
            break

    best_model_path =  find_test_model(f"model/{args.save_prefix}/")
    best_model = load_trained(model, f"model/{args.save_prefix}/{best_model_path}")
    pred_result = trainer.test(best_model, test_loader,  test_max_iter)
    answer = json.load(open(args.test_data_path , "r"))
    JGA, slot_acc, unseen_recall = trainer.evaluate(pred_result, answer, args.unseen_data_path)
    logger.info(f"JGA : {JGA}, slot_acc : {slot_acc}, unseen_recall : {unseen_recall}")
    
    
    

