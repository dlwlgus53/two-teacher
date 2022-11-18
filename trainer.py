import os
import torch
import pdb 
import json
import logging
import ontology
from utils import*
from logger_conf import CreateLogger
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate_metrics

# from utils import save_pickle
class mwozTrainer:
    def __init__(self, tokenizer, optimizer, log_folder):
        self.log_folder = log_folder
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.writer = SummaryWriter()
        self.logger = CreateLogger('trainer', os.path.join(log_folder,'info.log'))
        
    def train(self, model, train_loader, epoch_num, max_iter):

        loss_sum =0 
        model.train()

        for iter, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input']['input_ids'].to('cuda')
            labels = batch['label']['input_ids'].to('cuda')

            outputs = model(input_ids=input_ids, labels=labels)
            
            loss =outputs.loss.mean()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.detach()
        
            if (iter + 1) % 50 == 0:
                self.logger.info(f"training : {iter+1}/{max_iter} loss : {loss_sum/50:.4f}")
                outputs_text = model.module.generate(input_ids=input_ids)
                self.writer.add_scalar(f'Loss/train_epoch{epoch_num} ', loss_sum/50, iter)
                loss_sum =0 

                if (iter+1) % 150 ==0:
                    answer_text = self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                    predict_text = self.tokenizer.batch_decode(batch['label']['input_ids'], skip_special_tokens = True)
                    p_a_text = [f'ans : {a} pred : {p} || ' for (a,p) in zip(answer_text, predict_text)]

                    self.writer.add_text(f'Answer/train_epoch{epoch_num}',\
                    '\n'.join(p_a_text),iter)

                    # question_text = tokenizer.batch_decode(batch['input']['input_ids'], skip_special_tokens = True)
                    # writer.add_text(f'Question/train_epoch{epoch_num}',\
                    # '\n'.join(question_text),iter)


    def valid(self, model, valid_loader, epoch_num, max_iter):
        model.eval()
        loss_sum = 0
        self.logger.info("Validation Start")
        with torch.no_grad():
            for iter, batch in enumerate(valid_loader):
                input_ids = batch['input']['input_ids'].to('cuda')
                labels = batch['label']['input_ids'].to('cuda')
                outputs = model(input_ids=input_ids, labels=labels)
                loss =outputs.loss.mean()
                
                loss_sum += loss.detach()
                if (iter + 1) % 50 == 0:
                    self.logger.info(f"Validation : {iter+1}/{max_iter}")

        self.writer.add_scalar(f'Loss/valid ', loss_sum/iter, epoch_num)
        self.logger.info(f"Validation loss : {loss_sum/iter:.4f}")
        return  loss_sum/iter



    def test(self, model, test_loader, max_iter):
        belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
        model.eval()
        loss_sum = 0
        self.logger.info("Test start")
        with torch.no_grad():
            for iter,batch in enumerate(test_loader):
                outputs_text = model.module.generate(input_ids=batch['input']['input_ids'].to('cuda'))
                outputs_text =self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                
                for idx in range(len(outputs_text)):
                    dial_id = batch['dial_id'][idx]
                    turn_id = batch['turn_id'][idx]
                    schema = batch['schema'][idx]
                    if turn_id not in belief_state[dial_id].keys():
                        belief_state[dial_id][turn_id] = {}
                    if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
                    else: belief_state[dial_id][turn_id][schema] = outputs_text[idx]

                if (iter + 1) % 50 == 0:
                    self.logger.info(f"Test : {iter+1}/{max_iter}")
            
            with open(os.path.join(self.log_folder, 'pred_belief.json'), 'w') as fp:
                json.dump(belief_state, fp, indent=4, ensure_ascii=False)
        
        return belief_state
    
    def evaluate(self, pred_result, answer, unseen_data_path):
    with open(unseen_data_path, 'r') as file:
        unseen_data = json.load(file)

    return  evaluate_metrics(pred_result, answer, unseen_data)
