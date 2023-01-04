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
from utils import make_label_key
from collections import OrderedDict

from transformers import AutoModel

# from utils import save_pickle
class mwozTrainer:
    def __init__(self,model, train_batch_size, test_batch_size, tokenizer, optimizer, log_folder, save_prefix, max_epoch, logger_name, evaluate_fnc, belief_type, \
    train_data = None, valid_data = None,  test_data = None, patient = 3):
        self.log_folder = log_folder
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.writer = SummaryWriter()
        self.logger = CreateLogger(logger_name, os.path.join(log_folder,'info.log'))
        self.save_prefix = f'{save_prefix}_{logger_name}'
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.model = model
        self.max_epoch = max_epoch
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.patient = patient
        self.evaluate_fnc = evaluate_fnc
        self.belief_type= belief_type
        os.makedirs(f"model/{self.save_prefix}",  exist_ok=True)

    def work(self, train_data = None, test = False, train = True, save = False, model_path = None) :

        if train == False and model_path == None:
            self.logger.error("train is False and model_path is NOne")
        if train_data: self.train_data = train_data
        min_loss = float('inf')
        best_model = ''        

        if train:
            try_ = 0
            for epoch in range(self.max_epoch):
                try_ +=1
                self.train(epoch)
                loss = self.valid(epoch)
                if loss < min_loss:
                    try_ =0
                    min_loss = loss
                    best_model = self.model # deep copy 

                if try_ > self.patient:
                    self.logger.info(f"Early stop in Epoch {epoch}")
                    break

            self.model = best_model
            if save == True:
                torch.save(self.model.state_dict(), f"model/{self.save_prefix}/epoch_{epoch}_loss_{min_loss:.4f}.pt")

        
        if test == True:
            if self.belief_type:
                gold, pred = self.belief_test() 
            else :
                gold, pred = self.test()

            test_score = self.evaluate_fnc(gold, pred)
            self.logger.info(f"Test score : {test_score}")
            return test_score

    def set_model(self,model):
        self.model = model
    
    def get_model(self):
        return self.model 

    def init_model(self, base_model):
        self.model = AutoModel.from_pretrained(base_model, return_dict=True)

    def make_label(self, data):
        generated_label = {}
        max_iter = int(len(data) / self.test_batch_size)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=self.test_batch_size,\
            collate_fn=data.collate_fn)

        self.model.eval()
        self.logger.info("Labeling Start")

        with torch.no_grad():
            for iter,batch in enumerate(loader):
                # make_label_key(dial_id, turn_id, slot)
                outputs_text = self.model.module.generate(input_ids=batch['input']['input_ids'].to('cuda'))
                outputs_text =self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                for idx in range(len(outputs_text)):
                    dial_id = batch['dial_id'][idx]
                    turn_id = batch['turn_id'][idx]
                    slot = batch['schema'][idx]
                    label_key = make_label_key(dial_id, turn_id, slot)
                    generated_label[label_key] = outputs_text[idx]

                if (iter + 1) % 50 == 0:
                    self.logger.info(f"Labeling : {iter+1}/{max_iter}")
        return  generated_label 
        
    def train(self,epoch_num):
        train_max_iter = int(len(self.train_data) / self.train_batch_size)
        train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=self.train_batch_size,\
            collate_fn=self.train_data.collate_fn)

        loss_sum =0 
        self.model.train()

        for iter, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input']['input_ids'].to('cuda')
            labels = batch['label']['input_ids'].to('cuda')
            outputs = self.model(input_ids=input_ids, labels=labels)
            
            loss =outputs.loss.mean()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.detach()
        
            if (iter + 1) % 50 == 0:
                self.logger.info(f"Epoch {epoch_num} training : {iter+1}/{train_max_iter } loss : {loss_sum/50:.4f}")
                outputs_text = self.model.module.generate(input_ids=input_ids)
                self.writer.add_scalar(f'Loss/train_epoch{epoch_num} ', loss_sum/50, iter)
                loss_sum =0 

                if (iter+1) % 150 ==0:
                    predict_text = self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                    answer_text = self.tokenizer.batch_decode(batch['label']['input_ids'], skip_special_tokens = True)
                    p_a_text = [f'ans : {a} pred : {p} || ' for (a,p) in zip(answer_text, predict_text)]
                    self.logger.info(p_a_text)
                    self.writer.add_text(f'Answer/train_epoch{epoch_num}',\
                    '\n'.join(p_a_text),iter)

                    # question_text = tokenizer.batch_decode(batch['input']['input_ids'], skip_special_tokens = True)
                    # writer.add_text(f'Question/train_epoch{epoch_num}',\
                    # '\n'.join(question_text),iter)


    def valid(self,epoch_num):
        valid_max_iter = int(len(self.valid_data) / self.test_batch_size)
        valid_loader = torch.utils.data.DataLoader(dataset=self.valid_data, batch_size=self.test_batch_size,\
            collate_fn=self.valid_data.collate_fn)
        self.model.eval()
        loss_sum = 0
        self.logger.info("Validation Start")
        with torch.no_grad():
            for iter, batch in enumerate(valid_loader):
                input_ids = batch['input']['input_ids'].to('cuda')
                labels = batch['label']['input_ids'].to('cuda')
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss =outputs.loss.mean()
                
                loss_sum += loss.detach()
                if (iter + 1) % 50 == 0:
                    self.logger.info(f"Epoch {epoch_num} Validation : {iter+1}/{valid_max_iter}")

        self.writer.add_scalar(f'Loss/valid ', loss_sum/iter, epoch_num)
        self.logger.info(f"Epoch {epoch_num} Validation loss : {loss_sum/iter:.4f}")
        return  loss_sum/iter


    


    def test(self):
        answer, pred = [], []
        test_max_iter = int(len(self.test_data) / self.test_batch_size)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=self.test_batch_size,\
            collate_fn=self.test_data.collate_fn)
        self.model.eval()
        loss_sum = 0
        self.logger.info("Test start")
        with torch.no_grad():
            for iter,batch in enumerate(test_loader):
                outputs_text = self.model.module.generate(input_ids=batch['input']['input_ids'].to('cuda'))
                outputs_text =self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                label =self.tokenizer.batch_decode(batch['label']['input_ids'], skip_special_tokens = True)
                pred.extend(outputs_text)
                answer.extend(label)
                for idx in range(len(outputs_text)):
                    outputs_text[idx]
        return answer, pred
    



    def belief_test(self):
        test_max_iter = int(len(self.test_data) / self.test_batch_size)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=self.test_batch_size,\
            collate_fn=self.test_data.collate_fn)

        belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
        self.model.eval()
        loss_sum = 0
        self.logger.info("Test start")
        with torch.no_grad():
            for iter,batch in enumerate(test_loader):
                outputs_text = self.model.module.generate(input_ids=batch['input']['input_ids'].to('cuda'))
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
                    self.logger.info(f"Test : {iter+1}/{test_max_iter}")
            
            with open(os.path.join(self.log_folder, 'pred_belief.json'), 'w') as fp:
                json.dump(belief_state, fp, indent=4, ensure_ascii=False)
        
        answer = self.test_data.get_data()

        return answer, belief_state
    

