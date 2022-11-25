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
from utils import make_label_key
# from utils import save_pickle
class mwozTrainer:
    def __init__(self,model, train_batch_size, test_batch_size, tokenizer, optimizer, log_folder, save_prefix, max_epoch, logger_name, \
    train_data = None, valid_data = None,  test_data = None, patient = 3):
        self.log_folder = log_folder
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.writer = SummaryWriter()
        self.logger = CreateLogger(logger_name, os.path.join(log_folder,'info.log'))
        self.save_prefix = save_prefix
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.model = model
        self.max_epoch = max_epoch
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.patient = patient

    def work(self, train_data = None) :
        if train_data: self.train_data = train_data
        self.model.cuda()
        min_loss = float('inf')
        best_model = ''        

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
                logger.info(f"Early stop in Epoch {epoch}")
                break

        self.model = best_model

                # torch.save(self.model.state_dict(), f"model/{self.save_prefix}/epoch_{epoch}_loss_{loss:.4f}.pt")
                # torch.save(optimizer.state_dict(), f"model/optimizer/{self.save_prefix}/epoch_{epoch}_loss_{loss:.4f}.pt")


    def make_label(self, data):
        generated_label = {}
        max_iter = int(len(data) / self.test_batch_size)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=self.test_batch_size,\
            collate_fn=self.data.collate_fn)

        self.model.eval()
        self.logger.info("Labeling Start")

        with torch.no_grad():
            for iter,batch in enumerate(test_loader):
                # make_label_key(dial_id, turn_id, slot)
                outputs_text = self.model.module.generate(input_ids=batch['input']['input_ids'].to('cuda'))
                outputs_text =self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                for idx in range(len(outputs_text)):
                    dial_id = batch['dial_id'][idx]
                    turn_id = batch['turn_id'][idx]
                    slot = batch['schema'][idx]
                    pdb.set_trace()
                    label_key = make_label_key(dial_id, turn_id, slot)
                    generated_label[label_key] = outputs_text[idx]

                if (iter + 1) % 50 == 0:
                    self.logger.info(f"Test : {iter+1}/{test_max_iter}")

        return  labeled_dataset # format? {key : value is ok}

        
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
                    answer_text = self.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
                    predict_text = self.tokenizer.batch_decode(batch['label']['input_ids'], skip_special_tokens = True)
                    p_a_text = [f'ans : {a} pred : {p} || ' for (a,p) in zip(answer_text, predict_text)]

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



    def test(self,epoch_num):
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
        
        return belief_state
    
    def evaluate(self, pred_result, answer, unseen_data_path):
        with open(unseen_data_path, 'r') as file:
            unseen_data = json.load(file)

        return  evaluate_metrics(pred_result, answer, unseen_data)
