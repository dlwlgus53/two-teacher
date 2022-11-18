import os
import pdb
import pickle
import ontology
import csv, json
from collections import defaultdict, Counter

def evaluate_metrics(all_prediction, raw_file, unseen_data):
    # domain, schema accuracy 는 틀린부분이 있어서 return 하지 않음.
    
    schema = ontology.QA['all-domain'] # next response 는 제외
    turn_acc, joint_acc, micro_f1, unseen_recall, turn_cnt, usr_cnt = 0, 0, 0, 0, 0,0
    
    for key in raw_file.keys():
        if key not in all_prediction.keys(): continue
        dial = raw_file[key]['log']
        for turn_idx, turn in enumerate(dial):
            belief_unseen_label = []
            try:
                raw_belief_label = turn['belief']
                if type(list(all_prediction[key].keys())[0]) is str:
                    raw_belief_pred = all_prediction[key][str(turn_idx)]
                else:   
                    raw_belief_pred = all_prediction[key][turn_idx]
            except:
                pdb.set_trace()
            belief_label = [f'{k} : {v}' for (k,v) in raw_belief_label.items()] 
            belief_pred = [f'{k} : {v}' for (k,v) in raw_belief_pred.items()] 

            for (k,v) in raw_belief_label.items():
                if k in unseen_data and v in unseen_data[k]:
                    belief_unseen_label.append(f'{k} : {v}')

            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            turn_cnt +=1
            
            acc = compute_acc(belief_label, belief_pred, schema)
            micro_f1 += cal_f1(belief_label, belief_pred)

            if len(belief_unseen_label) != 0:
                unseen_recall += cal_num_same(belief_unseen_label, belief_pred,)
                usr_cnt += len(belief_unseen_label)

            turn_acc += acc
    
    return joint_acc/turn_cnt, turn_acc/turn_cnt, unseen_recall/usr_cnt

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
    
            
def compute_acc(gold, pred, slot_temp):
    detail_wrong = []
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.split(" : ")[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def cal_num_same(a,p):
    answer_tokens = a
    pred_tokens = p
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    return num_same


def cal_f1(a, p, return_all = False):
    answer_tokens = a
    pred_tokens = p
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        precision = 0
        recall =0
        mini_f1 = 0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(answer_tokens)
        mini_f1 = (2 * precision * recall) / (precision + recall)
    if return_all:
        return precision, recall, mini_f1
    else:
        return mini_f1


if __name__ == '__main__':
    seed = 2
    # pred_file = json.load(open(f'logs/baseline_sample{seed}/pred_belief.json', "r"))
    pred_file = json.load(open(f'logs/baseline_sample1.0/pred_belief.json', "r"))

    ans_file = json.load(open('../woz-data/MultiWOZ_2.1/test_data.json' , "r"))
    # unseen_data = json.load(open(f'../woz-data/MultiWOZ_2.1/split0.1/unseen_data{seed}0.1.json' , "r"))
    unseen_data = json.load(open(f'../woz-data/MultiWOZ_2.1/unseen_data.json' , "r"))

    JGA, slot_acc, unseen_recall = evaluate_metrics(pred_file, ans_file, unseen_data)
    print(f'JGA : {JGA*100:.4f} ,  slot_acc : {slot_acc*100:.4f}, unseen_recall : {unseen_recall*100:.4f}')
