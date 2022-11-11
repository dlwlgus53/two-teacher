import os
import pdb
import pickle
import ontology
import csv, json
from collections import defaultdict, Counter

def evaluate_metrics(all_prediction, raw_file):
    # domain, schema accuracy 는 틀린부분이 있어서 return 하지 않음.
    
    schema = ontology.QA['all-domain'] # next response 는 제외
    turn_acc, joint_acc, micro_f1, turn_cnt, joint_cnt = 0, 0, 0, 0, 0
    
    for key in raw_file.keys():
        if key not in all_prediction.keys(): continue
        dial = raw_file[key]['log']
        for turn_idx, turn in enumerate(dial):
            try:
                belief_label = turn['belief']
                belief_pred = all_prediction[key][str(turn_idx)]
            except:
                pdb.set_trace()
            belief_label = [f'{k} : {v}' for (k,v) in belief_label.items()] 
            belief_pred = [f'{k} : {v}' for (k,v) in belief_pred.items()] 
            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            joint_cnt +=1
            
            ACC, schema_acc_temp  = compute_acc(belief_label, belief_pred, schema)
            micro_f1 += cal_f1(belief_label, belief_pred)
            
            turn_acc += ACC
            # schema_acc = {k : v + schema_acc_temp[k] for (k,v) in schema_acc.items()}

            turn_cnt += 1
    
    return joint_acc/joint_cnt, turn_acc/turn_cnt

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
    schema_acc = {s:1 for s in slot_temp}
    for g in gold:
        if g not in pred:
            miss_gold += 1
            schema_acc[g.split(" : ")[0]] -=1
            miss_slot.append(g.split(" : ")[0])

                                    
            
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
            schema_acc[p.split(" : ")[0]] -=1

            
        
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC, schema_acc

def cal_f1(a, p):
    answer_tokens = a
    pred_tokens = p
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        mini_f1 = 0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(answer_tokens)
        mini_f1 = (2 * precision * recall) / (precision + recall)
    return mini_f1


if __name__ == '__main__':
    pred_file = json.load(open('logs/debugging/pred_belief.json', "r"))
    ans_file = json.load(open('../woz-data/MultiWOZ_2.1/split0.01/train_data10.01.json' , "r"))
    print(evaluate_metrics(pred_file, ans_file))
