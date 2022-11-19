import random
def make_label_key(dial_id, turn_id, slot):
    return f'[d]{dial_id}[t]{turn_id}[s]{slot}'

def dictionary_split(data, ratio = 0.8):
    keys = list(data.keys())
    random.shuffle(keys)

    train_keys = keys[:int(len(keys)*0.8)]
    dev_keys = keys[int(len(keys)*0.8):]

    train = {}
    dev = {}

    for key in train_keys:
        train[key] = data[key]

    for key in dev_keys:
        dev[key] = data[key]

    return train, dev
