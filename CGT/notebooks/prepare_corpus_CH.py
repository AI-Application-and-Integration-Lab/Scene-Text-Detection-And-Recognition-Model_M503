from cgitb import text
import os
import random
import re
import pandas as pd

max_length = 13
min_length = 3
root = '/media/avlab/DATA/ABINet/workdir/wikitext-103'

with open('/media/avlab/DATA/ABINet/workdir/wikitext-103/wiki.train.tokens', 'r', encoding='utf8') as file:
    lines = [Line.rstrip() for Line in file.readlines()]

inp, gt = [], []
for line in lines:
    # text = re.sub('[^\u4e00-\u9fff]+', '', line)
    line = re.sub('[^\w]', '', line)
    line = re.sub('[丨丿]', '', line)
    # print(text)
    if len(line) < min_length:
        # print('short-text', text)
        continue
    if len(line) >= max_length:
        # print('long-text', text)
        continue
    inp.append(line)
    gt.append(line)

ch_cha = inp.copy()
ch_cha[0:] = [''.join(ch_cha[0:])]
ch_cha = ch_cha[0]
# print(ch_cha)
# exit()

train_voc = os.path.join(root, 'chi_train.csv')
pd.DataFrame({'inp':inp, 'gt':gt}).to_csv(train_voc, index=None, sep='\t', encoding='utf_8_sig')

print(len(inp))

# ################### create evaluation set ##################
def disturb(word, degree, p=0.2):
    if len(word) // 2 < degree: 
        return word
    if random.random() < p: 
        return word
    else:
        # print(word)
        index = list(range(len(word)))
        random.shuffle(index)
        index = index[:degree]
        new_word = []
        for i in range(len(word)):
            if i not in index: 
                new_word.append(word[i])
                continue
            op = random.random()
            if op < 0.1: # add
                new_word.append(random.choice(ch_cha))
                new_word.append(word[i])
            elif op < 0.2: continue  # remove
            else: new_word.append(random.choice(ch_cha))  # replace
        return ''.join(new_word)

lines = inp
degree = 1
keep_num = 1000

random.shuffle(lines)
part_lines = lines[:keep_num]
inp, gt = [], []

for w in part_lines:
    w = w.strip().lower()
    new_w = disturb(w, degree)
    inp.append(new_w)
    gt.append(w)
    
eval_voc = os.path.join(root, f'chi_eval.csv')
pd.DataFrame({'inp':inp, 'gt':gt}).to_csv(eval_voc, index=None, sep='\t', encoding='utf_8_sig')
print(len(inp))