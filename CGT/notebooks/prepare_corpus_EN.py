import os
import random
import re
import pandas as pd

max_length = 12
min_length = 3
root = '/media/avlab/DATA/ABINet/workdir/wikitext-103'

with open('/media/avlab/DATA/ABINet/workdir/wikitext-103/wiki.train.tokens', 'r') as file:
    lines = file.readlines()

inp, gt = [], []
for line in lines[900000:1000000]:
    token = line.lower().split()
    for text in token:
        text = re.sub('[^0-9a-zA-Z]+', '', text)
        if len(text) < min_length:
            # print('short-text', text)
            continue
        if len(text) > max_length:
            # print('long-text', text)
            continue
        # inp.append(text)
        gt.append(text)
    if len(gt)>5000:
        break

train_voc = os.path.join(root, 'WikiText-103-corpus_eval.csv')
pd.DataFrame({'gt':gt}).to_csv(train_voc, index=None, sep='\t')