from collections import defaultdict
import json

f = open('data/x_training.txt','r',encoding='utf-8')
word_dict = defaultdict(int)
pos_dict = defaultdict(int)
ner_dict = defaultdict(int)

pre_question = ''
for line in f.readlines():
    sents = line.strip().split('\t')
    for sent in sents:
        if sent == pre_question:
            continue
        for item in sent.strip().split(' '):
            item = item.split('/')
            #print(item)
            word_dict[item[0]]+=1
            pos_dict[item[1]]+=1
            ner_dict[item[2]]+=1
    pre_question = sents[0]

json.dump({k:v for k,v in word_dict.items() if v > 5},open('data/word_dict','w',encoding='utf-8'))
json.dump({k:v for k,v in pos_dict.items() if v > 5},open('data/pos_dict','w',encoding='utf-8'))
json.dump({k:v for k,v in ner_dict.items() if v > 5},open('data/ner_dict','w',encoding='utf-8'))