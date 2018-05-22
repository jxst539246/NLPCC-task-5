import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import numpy as np

class MyDataSet(Dataset):
    def __init__(self, file_path, label_path, word_dict_path, pos_dict_path, ner_dict_path,max_len=100, is_test=False):
        _word_set = sorted(json.load(open(word_dict_path)).keys())
        self.word2idx = {w: i + 1 for i, w in enumerate(_word_set)}
        self.word2idx[u'<UNK>'] = 0
        _pos_set = sorted(json.load(open(pos_dict_path)).keys())
        self.pos2idx = {w: i + 1 for i, w in enumerate(_pos_set)}
        self.pos2idx[u'<UNK>'] = 0
        _ner_set = sorted(json.load(open(ner_dict_path)).keys())
        self.ner2idx = {w: i + 1 for i, w in enumerate(_ner_set)}
        self.ner2idx[u'<UNK>'] = 0
        print(len(self.word2idx),len(self.ner2idx),len(self.pos2idx))
        self.data = [[],[]]
        self.label = []
        self.is_test = is_test
        self.max_len = max_len
        for lineno, line in enumerate(open(file_path, 'r', encoding='utf8').readlines()):
            sents = line.strip().split('\t')

            if len(sents)==0:
                continue
            if len(sents)<2:
                sents.append('<UNK>/<UNK>/<UNK>')
            for sentno, sent in enumerate(sents):
                tokens = sent.strip().split(' ')
                self.data[sentno].append(tokens[:self.max_len])
        if not self.is_test:
            for line in open(label_path,'r',encoding='utf-8').readlines():
                self.label.append(int(line.strip()))



    def __len__(self):

        assert len(self.data[0])== len(self.data[1])
        return len(self.data[0])
    def __getitem__(self, index):
        print(index)
        tokens = [self.data[0][index],self.data[1][index]]

        word_seq = [[],[]]
        pos_seq = [[],[]]
        ner_seq = [[],[]]
        sent_len = []
        for sentno in range(2):
            for i, token in enumerate(tokens[sentno]):
                v = token.split('/')
                word_seq[sentno].append(self.word2idx.get('/'.join(v[:-2]), 0))
                pos_seq[sentno].append(self.pos2idx.get(v[-2], 0))
                ner_seq[sentno].append(self.ner2idx.get(v[-1], 0))
            sent_len.append(len(word_seq[sentno]))
            #print(len(word_seq[sentno]))
            word_seq[sentno] = np.pad(word_seq[sentno], (self.max_len - sent_len[sentno], 0), 'constant')
            pos_seq[sentno] = np.pad(pos_seq[sentno], (self.max_len - sent_len[sentno], 0), 'constant')
            ner_seq[sentno] = np.pad(ner_seq[sentno], (self.max_len - sent_len[sentno], 0), 'constant')
            #print(len(word_seq[sentno]))
        ret = {'word_seq_question': word_seq[0], 'word_seq_document': word_seq[1],
                'pos_seq_question': pos_seq[0], 'pos_seq_document': pos_seq[1],
                'ner_seq_question': ner_seq[0], 'ner_seq_document': ner_seq[1],
                }
        if not self.is_test:
            label = self.label[index]
            ret['label'] = label

        return ret

if __name__ == '__main__':
    dataset = MyDataSet('data/x_testing.txt','data/y_testing.txt','data/word_dict','data/pos_dict','data/ner_dict')
    dataloader = DataLoader(dataset=dataset, batch_size=19, shuffle=False, drop_last=True)
    count = 0
    for d in dataloader:
        break
        #count+=19
        #print(count)
