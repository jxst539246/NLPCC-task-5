import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_helper import MyDataSet
import sys
from model import *

testset = MyDataSet('data/x_testing.txt','data/y_testing.txt','data/word_dict','data/pos_dict','data/ner_dict',max_len=100,is_test=False)
testloader = DataLoader(dataset=testset, batch_size=19, shuffle=False)


with open('model.pkl', 'rb') as f:
    model = torch.load(f)

DEVICE_NO = 0
if DEVICE_NO != -1:
    model = model.cuda(DEVICE_NO)
log_interval = 500
epochs = 20


def test(dataloader, out=sys.stdout):
    for batch in dataloader:
        if 'label' in batch:
            del batch['label']
        for k in batch:
            batch[k] = Variable(batch[k])
        if DEVICE_NO != -1:
            for k in batch:
                batch[k] = batch[k].cuda(DEVICE_NO)
        pred = model.forward(**batch)
        pred = F.softmax(pred, dim=-1)
        for i in range(19):
           out.write(str(pred[i][-1])+'\n')


test(testloader, out=open('test_output.txt', 'w'))