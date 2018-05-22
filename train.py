import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_helper import MyDataSet
from model import *

trainset = MyDataSet('data/x_training.txt','data/y_training.txt','data/word_dict','data/pos_dict','data/ner_dict',max_len=100,is_test=False)
trainloader = DataLoader(dataset=trainset, batch_size=16, shuffle=True)
testset = MyDataSet('data/x_testing.txt','data/y_testing.txt','data/word_dict','data/pos_dict','data/ner_dict',max_len=100,is_test=False)
testloader = DataLoader(dataset=testset, batch_size=16, shuffle=False)

config = {
    'vocab_size': max(trainloader.dataset.word2idx.values()) + 1,
    'word_embedding_dim': 100,
    'pos_embedding_dim': 25,
    'ner_embedding_dim': 25,
    'pos_set_size': max(trainloader.dataset.pos2idx.values()) + 1,
    'ner_set_size': max(trainloader.dataset.ner2idx.values()) + 1,
    'hidden_size': 100,
    'num_layers': 2,
    'drop_out': 0.3,
    'categories': 2,
    'use_pos': True,
    'use_ner': True
}

model = MyModel(config)

DEVICE_NO = 1
if DEVICE_NO != -1:
    model = model.cuda(DEVICE_NO)

optimizer = torch.optim.Adam(model.parameters())
criteria = nn.CrossEntropyLoss()

log_interval = 500
epochs = 20


def train(dataloader):
    model.train()
    total_loss = 0
    total_items = 0
    start_time = time.time()
    for i_batch, batch in enumerate(dataloader):
        output_seq = Variable(batch['label'])
        del (batch['label'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if DEVICE_NO != -1:
            output_seq = output_seq.cuda(DEVICE_NO)
            for k in batch:
                batch[k] = batch[k].cuda(DEVICE_NO)
        model.zero_grad()
        pred = model.forward(**batch)
        pred = pred.view(-1, pred.size(-1))
        output_seq = output_seq.view(-1)
        loss = criteria(pred, output_seq)
        loss.backward()
        num_items = len([x for x in output_seq])
        total_loss += num_items * loss.data
        total_items += num_items
        optimizer.step()

        if i_batch % log_interval == 0 and i_batch > 0:
            cur_loss = total_loss[0] / total_items
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:04.4f} | ms/batch {:5.2f} | '
                  'loss {:5.6f}'.format(
                epoch, i_batch, len(dataloader.dataset) // dataloader.batch_size, optimizer.param_groups[0]['lr'],
                                elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            total_items = 0
            start_time = time.time()


def evaluate(dataloader):
    total_loss = 0
    total_items = 0
    model.eval()
    for batch in dataloader:
        output_seq = Variable(batch['label'])
        del (batch['label'])
        for k in batch:
            batch[k] = Variable(batch[k])
        if DEVICE_NO != -1:
            output_seq = output_seq.cuda(DEVICE_NO)
            for k in batch:
                batch[k] = batch[k].cuda(DEVICE_NO)
        pred = model.forward(**batch)
        pred = pred.view(-1, pred.size(-1))
        output_seq = output_seq.view(-1)
        num_items = len([x for x in output_seq ])
        total_loss += num_items * criteria(pred, output_seq).data
        total_items += num_items

    return total_loss[0] / total_items


best_val_loss = 1000
try:
    print(model)
    model.init_weights()
    for epoch in range(1, epochs + 1):
        # scheduler.step()
        epoch_start_time = time.time()
        train(trainloader)
        val_loss = evaluate(testloader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.6f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss,))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print('new best val loss, saving model')
            with open('model.pkl', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            pass
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    val_loss = evaluate(testloader)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        print('new best val loss, saving model')
        with open('model.pkl', 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        pass
