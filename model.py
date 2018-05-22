#import IPython
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from data_helper import MyDataSet


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.pretrained_word_embedding = nn.Embedding(config['vocab_size'], config['word_embedding_dim'])
        self.untrained_word_embedding = nn.Embedding(config['vocab_size'], config['word_embedding_dim'])
        self.pos_embedding = nn.Embedding(config['pos_set_size'], config['pos_embedding_dim'])
        self.ner_embedding = nn.Embedding(config['ner_set_size'], config['ner_embedding_dim'])
        self.use_pos = config['use_pos']
        self.use_ner = config['use_ner']
        self.lstm_input_dim = config['word_embedding_dim'] * 2
        if self.use_pos:
            self.lstm_input_dim += config['pos_embedding_dim']
        if self.use_ner:
            self.lstm_input_dim += config['ner_embedding_dim']
        self.word_lex_repr_weight = nn.Linear(self.lstm_input_dim,self.lstm_input_dim)
        self.question_repr = nn.LSTM(
            self.lstm_input_dim, config['hidden_size'], config['num_layers'],
            bidirectional=False, batch_first=True, dropout=config['drop_out'])

        self.document_repr = nn.LSTM(
            self.lstm_input_dim, config['hidden_size'], config['num_layers'],
            bidirectional=False, batch_first=True, dropout=config['drop_out'])
        self.dropout = nn.Dropout(config['drop_out'])
        self.output = nn.Linear(config['hidden_size']*2, config['categories'])


    def forward(self, word_seq_question, pos_seq_question, ner_seq_question,
                word_seq_document, pos_seq_document, ner_seq_document):

        # Global Context Representation
        word_question_repr = self.word_repr(word_seq_question, pos_seq_question, ner_seq_question)
        word_document_repr = self.word_repr(word_seq_document, pos_seq_document, ner_seq_document)

        batch_size = word_seq_question.size(0)


        question_hidden = self.init_hidden(batch_size, self.config['num_layers'], self.config['hidden_size'])
        document_hidden = self.init_hidden(batch_size, self.config['num_layers'], self.config['hidden_size'])

        question_output, question_hidden = self.question_repr(word_question_repr, question_hidden)
        document_output, document_hidden = self.question_repr(word_document_repr, document_hidden)

        question_output = self.dropout(question_output.transpose(0, 1)[-1])
        document_output = self.dropout(document_output.transpose(0, 1)[-1])

        mix_repr = torch.cat((question_output,document_output),dim=1)

        output = self.output(mix_repr)
        return output

    def init_hidden(self, batch_size, num_layers, hidden_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layers, batch_size, hidden_size).zero_()),
                Variable(weight.new(num_layers, batch_size, hidden_size).zero_()))

    def init_weights(self, init_range=0.1, pre_trained_filename=None):
        if pre_trained_filename:
            print('Using pre-trained embedding from {}'.format(pre_trained_filename))
            weights = torch.from_numpy(np.load(pre_trained_filename)).type(torch.FloatTensor)
            self.pretrained_word_embedding.weight.data.copy_(weights)
            self.pretrained_word_embedding.weight.requires_grad = False
        else:
            self.pretrained_word_embedding.weight.data.normal_()
        self.untrained_word_embedding.weight.data.normal_()
        self.pos_embedding.weight.data.normal_()
        self.ner_embedding.weight.data.normal_()
        self.output.weight.data.uniform_(-0.1, 0.1)
        self.output.bias.data.fill_(0)

        for name, param in self.question_repr.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

        for name, param in self.document_repr.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)


    def word_repr(self, token_seq, pos_seq, ner_seq):
        pretrained = self.pretrained_word_embedding(token_seq)
        untrained = self.untrained_word_embedding(token_seq)
        word_lex_repr = torch.cat((pretrained, untrained), dim=-1)

        if self.use_pos:
            pos_repr = self.pos_embedding(pos_seq)
            word_lex_repr = torch.cat((word_lex_repr, pos_repr), dim=-1)
        if self.use_ner:
            ner_repr = self.ner_embedding(ner_seq)
            word_lex_repr = torch.cat((word_lex_repr,ner_repr), dim=-1)
        word_lex_repr = F.relu(self.word_lex_repr_weight(word_lex_repr))

        return word_lex_repr


if __name__ == '__main__':
    trainset = MyDataSet('data/x_training.txt', 'data/y_training.txt', 'data/word_dict', 'data/pos_dict',
                         'data/ner_dict', max_len=100, is_test=False)
    trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)

    config = {
        'vocab_size': max(trainloader.dataset.word2idx.values()) + 1,
        'word_embedding_dim': 10,
        'pos_embedding_dim': 5,
        'ner_embedding_dim': 5,
        'pos_set_size': max(trainloader.dataset.pos2idx.values()) + 1,
        'ner_set_size': max(trainloader.dataset.ner2idx.values()) + 1,
        'hidden_size': 10,
        'num_layers': 2,
        'drop_out': 0.3,
        'categories': 2,
        'use_pos': True,
        'use_ner': True
    }
    model = MyModel(config)
    model.init_weights()
    for d in trainloader:
        del d['label']
        for k in d:
            d[k] = Variable(d[k])
        pred = model(**d)
        print(F.softmax(pred, dim=-1))
        break
