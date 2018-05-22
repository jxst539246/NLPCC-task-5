import json
import numpy as np

wv = open('sgns.zhihu.word','r',encoding='utf-8')
wv.readline()
wv_dict = {}
for line in wv.readlines():
    line = line.strip().split()
    word = line[0]
    weight = []
    zero_vec = np.zeros(300)
    wv_dict[word] = zero_vec
    for i,item in enumerate(line[1:]):
        wv_dict[word][i] = float(item)

print('load word2vec end')
_word_set = sorted(json.load(open('data/word_dict')).keys())
idx2word = {i + 1: w for i, w in enumerate(_word_set)}

weights = np.zeros((max(idx2word.keys()) + 1, 300))
oov_num = 0
for i in range(1, max(idx2word.keys()) + 1):
    if idx2word[i] in wv_dict:
        weights[i, :] = wv_dict[idx2word[i]]
    else:
        oov_num += 1
np.save('weights.pkl', weights)
print('total:{} oov:{}'.format(weights.shape[0], oov_num))
