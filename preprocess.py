from pyltp import *
import os
LTP_DATA_DIR = '/Users/menrui/PycharmProjects/ltp_data'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型
recognizer = NamedEntityRecognizer()  # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

data_list = ['testing']
for data_name in data_list:
    f = open('data/nlpcc-iccpol-2016.dbqa.'+data_name+'-data','r',encoding='utf-8')
    x_output = open('data/x_' + data_name + '.txt', 'w', encoding='utf-8')
    y_output = open('data/y_' + data_name + '.txt', 'w', encoding='utf-8')
    count = 0
    for line in f.readlines():
        count+=1
        if count% 100 == 0:
            print(count)
        line = line.strip().split('\t')
        sents = line[:2]
        if len(line)>2:
            label = line[-1]
            y_output.write(label+'\n')
        for sent in sents:
            words = segmentor.segment(sent)  # 分词
            postags = postagger.postag(words)
            netags = recognizer.recognize(words,postags)
            for i,word in enumerate(words):
                x_output.write(word+'/'+postags[i]+'/'+netags[i]+' ')
            x_output.write('\t')
        x_output.write('\n')

    f.close()
    x_output.close()
    y_output.close()

segmentor.release()
postagger.release()
recognizer.release()