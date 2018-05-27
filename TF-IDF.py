from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


document = []
test_data = []


def deal_sentence(sentence):
    sentence_words = sentence.split(' ')
    sentence = ''
    for sentence_word in sentence_words:
        sentence += '/'.join(sentence_word.split('/')[:-2]) + ' '
    return sentence


def deal_data(file_name, test_flag):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        pre_question = ''
        temp_array = []
        flag = False
        for line in lines:
            question = line.split('\t')[0]
            question = deal_sentence(question)
            answer = line.split('\t')[1]
            answer = deal_sentence(answer)
            if question != pre_question:
                if flag:
                    document.append(temp_array)
                    if test_flag:
                        test_data.append(temp_array)
                temp_array = []
                temp_array.append(question)
                temp_array.append(answer)
                pre_question = question
                flag = True
            else:
                temp_array.append(answer)
        document.append(temp_array)
        if test_flag:
            test_data.append(temp_array)


if __name__ == '__main__':
    deal_data('data/x_training.txt', 0)
    deal_data('data/x_testing.txt', 1)
    vectorizer = CountVectorizer(stop_words='english', max_df=0.5)
    temp_document = []
    for d in document:
        temp_document.append(" ".join(d))
    count_sentence = vectorizer.fit_transform(temp_document)

    tfidf = TfidfTransformer()
    tfidf_train = tfidf.fit(count_sentence)
    count_res = []
    print(111)
    count = 0
    for d in test_data:
        temp_res = []
        for sentence in d:
            temp = tfidf_train.transform(vectorizer.transform([sentence]))
            temp_res.append(temp)
            count += 1
        count -= 1
        count_res.append(temp_res)
    print(222)
    print(count)
    f_write = open('tf-idf.txt', 'w')
    for document in count_res:
        question = document[0].toarray()[0]
        for sentence in document[1:]:
            temp = [question, sentence.toarray()[0]]
            simlarity = cosine_similarity(temp)[0][1]
            f_write.write(str(simlarity) + '\n')
    f_write.close()

