from util.Preprocessing import preprocess
import pickle
word_data = pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/msr_paraphrase_train.pickle', 'rb'))
data_processing = preprocess(parsing_type='syn', structure_type='nh', stopword=1)
data1, _ = data_processing.process_words_data(word_data)

data_processing = preprocess(parsing_type='chk', structure_type='h', stopword=0)
data2, _ = data_processing.process_words_data(word_data)

for i in range(len(data1)):
    words = data1[i]['words']
    wlen = len(words)
    for j in data1[i]['h_vect']:
        print [words[k] for k in j],
        words[wlen] = ' '.join([words[k] for k in j])
        wlen+=1
    print
    words = data2[i]['words']
    for j in data2[i]['h_vect']:
        print [words[k] for k in j],
    print
    raw_input()