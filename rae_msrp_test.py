from models.Recursive_Autoencoder import Full_RAE as RAE
from util.Preprocessing import preprocess
from util.Vectorization import Word_vector
from util.Utility import get_n_feature, dynamic_pooling, similarity_matrix, get_results
from util.Activations import tanh, dtanh, elu, delu
import warnings, numpy as np, pickle, os
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from sklearn import svm


def get_msrp_data(stp):
    train_set = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/train/msr_paraphrase_train"+str(stp)+".pickle",'rb'))
    train_label = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/train/msr_paraphrase_trainscore"+str(stp)+".pickle",'rb'))
    # train_sent = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/train/msr_paraphrase_trainsent"+str(stp)+".pickle",'rb'))
    test_set = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/test/msr_paraphrase_test"+str(stp)+".pickle",'rb'))
    test_label = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/test/msr_paraphrase_testscore"+str(stp)+".pickle",'rb'))
    # test_sent = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/test/msr_paraphrase_testsent"+str(stp)+".pickle",'rb'))
    return train_set, train_label, test_set, test_label
    # return train_set[:100], train_label[:50], train_set[:100], train_label[:50]

def generate_fixed_vector(nn, x, nfeat, pool_size):
    o = []
    for i in range(0, len(x), 2):
        temp1 = nn.predict(x[i])
        temp2 = nn.predict(x[i+1])
        _, s = similarity_matrix(temp1, temp2)
        if nfeat == 1:
            feat = get_n_feature(' '.join([x[i]['words'][j] for j in x[i]['words']]), ' '.join([x[i+1]['words'][j] for j in x[i+1]['words']]))
            o.append(np.concatenate((dynamic_pooling(s, pool_size=pool_size, pf=min).reshape(pool_size * pool_size), feat)))
        else:
            o.append(dynamic_pooling(s, pool_size=pool_size, pf=min).reshape(pool_size * pool_size))
    return o

def init_model(var_file):
    a = pickle.load(open(var_file, 'rb'))
    if len(a) == 5:
        isize, hsize, w, b, g = a
        nn = RAE(input_size=isize, hidden_size=hsize, hidden_activation=tanh, hidden_deactivation=dtanh, output_activation=tanh, output_deactivation=dtanh)
    else:
        isize, hsize, hact, dhact, oact, doact, w, b, g = a
        nn = RAE(input_size=isize, hidden_size=hsize, hidden_activation=hact, hidden_deactivation=dhact, output_activation=oact, output_deactivation=doact)
    nn.w = w
    nn.b = b
    return nn, isize

def test_fun(var_file, pool_size, num_feature, stp, parse_type):

    #create result_directory
    res_dir = '/'.join(var_file.split('/')[:-1]) + '/results/'
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    #initalizing model
    nn, isize = init_model(var_file)
    isize = 50
    wvect = Word_vector(isize, vtype='msrp')

    # getting data that is processed for testing porpose
    train, train_label, test, test_label = get_msrp_data(stp)

    # preprocessing for train and test set
    if os.path.isfile('/'.join(var_file.split('/')[:-1])+'/vectors.pickle'):
        wvect.load_vector('/'.join(var_file.split('/')[:-1])+'/vectors.pickle')
    data_processing = preprocess(parsing_type=parse_type, structure_type='h', stopword=stp, wvect=wvect)
    train_set, _ = data_processing.process_words_data(train)
    test_set, _ = data_processing.process_words_data(test)

    # generating fixed size phrase vector for train and test set
    otrain = generate_fixed_vector(nn, train_set, num_feature, pool_size)
    otest = generate_fixed_vector(nn, test_set, num_feature, pool_size)

    # classifier defination
    # clf = MLPClassifier(activation='logistic', solver='adam',alpha=0.0001,batch_size='auto',learning_rate='adaptive',max_iter=10000,tol=1e-5, verbose=0)
    clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.00001, C=1.0, multi_class='ovr',fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=10000)
    clf.fit(otrain, train_label)
    score = clf.predict(otest)

    # nn classifier
    # train_label = [[0.0, 1.0] if i == 1 else [1.0, 0.0] for i in train_label]
    # i_size, o_size = len(otrain[0]), len(train_label[0])
    # clf = NN(i_size, 2*i_size, o_size, batch_size=50, epoch=200, neta=.0001, op_method='adam', errtol=0.0001)
    # clf.train(zip(otrain, train_label))
    # score = clf.pridect(otest)
    # score = [np.argmax(i) for i in score]

    # getting results
    tp, tn, fp, fn, acc, f1 = get_results(score, test_label)
    print '\npool size : %d,\tnumber feature : %d,\t stopword : %d\n\tTrue positive : %d\n\tTrue negative : %d\n\tFalse positive : %d\n\tFalse negatie : %d\n\taccuracy : %f\n\tf1 score : %f\n'%(pool_size,num_feature,stp,tp, tn, fp, fn,acc, f1)

    # logging result in file
    open(res_dir+'res.txt','a').write('\npool size : %d,\tnumber feature : %d,\t stopword : %d\n\tTrue positive : %d\n\tTrue negative : %d\n\tFalse positive : %d\n\tFalse negatie : %d\n\taccuracy : %f\n\tf1 score : %f\n'%(pool_size,num_feature,stp,tp, tn, fp, fn,acc, f1))

if __name__ == "__main__":
    var_file = ['/media/zero/41FF48D81730BD9B/DT_RAE/IMPLEMENTATION/weights/Full_RAE_50_50_1_rmsprop_dep_h_tanh_tanh/model_variables.pickle'
               ]
    pool_size = 10
    num_feature = [0,1]
    stp = [1]
    parsing_type = ['dep']
    for i in range(len(var_file)):
        for nfeat in num_feature:
           test_fun(var_file[i], pool_size, nfeat, stp[i], parse_type=parsing_type[i])