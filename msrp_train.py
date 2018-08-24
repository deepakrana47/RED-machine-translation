from models.Stacked_Recursive_Autoencoder import stack_RAE
from util.Preprocessing import preprocess
from util.Vectorization import Word_vector
from util.optimization import Optimization_variable, Optimization
from util.Utility import get_n_feature, mini_batch, dynamic_pooling, similarity_matrix, get_results
from models.summation import sum_vector
from util.Activations import ce_erro, tanh, dtanh
from util.Logger import logger
import warnings, numpy as np, pickle, os
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from models.Neural_layer import NN


def get_msrp_data(stp):
    train_set = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/train/msr_paraphrase_train"+str(stp)+".pickle",'rb'))
    train_label = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/train/msr_paraphrase_trainscore"+str(stp)+".pickle",'rb'))
    # train_sent = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/train/msr_paraphrase_trainsent"+str(stp)+".pickle",'rb'))
    test_set = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/test/msr_paraphrase_test"+str(stp)+".pickle",'rb'))
    test_label = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/test/msr_paraphrase_testscore"+str(stp)+".pickle",'rb'))
    # test_sent = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/MSRP/test/msr_paraphrase_testsent"+str(stp)+".pickle",'rb'))
    return train_set, train_label, test_set, test_label
    # return train_set[:40], train_label[:20], test_set[:10], test_label[:5]

def generate_fixed_vector(nn, x, nfeat, pool_size):
    o = []
    for i in range(0, len(x), 2):
        # temp1 = nn.predict(x[i])
        # temp2 = nn.predict(x[i+1])
        temp1 = sum_vector(x[i])
        temp2 = sum_vector(x[i+1])
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
        nn = stack_RAE(input_size=isize, hidden_size=hsize)
    else:
        isize, hsize, hact, dhact, oact, doact, w, b, g = a
        nn = stack_RAE(input_size=isize, hidden_size=hsize, hidden_activation=hact, hidden_deactivation=dhact, output_activation=oact, output_deactivation=doact)
    nn.w = w
    nn.b = b
    return nn, hsize

def msrp_test(data, test):
    # testing of model
    score = []
    rae_layer, nn_layer = data
    for data in test:
        # forward pass of RAE
        vect1, v1 = rae_layer.encoding(data[0])
        vect2, v2 = rae_layer.encoding(data[1])

        # forward pass of NN
        o, _ = nn_layer.forward_pass(np.concatenate((vect1[len(vect1) - 1], vect2[len(vect2) - 1]), axis=0))
        score.append(o)
    return [np.argmax(i) for i in score]

def msrp_train_test(var_file, stp, parse_type):

    #create result_directory
    res_dir = '/'.join(var_file.split('/')[:-1]) + '/results/'
    log_file = '/'.join(var_file.split('/')[:-1]) + '/msrp_train_log.txt'
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)

    # getting data that is processed for testing porpose
    train, train_label, test, test_label = get_msrp_data(stp)

    #initalizing model
    isize = 50
    osize = 2
    wvect = Word_vector(isize, vtype='msrp')
    # a.optimum.opt_var.g=g

    # preprocessing for train and test set
    data_processing = preprocess(parsing_type=parse_type, structure_type='h', stopword=stp, wvect=wvect)
    train_set, _ = data_processing.process_words_data(train)
    test_set, _ = data_processing.process_words_data(test)
    train_label = [np.array([[0.0], [1.0]]) if i == 1 else np.array([[1.0], [0.0]]) for i in train_label]
    train = [[train_set[i], train_set[i+1]] for i in range(0,len(train_set),2)]
    test = [[test_set[i], test_set[i+1]] for i in range(0,len(test_set),2)]


    # model layers initalization
    isize, hsize, w, b, g = pickle.load(open(var_file, 'rb'))
    opt_var = Optimization_variable(method='rmsprop', isize=isize, osize=hsize, model_type='RAE', option={'wpresent': []})
    opt = Optimization(optimization_variable=opt_var, learning_rate=0.0001)
    rae_layer = stack_RAE(input_size=isize, hidden_size=hsize, optimization=opt, hidden_activation=tanh, hidden_deactivation=dtanh)
    rae_layer.w = w
    rae_layer.b = b
    rae_layer.optimum.opt_var.g = g
    rae_layer.init_dw(w,b)
    nn_layer = NN(2*hsize, osize)
    nn_layer.optimum = Optimization(optimization_variable=Optimization_variable(method='rmsprop', isize=2*hsize, osize=osize), learning_rate=0.0001)
    logg = logger('text', log_file)


    # trainning of model
    epoch = 10000
    batch_size = 50
    for ep in range(epoch):
        batches = mini_batch(zip(train, train_label), len(train_set), batch_size)
        cost = 0.0; ecount = 0
        for batch in range(len(batches)):
            for data in batches[batch]:
                # forward pass of RAE
                vect1, v1 =rae_layer.encoding(data[0][0])
                vect2, v2 =rae_layer.encoding(data[0][1])
                data[0][0]['vects'] = vect1
                data[0][1]['vects'] = vect2

                # forward pass of NN
                o, vnn = nn_layer.forward_pass(np.concatenate((vect1[len(vect1)-1], vect2[len(vect2)-1]), axis=0))

                # cost calculation
                cost += ce_erro(o, data[1])
                ecount +=1
                grad = o-data[1]

                # backward pass of NN
                nngrad = nn_layer.backward_pass(grad, vnn)

                # backward padd of RAE
                nngrad1, nngrad2 = np.split(nngrad,[hsize], axis=0)
                grad1 = rae_layer.encoding_back(nngrad1, v1, data[0][0])
                grad2 = rae_layer.encoding_back(nngrad2, v2, data[0][1])

                # calculating weight update
                nn_layer.calc_dw(grad, np.concatenate((vect1[len(vect1)-1], vect2[len(vect2)-1]),axis=0))
                rae_layer.calc_dw(grad1, data[0][0])
                rae_layer.calc_dw(grad2, data[0][1])

            # updating weights
            nn_layer.update_weights()
            rae_layer.update_weights()
        if (ep+1)%50 == 0:
            score = msrp_test([rae_layer, nn_layer], test)
            tp, tn, fp, fn, acc, f1 = get_results(score, test_label)
            print 'stopword : %d, Tp : %d, Tn : %d, Fp : %d Fn : %d,acc : %ff1 score : %f'%(stp, tp, tn, fp, fn, acc, f1)
            logg.log_text('stopword : %d, Tp : %d, Tn : %d, Fp : %d Fn : %d, acc : %f, f1 score : %f'%(stp, tp, tn, fp, fn, acc, f1))
            pickle.dump([ rae_layer.w, rae_layer.b,nn_layer.w], open('/'.join(var_file.split('/')[:-1]) + '/results/weights'+str(ep)+'.pickle','wb'))

        print "%d/%d epoch completed .... error : %f" % (ep + 1, epoch, cost/ecount)
        logg.log_text("%d/%d epoch completed ....\n" % (ep + 1, epoch))
        if cost/ecount < 0.01:
            break

    # getting results
    score = msrp_test([rae_layer, nn_layer], test)
    tp, tn, fp, fn, acc, f1 = get_results(score, test_label)
    print '\nstopword : %d\n\tTrue positive : %d\n\tTrue negative : %d\n\tFalse positive : %d\n\tFalse negatie : %d\n\taccuracy : %f\n\tf1 score : %f\n'%(stp,tp, tn, fp, fn,acc, f1)

    # logging result in file
    open(res_dir+'res.txt','a').write('\nstopword : %d\n\tTrue positive : %d\n\tTrue negative : %d\n\tFalse positive : %d\n\tFalse negatie : %d\n\taccuracy : %f\n\tf1 score : %f\n'%(stp,tp, tn, fp, fn,acc, f1))

if __name__ == "__main__":
    # var_file = ['/media/zero/41FF48D81730BD9B/DT_RAE/IMPLEMENTATION/weights/rae_50_rmsprop_dep_h_elu_tanh/model_variable.pickle']
    var_file = ['/media/zero/41FF48D81730BD9B/DT_RAE/IMPLEMENTATION/weights/RAE_50_50_1_3_rmsprop_syn_h_elu_tanh/model_variables.pickle']
    # pool_size = 10
    # num_feature = [0,1]
    stp = [1]
    parsing_type = ['syn']
    for i in range(len(var_file)):
        msrp_train_test(var_file[i], stp[i], parse_type=parsing_type[i])