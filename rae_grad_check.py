import pickle
# from models.Recursive_Autoencoder import Full_RAE as RAE
from models.Stacked_Recursive_Autoencoder import stack_RAE as RAE
from util.optimization import Optimization, Optimization_variable
from util.Preprocessing import preprocess
from util.Activations import *
from util.Vectorization import Word_vector
import os

def print_setting(args):
    print "\n\nTrainning Parameter:\n neta : %f \n word vector size :%d \n pharse vector size :%d " \
          "\n batch size :%d \n optimization method :%s \n model :%s \n Stopword :%d \n parse-type :%s " \
          "\n type :%s \nhidden activation:%s \noutput activation:%s \n epoch :%d \n Input flag :%s \n " \
          "Input file :%s \n save model file :%s\n"%\
          (args['neta'], args['v_size'], args['h_size'], args['batch_size'], args['method'],args['model'],
           args['stp'], args['parse-type'], args['type'], args['hactiv'][0].__name__, args['oactiv'][0].__name__,
           args['epoch'], args['flag'], args['src'], args['wfname'])

def log_data(args):
    # wt = args['wfname'].split('/')[-1]
    ddir = args['wfname']
    if not os.path.isdir(ddir):
        os.mkdir(ddir)
    open(ddir + '/setting.txt', 'w').write("\n\nTrainning Parameter:\n neta : %f \n word vector size :%d \n pharse vector size :%d \n batch size :%d \n optimization method :%s \n model :%s \n Stopword :%d \n parse-type :%s \n type :%s \n epoch :%d \n Input flag :%s \n Input file :%s \n save model file :%s"% (args['neta'], args['v_size'], args['h_size'], args['batch_size'], args['method'],args['model'], args['stp'], args['parse-type'], args['type'], args['epoch'], args['flag'], args['src'], args['wfname']))
    wfname = ddir + "/model_variables.pickle"
    log = {}
    log['iter'] = ddir + '/iter_count.pickle'
    log['epoch'] = ddir + '/epoch.pickle'
    log['iter_err'] = ddir + '/iter_err_count.pickle'
    log['rae'] = ddir + '/log.txt'
    return wfname, log

def create_model(wfname, wload, model_type, op_method, i_size, o_size, neta, wpresent, logg, vector):
    opt_var = Optimization_variable(method=op_method, isize=i_size, osize=o_size, model_type=model_type, option={'wpresent':wpresent})
    opt = Optimization(optimization_variable=opt_var, learning_rate=neta)

    a = pickle.load(open(wload, 'rb')) if wload else pickle.load(open(wfname, 'rb'))
    if len(a) == 5:
        i_size, h_size, w, b, g = a
        a = RAE(input_size=i_size, hidden_size=h_size, optimization=opt,
                      wpresent=wpresent, log_file=logg, log=1, vector=vector)
    else:
        i_size, h_size, hact, dhact, oact, doact, w, b, g = a
        a = RAE(input_size=i_size, hidden_size=h_size, optimization=opt,
                      hidden_activation=hact, hidden_deactivation=oact,
                      output_activation=dhact, output_deactivation=doact,
                      wpresent=wpresent, log_file=logg, log=1, vector=vector)
    a.w, a.b, a.optimum.opt_var.g = w, b, g
    return a

def main_train(args):

    print_setting(args)
    wfname, log = log_data(args)

    words_data = pickle.load(open("/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/110.pickle", 'rb'))
    # words_data = pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/msr_paraphrase_train.pickle', 'rb')) \
    #              + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/bbc.pickle', 'rb')) \
                 # + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/1all-news.pickle', 'rb')) \
                 # + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/2all-news.pickle', 'rb')) \
                 # + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/3all-news.pickle', 'rb'))
                 # + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/4all-news.pickle', 'rb')) \
                 # + pickle.load(open('/media/zero/41FF48D81730BD9B/DT_RAE/data/pickle/5all-news.pickle', 'rb'))

    wvect = Word_vector(args['v_size'])
    data_processing = preprocess(parsing_type=args['parse-type'], structure_type=args['type'], stopword=args['stp'], wvect=wvect)
    data, wpresent = data_processing.process_words_data(words_data)
    nn = create_model(wfname, args['wload'], args['model'], args['method'], args['v_size'], args['h_size'], neta=args['neta'], wpresent=wpresent, logg = log['rae'], vector=wvect)
    nn.gradient_check(data, sorted(wpresent))
    return

if __name__ == '__main__':
    args = {
        'neta': 0.001,
        'v_size': 50,
        'h_size': 50,
        'batch_size': 100,
        'method': 'rmsprop',
        'model': 'RAE',
        'stp': 1,
        'parse-type': 'dep',
        'type': 'h',
        'epoch': 3,
        'hactiv': [elu, delu],
        'oactiv': [tanh, dtanh],
        'flag': 'd',
        'src': '',
        'wload': '',
        'wfname': '/media/zero/41FF48D81730BD9B/DT_RAE/IMPLEMENTATION/weights/1stack_',
    }
    n = "%s_%d_%d_%d_%s_%s_%s_%s_%s" % (
    args['model'], args['v_size'], args['h_size'], args['stp'], args['method'], args['parse-type'], args['type'],
    args['hactiv'][0].__name__, args['oactiv'][0].__name__)
    args['wfname'] += n
    main_train(args)
    exit()