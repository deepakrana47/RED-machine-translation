import pickle, os, sys
from random import shuffle
import warnings
from util.Activations import *
from util.optimization import Optimization_variable, Optimization

log_file = open('log.txt','w')
warnings.filterwarnings("error")

def unzip(data):
    x=[]
    y=[]
    for i in data:
        x.append(i[0])
        y.append(i[1])
    return x, y

class Util(object):

    def __init_(self):
        pass

    # weight initalization function using gauss normal mean=0 and varience=0.1
    def init_weight(self, size1, size2, dropout=0.0):
        # r = np.random.rand(size1, size2)
        m = np.random.normal(0, 0.1, (size1, size2))
        # mask = np.where(r < dropout, 0.0, 1.0)
        return m #, mask

    def shuff(self, t, l):
        n = len(t)
        a = [(t[i], l[i]) for i in range(n)]
        shuffle(a)
        return a

    def mini_batch(self, data, batch_size):
        batches = []
        shuffle(data)
        data_size = len(data)
        i = 0
        for i in range(1, data_size / batch_size):
            batches.append(data[(i - 1) * batch_size: i * batch_size])
        if data_size % batch_size != 0:
            batches.append(data[i * batch_size: data_size])
        return batches

class NN:

    def __init__(self, isize, hsize, osize, batch_size=100, epoch=5, neta=0.0001, op_method='rmsprop', errtol=0.001):
        self.util = Util()
        self.isize = isize
        self.hsize = hsize
        self.osize = osize
        self.wi2h1 = self.util.init_weight(hsize, isize)
        self.wh12o = self.util.init_weight(osize, hsize)

        self.neta = neta
        self.batch_size = batch_size
        self.epoch = epoch
        self.errtol = errtol

        self.optimum_i2h = Optimization(optimization_variable=Optimization_variable(method=op_method, i_size=isize, o_size=hsize), method=op_method, learning_rate=neta)
        self.optimum_h2o = Optimization(optimization_variable=Optimization_variable(method=op_method, i_size=hsize, o_size=osize), method=op_method, learning_rate=neta)

    def save_weights(self, wfname):
        pickle.dump([self.wi2h1, self.wh12o], open(wfname, 'wb'))

    def load_weights(self, wfname):
        self.wi2h1, self.wh12o = pickle.load(open(wfname, 'rb'))

    def forward_pass(self, x):

        # input to hidden1 activation
        self.vx = np.dot(self.wi2h1, x)
        self.h1 = relu(self.vx)

        # hidden to output activation
        o = softmax(np.dot(self.wh12o, self.h1))
        return o

    def backward_pass(self, o, t):

        # gradient at output
        g_o2h1 = o - t

        # gradient at hidden
        g_h12i = np.dot(self.wh12o.T, g_o2h1) * drelu(self.vx)
        return g_o2h1, g_h12i

    # return generated output for given input
    def pridect(self, xs):
        o = []
        for x in xs:
            x = np.array(x).reshape((self.isize, 1))
            h1 = relu(np.dot(self.wi2h1, x))
            o.append(softmax(np.dot(self.wh12o, h1)))
        return o

    def train(self, d):
        err = 0.0
        epoch_count = 0
        count = 0
        dw_h12o = np.zeros(self.wh12o.shape)
        dw_i2h1 = np.zeros(self.wi2h1.shape)

        while 1:

            tdata = self.util.mini_batch(d, self.batch_size)

            for i in range(len(tdata)):
                for data in tdata[i]:
                    x = np.array(data[0]).reshape((self.isize,1))
                    y = np.array(data[1]).reshape((self.osize,1))

                    ## feedforward
                    o=self.forward_pass(x)

                    #error calculation
                    t0 = ce_erro(o, y)
                    err += t0
                    count +=1

                    ## backpropogation
                    g_o2h1, g_h12i = self.backward_pass(o, y)

                    # weight change values
                    dw_h12o += np.dot(g_o2h1, self.h1.T)
                    dw_i2h1 += np.dot(g_h12i, x.T)

                # weight are updated
                self.wh12o, _ = self.optimum_h2o.update(self.wh12o, dw_h12o / len(tdata[i]))
                self.wi2h1, _= self.optimum_i2h.update(self.wi2h1, dw_i2h1 / len(tdata[i]))

                # if i%10 == 0:
                #     print "%d/%d batches are complete...... error : %f"%(i, len(tdata), err/count)
            if epoch_count > self.epoch or  err/count < self.errtol:
                break
            print "%d epoch error is %f" % (epoch_count, err/count)
            err, count = 0.0, 0

            # print "%d/%d epoch complete..." % (epoch_count, self.epoch)
            epoch_count += 1
        return

def test(x, y, nn, log=0):
    correct = 0.0
    total = len(x)
    for i in range(total):
        o = nn.pridect(x[i])
        t1 = np.argmax(o)
        if np.argmax(y[i]) == t1:
            correct +=1
    print "accuracy : %f" % (correct/float(len(x)))
    if log:
        log_file.write("accuracy : %f\n" % (correct/float(len(x))))

def usage():
    print "\nUsage:"
    print "\tpython nn_git.py [filename]"

# if __name__=="__main__":
#
#     if sys.argv[1] == '-h' or sys.argv[1] == '--h' or  sys.argv[1] == '-help':
#         usage()
#
#     train_d, test_d = make_dataset(sys.argv[1])
#     print "Training data size : %d\nTesting data size : %d"%(len(train_d),len(test_d))
#
#     test_data, test_label = unzip(test_d)
#
#     i_size = len(test_data[0])
#     h1_size = 2*i_size
#     o_size = len(test_label[0])
#     method = 'rmsprop'
#
#     print "\nNeural Network parameters"
#     print "\tInput size : %d\n\tHidden layer size : %d\n\tOutput layer size : %d\n\tMethod : %s"%(i_size,h1_size,o_size,method)
#     # if raw_input("q to discontinue : ") == 'q':
#     #     exit()
#
#     nn = NN(i_size, h1_size, o_size, batch_size=100, epoch=2, neta=.0001, op_method=method)
#     # if os.path.isfile('weights.pickle'):
#     #     nn.load_weights('weights.pickle')
#     nn.train(train_d)
#     nn.save_weights('weights.pickle')
#     test(test_data, test_label, nn, log=1)