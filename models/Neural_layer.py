import pickle
import warnings
from util.Activations import *
from util.Utility import init_weight

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

class NN:

    def __init__(self, isize, osize, optimization=None, activation=softmax, deactivation=dsoftmax):
        self.isize = isize
        self.osize = osize
        self.optimum = optimization
        self.activ = activation
        self.dactiv = deactivation
        self.w = init_weight(osize, isize)
        self.dw = np.zeros(self.w.shape)
        self.count = 0

    def save_weights(self, wfname):
        pickle.dump(self.w, open(wfname, 'wb'))

    def load_weights(self, wfname):
        self.w = pickle.load(open(wfname, 'rb'))

    def forward_pass(self, x):
        vx = np.dot(self.w, x)
        o = self.activ(vx)
        return o, vx

    def backward_pass(self, igrad, vx, inp):
        ograd = np.dot(self.w.T, igrad) * self.dactiv(vx)
        self.dw += np.dot(igrad, inp.T)
        self.count += 1
        return ograd

    def calc_dw(self, grad, inp):

        return

    def update_weights(self):
        self.w, _ = self.optimum.update(self.w, self.dw/self.count)
