import numpy as np

class Optimization:

    def __init__(self, optimization_variable, learning_rate=0.001, regularization=0.0001, options=None, learning_decay=0.00005, learning='adaptive'):
        self.neta = learning_rate
        self.opt_var = optimization_variable
        self.regu = regularization
        if self.opt_var.method == 'adam':
            self.update = self.adam
            if options:
                self.b1, self.b2, self.e = options['b1'], options['b2'], options['e']
            else:
                self.b1, self.b2, self.e = 0.9, 0.99, 1e-8

        elif self.opt_var.method == 'rmsprop':
            self.update = self.rmsprop
            if options:
                self.b1, self.e = options['b1'], options['e']
            else:
                self.b1, self.e = 0.99, 1e-8

        elif self.opt_var.method == 'sgd':
            self.update = self.sgd
            self.learning_decay = learning_decay
            self.learning = learning

        else :
            print "Optimization method name is not correct !!!"
            exit()

    def adam(self, w, dw, b=None, db=None, extra=None):
        m, v = self.opt_var.getm(extra), self.opt_var.getv(extra)
        m = (self.b1 * m) + ((1. - self.b1) * dw)
        v = (self.b2 * v) + ((1. - self.b2) * np.square(dw))
        m_h = m / (1. - self.b1)
        v_h = v / (1. - self.b2)
        # w -= neta * (m_h/(np.sqrt(v_h) + e) + regu * w)
        w -= self.neta * m_h / (np.sqrt(v_h) + self.e)
        self.opt_var.setm(m, extra), self.opt_var.setv(v, extra)
        if b is not None:
            b -= self.neta * db
        return w, b

    def rmsprop(self, w, dw, b=None, db=None, extra=None):
        m = self.opt_var.getm(extra)
        m = self.b1*m + (1 - self.b1)*np.square(dw)
        w -= self.neta * np.divide(dw, (np.sqrt(m) + self.e))
        self.opt_var.setm(m ,extra)
        if b is not None:
            b -= self.neta * db
        return w, b

    def sgd(self, w, dw, b=None, db=None, extra=None):
        w -= self.neta*(dw)# + regu*(w))
        if b is not None:
            b -= self.neta * (db)
        if self.learning == 'adaptive':
            self.neta -= self.learning_decay

        return w, b

class Optimization_variable:
    def __init__(self, method, isize, osize, model_type=None, option=None):
        self.g = {}
        self.isize = isize
        self.osize = osize
        self.method = method
        self.model_type = model_type
        self.option = option
        if method == 'adam':
            self.init_adam_var()
        elif method == 'rmsprop':
            self.init_rmsprop_var()

    def init_adam_var(self):
        if self.model_type == 'RAE':
            self.g[0] = {'e': {}, 'd': {}}
            self.g[1] = {'e': {}, 'd': {}}
            wpresent =  self.option['wpresent']
            for i in wpresent:
                self.g[0]['e'][i] = np.zeros((self.osize, self.osize))
                self.g[0]['d'][i] = np.zeros((self.osize, self.osize))
                self.g[1]['e'][i] = np.zeros((self.osize, self.osize))
                self.g[1]['d'][i] = np.zeros((self.osize, self.osize))

            for i in [0, 0.1]:
                self.g[0]['e'][i] = np.zeros((self.osize, self.isize))
                self.g[0]['d'][i] = np.zeros((self.isize, self.osize))
                self.g[1]['e'][i] = np.zeros((self.osize, self.isize))
                self.g[1]['d'][i] = np.zeros((self.isize, self.osize))
        elif self.model_type == 'GRU':
            self.g[0] = {}
            self.g[1] = {}
            self.hsize =  self.option['hsize']
            self.g[0]['ur'] = np.zeros((self.hsize, self.isize))
            self.g[1]['ur'] = np.zeros((self.hsize, self.isize))

            self.g[0]['wr'] = np.zeros((self.hsize, self.hsize))
            self.g[1]['wr'] = np.zeros((self.hsize, self.hsize))

            self.g[0]['uz'] = np.zeros((self.hsize, self.isize))
            self.g[1]['uz'] = np.zeros((self.hsize, self.isize))

            self.g[0]['wz'] = np.zeros((self.hsize, self.hsize))
            self.g[1]['wz'] = np.zeros((self.hsize, self.hsize))

            self.g[0]['u_h'] = np.zeros((self.hsize, self.isize))
            self.g[1]['u_h'] = np.zeros((self.hsize, self.isize))

            self.g[0]['w_h'] = np.zeros((self.hsize, self.hsize))
            self.g[1]['w_h'] = np.zeros((self.hsize, self.hsize))

            self.g[0]['wo'] = np.zeros((self.osize, self.hsize))
            self.g[1]['wo'] = np.zeros((self.osize, self.hsize))
        else:
            self.g[0] = np.zeros((self.osize, self.isize))
            self.g[1] = np.zeros((self.osize, self.isize))
        return

    def init_rmsprop_var(self):

        if self.model_type == 'RAE':
            self.g[0] = {'e': {}, 'd': {}}
            wpresent =  self.option['wpresent']
            for i in wpresent:
                self.g[0]['e'][i] = np.zeros((self.osize, self.osize))
                self.g[0]['d'][i] = np.zeros((self.osize, self.osize))

            for i in [0, 0.1]:
                self.g[0]['e'][i] = np.zeros((self.osize, self.isize))
                self.g[0]['d'][i] = np.zeros((self.isize, self.osize))
        elif self.model_type == 'GRU':
            self.g[0] = {}
            self.hsize =  self.option['hsize']
            self.g[0]['ur'] = np.zeros((self.hsize, self.isize))
            self.g[0]['wr'] = np.zeros((self.hsize, self.hsize))

            self.g[0]['uz'] = np.zeros((self.hsize, self.isize))
            self.g[0]['wz'] = np.zeros((self.hsize, self.hsize))

            self.g[0]['u_h'] = np.zeros((self.hsize, self.isize))
            self.g[0]['w_h'] = np.zeros((self.hsize, self.hsize))

            self.g[0]['wo'] = np.zeros((self.osize, self.hsize))
        else:
            self.g[0] = np.zeros((self.osize, self.isize))
        return

    def getm(self, extra):
        if  self.model_type == 'RAE':
            return self.g[0][extra[0]][extra[1]]
        elif  self.model_type == 'GRU':
            return self.g[0][extra[0]]
        else:
            return self.g[0]

    def getv(self, extra):
        if  self.model_type == 'RAE':
            return self.g[1][extra[0]][extra[1]]
        elif  self.model_type == 'GRU':
            return self.g[1][extra[0]]
        else:
            return self.g[1]

    def setm(self, m, extra):
        if  self.model_type == 'RAE':
            self.g[0][extra[0]][extra[1]] = m
        elif  self.model_type == 'GRU':
            self.g[0][extra[0]]=m
        else:
            self.g[0] = m

    def setv(self, v, extra):
        if  self.model_type == 'RAE':
            self.g[1][extra[0]][extra[1]] = v
        elif  self.model_type == 'GRU':
            self.g[1][extra[0]]=v
        else:
            self.g[1] = v