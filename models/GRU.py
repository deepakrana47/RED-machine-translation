import numpy as np, pickle
from util.Activations import sigmoid, dsigmoid, tanh, dtanh
class gru:
    def __init__(self, isize, hsize, osize, activation=sigmoid, deactivation=dsigmoid, optimization=None):

        self.isize = isize
        self.hsize = hsize
        self.osize = osize
        self.optimum = optimization

        # self.names = {'ur':0,'wr':1, 'uz':2, 'wz':3, 'u_h':4, 'w_h':5, 'wo':6}
        self.names = {0: 'ur', 1: 'wr', 2: 'uz', 3: 'wz', 4: 'u_h', 5: 'w_h', 6: 'wo'}
        self.w = {}
        self.b = {}
        self.dw = {}
        self.db = {}

        # reset weights
        self.w['ur'] = np.random.normal(0, 0.1, (hsize, isize))
        self.b['r'] = np.zeros((hsize, 1))
        self.w['wr'] = np.random.normal(0, 0.1, (hsize, hsize))
        self.dw['ur'] = np.zeros((self.hsize, self.isize))
        self.db['r'] = np.zeros((self.hsize, 1))
        self.dw['wr'] = np.zeros((self.hsize, self.hsize))

        # update weights
        self.w['uz'] = np.random.normal(0, 0.1, (hsize, isize))
        self.b['z'] = np.zeros((hsize, 1))
        self.w['wz'] = np.random.normal(0, 0.1, (hsize, hsize))
        self.dw['uz'] = np.zeros((self.hsize, self.isize))
        self.db['z'] = np.zeros((self.hsize, 1))
        self.dw['wz'] = np.zeros((self.hsize, self.hsize))

        # _h weights
        self.w['u_h'] = np.random.normal(0, 0.1, (hsize, isize))
        self.b['_h'] = np.zeros((hsize, 1))
        self.w['w_h'] = np.random.normal(0, 0.1, (hsize, hsize))
        self.dw['u_h'] = np.zeros((self.hsize, self.isize))
        self.db['_h'] = np.zeros((self.hsize, 1))
        self.dw['w_h'] = np.zeros((self.hsize, self.hsize))

        # out weight
        self.w['wo'] = np.random.normal(0, 0.1, (osize, hsize))
        self.b['o'] = np.zeros((osize, 1))
        self.dw['wo'] = np.zeros((osize, hsize))
        self.db['o'] = np.zeros((osize, 1))

        self.count = 0.0

        # output activation
        self.activation = activation
        self.deactivation = deactivation

    def forward_pass(self, inputs):
        v = { 'r' : [], 'z' : [], '_h' : [], 'h' : {-1:np.zeros((self.hsize, 1))}, 'vr':[], 'vz':[], 'v_h':[], 'vo':[]}
        o = []
        for i in range(len(inputs)):
            # calculating reset gate value
            v['vr'].append(np.dot(self.w['ur'], inputs[i]) + np.dot(self.w['wr'], v['h'][i - 1]) + self.b['r'])
            v['r'].append(sigmoid(v['vr'][i]))

            # calculation update gate value
            v['vz'].append(np.dot(self.w['uz'], inputs[i]) + np.dot(self.w['wz'], v['h'][i - 1]) + self.b['z'])
            v['z'].append(sigmoid(v['vz'][i]))

            # applying reset gate value
            v['v_h'].append(
                np.dot(self.w['u_h'], inputs[i]) + np.dot(self.w['w_h'], np.multiply(v['h'][i - 1], v['r'][i])) + +
                self.b['_h'])
            v['_h'].append(tanh(v['v_h'][i]))

            # applying update gate value
            v['h'][i] = np.multiply(v['z'][i], v['h'][i - 1]) + np.multiply(1 - v['z'][i], v['_h'][i])

            # calculating output
            v['vo'].append(np.dot(self.w['wo'], v['h'][i]))
            o.append(self.activation(v['vo'][i]))
        return o[-1], v

    def backward_pass(self, ugrad, v, inputs):

        lgrad = {}

        # hidden to outpur weight's dw
        ugrad *= self.deactivation(v['vo'][-1])
        self.dw['wo'] += np.dot(ugrad, v['h'][-1].T)
        self.db['o'] += ugrad

        # gradient at top hidden layer
        dh = np.dot(self.w['wo'].T, ugrad)

        for i in reversed(range(len(inputs))):

            dz = (v['h'][i - 1] - v['_h'][i]) * dh
            dz__ = v['z'][i] * dh
            dz_ = dsigmoid(v['vz'][i]) * dz

            d_h = np.subtract(1.0, v['z'][i]) * dh
            d_h_ = dtanh(v['v_h'][i]) * d_h

            temp = np.dot(self.w['w_h'].T, d_h_)
            dr = v['h'][i - 1] * temp
            dr_ = dsigmoid(v['vr'][i]) * dr
            dr__ = v['r'][i] * temp

            # calculating reset dw
            self.dw['ur'] += np.dot(dr_, inputs[i].T)
            self.db['r'] += dr_
            self.dw['wr'] += np.dot(dr_, v['h'][i - 1].T)

            # calculating update dw
            self.dw['uz'] += np.dot(dz_, inputs[i].T)
            self.db['z'] += dz_
            self.dw['wz'] += np.dot(dz_, v['h'][i - 1].T)

            # calculating _h dw
            self.dw['u_h'] += np.dot(d_h_, inputs[i].T)
            self.db['_h'] += d_h_
            self.dw['w_h'] += np.dot(d_h_, np.multiply(v['r'][i], v['h'][i - 1]).T)

            dh = np.dot(self.w['wr'].T, dr_) + dr__ + np.dot(self.w['wz'].T, dz_) + dz__
            lgrad[i] = np.dot(self.w['ur'].T, dr_) + np.dot(self.w['uz'].T, dz_) + np.dot(self.w['u_h'].T, d_h_)

        self.count+=1
        return lgrad

    def update_weights(self):
        for en in ['ur', 'wr', 'uz', 'wz', 'u_h', 'w_h', 'wo']:
            self.w[en], _ = self.optimum.update(self.w[en], self.dw[en]/self.count, extra=[en])
        for wpi in self.db:
            self.b[wpi] -= self.optimum.neta * self.db[wpi]/self.count
        return

    def load_values(self, fname):
        self.isize, self.hsize, self.osize, self.w, self.b, self.optimum.opt_var.g = pickle.load(open(fname, 'rb'))

    def get_values(self, wfname):
        return [self.isize, self.hsize, self.osize, self.w, self.b, self.optimum.opt_var.g]

    def save_values(self, wfname):
        pickle.dump([self.isize, self.hsize, self.osize, self.w, self.b, self.optimum.opt_var.g], open(wfname, 'wb'))
