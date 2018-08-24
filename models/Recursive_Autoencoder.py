import pickle
from util.Utility import init_weight
from util.Activations import *
from Autoencoder import autoencoder
from util.Utility import mini_batch
from util.Logger import logger

class Full_RAE:
    def __init__(self, input_size, hidden_size, optimization=None, wpresent=[], hidden_activation=elu, hidden_deactivation=delu, output_activation=tanh, output_deactivation=dtanh, log_file='log.txt', log=1, vector=None):
        self.model_type = 'RAE'
        self.i_size = input_size
        self.h_size = hidden_size
        self.optimum = optimization
        self.hidactivation = hidden_activation
        self.hiddeactivation = hidden_deactivation
        self.outactivation = output_activation
        self.outdeactivation = output_deactivation
        self.ae = autoencoder(hidden_activation, hidden_deactivation, output_activation, output_deactivation)
        self.w , self.b, self.dw, self.db, self.count = self.init_weights(input_size, hidden_size, wpresent)
        self.vcount = 0.0
        self.count2 = 0.0
        self.wpresent = wpresent
        self.log = log
        self.logg = logger('text', log_file)
        self.vector = vector

    def init_weights(self, i_size, h_size, wpresent):
        w = {'e': {}, 'd': {}}
        b = {'e': {}, 'd': {}}
        db = {'e': {}, 'd': {}}
        dw = {'e': {}, 'd': {}}; count = {}
        for i in wpresent:
            w['e'][i] = init_weight(h_size, h_size)
            b['e'][i] = 0.0
            w['d'][i] = init_weight(h_size, h_size)
            b['d'][i] = np.zeros((h_size,1))
            dw['e'][i] = 0.0;dw['d'][i] = 0.0
            db['e'][i] = 0.0;db['d'][i] = 0.0;count[i] = 0.0
        b['e'][0] = np.zeros((h_size, 1))
        # b['e'][0.1] = np.zeros((h_size, 1))

        for i in [0, 0.1]:
            w['e'][i] = init_weight(h_size, i_size)
            w['d'][i] = init_weight(i_size, h_size)
            b['d'][i] = np.zeros((i_size,1))
        return w, b, dw, db, count

    def init_dw(self, w, b):
        for en in w:
            self.dw[en] = {}
            for ind in w[en]:
                self.dw[en][ind] = 0.0
                self.count[ind] = 0
        for en in b:
            self.db[en] = {}
            for ind in b[en]:
                self.db[en][ind] = 0.0


    def predict(self, data):
        vect = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        h = {}
        for j in range(len(h_vect)):
            to = 0.0
            for i in range(len(h_vect[j])):
                to += np.dot(self.w['e'][wp[j][i]], vect[h_vect[j][i]])
            t=self.hidactivation(to + self.b['e'][0])
            vect[wsize+j] = t
            h[j] = t
        return h

    def encoder(self, data):
        vect = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        v = {}
        for j in range(len(h_vect)):
            to = 0.0
            for i in range(len(h_vect[j])):
                to += np.dot(self.w['e'][wp[j][i]], vect[h_vect[j][i]])
            v[wsize + j] = to
            vect[wsize + j] = self.hidactivation(to + self.b['e'][0])
        data['vects'] = vect
        return vect, v

    def decoder(self, data):
        vects = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        v = {}
        vect_ = {}
        for j in reversed(range(len(h_vect))):
            for i in range(len(h_vect[j])):
                v[h_vect[j][i]] = np.dot(self.w['d'][wp[j][i]], vects[wsize + j]) + self.b['d'][wp[j][i]]
                vect_[h_vect[j][i]] = self.outactivation(v[h_vect[j][i]]) if h_vect[j][i] < wsize else self.hidactivation(v[h_vect[j][i]])
        return vect_, v

    def decoder_back(self, grad, v, data):
        vect = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        for j in range(len(h_vect)):
            tgrad = 0
            for i in range(len(h_vect[j])):
                grad[h_vect[j][i]] *= self.outdeactivation(v[h_vect[j][i]])  if h_vect[j][i] < wsize else self.hidactivation(v[h_vect[j][i]])
                self.dw['d'][wp[j][i]] += np.dot(grad[h_vect[j][i]], vect[wsize+j].T)
                self.db['d'][wp[j][i]] += tgrad
                tgrad += np.dot(self.w['d'][wp[j][i]].T ,grad[h_vect[j][i]])
                self.count[wp[j][i]] += 1
            grad[wsize + j] = tgrad
        return {wsize + j:grad[wsize + j]}

    def encoder_back(self, grad, v, data):
        vect = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        for j in reversed(range(len(h_vect))):
            for i in range(len(h_vect[j])):
                grad[wsize + j] = grad[wsize + j] * self.hiddeactivation(v[wsize+j])
                self.dw['e'][wp[j][i]] += np.dot(grad[wsize + j], vect[h_vect[j][i]].T)
                grad[h_vect[j][i]] = np.dot(self.w['e'][wp[j][i]].T, grad[wsize + j])
            self.db['e'][0] += grad[wsize + j]
            self.count2 += 1
        return

    def cost(self, o, t):
        cost = 0.0
        for i in o:
            cost += np.linalg.norm(t[i]-o[i])
            self.vcount += 1
        return cost

    def gradient(self, o, t, wsize):
        grad = {}
        for i in range(wsize):
            grad[i] = o[i] - t[i]
        return grad

    def calc_dw(self, grad, data):
        vect = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        for j in range(len(h_vect)):
            for i in range(len(h_vect[j])):
                self.dw['e'][wp[j][i]] += np.dot(grad[wsize+j], vect[h_vect[j][i]].T)
                self.count[wp[j][i]] += 1
            self.db['e'][0] += grad[wsize+j]
        return



    def train(self, xs, epoch, batch_size):
        for ep in range(epoch):
            batches = mini_batch(xs, len(xs), batch_size)
            cost = 0.0
            for batch in range(len(batches)):
                for data in batches[batch]:
                    _, v = self.encoder(data)
                    vect_, v_ = self.decoder(data)

                    cost += self.cost(vect_, data['vects'])
                    grad = self.gradient(vect_, data['vects'], wsize=data['w_size'])

                    dgrad = self.decoder_back(grad, v_, data)
                    self.encoder_back(dgrad, v, data)

                # updating weight
                self.update_weights()

                if (batch+1)%1 == 0:
                    print "%d/%d batch error is : %f"%(batch+1, len(batches), cost/self.vcount)
                    if self.log == 1:
                        self.logg.log_text("%d/%d batch error is : %f\n"%(batch+1, len(batches), cost/self.vcount))
                    cost = 0.0
                    self.vcount = 0
            print "%d/%d epoch completed ...." % (ep + 1, epoch)

            if self.log == 1:
                self.logg.log_text("%d/%d epoch completed ....\n" % (ep + 1, epoch))

    # def update_vector(self, words, hvect, grad, wsize, neta=0.0001):
    #     for hv in range(len(hvect)):
    #         if hvect[hv] < wsize:
    #             self.vector.set_word_vect(words[hvect[hv]], self.vector.get_word_vect(words[hvect[hv]]) + np.multiply(neta, grad[hv]))
    #     return


    def update_weights(self):
        for en in ['e', 'd']:
            self.w[en], self.b[en] = self.update_weight(self.w[en], self.dw[en], self.b[en], self.db[en], en=en)
        for i in self.wpresent:
            self.dw['e'][i] = 0.0; self.dw['d'][i] = 0.0
            self.db['e'][i] = 0.0; self.db['d'][i] = 0.0;            self.count[i] = 0.0
        self.count2 = 0.0
        return

    def update_weight(self, w, dw, b, db, en):
        for wpi, g in dw.items():
            dw[wpi] /= self.count[wpi] if self.count[wpi] else 1.0
            db[wpi] /= self.count[wpi] if self.count[wpi] else 1.0
            w[wpi], b[wpi] = self.optimum.update(w[wpi], dw[wpi], b[wpi], db[wpi], extra=[en, wpi])
        return w, b

    def save_variables(self, fname):
        pickle.dump([self.i_size, self.h_size, self.hidactivation,
                     self.hiddeactivation, self.outactivation,
                     self.outdeactivation, self.w, self.b,
                     self.optimum.opt_var.g], open(fname, 'wb'))

    def load_variables(self, fname):
        self.i_size, self.h_size, self.hidactivation,\
        self.hiddeactivation, self.outactivation,\
        self.outdeactivation, self.w, self.b, self.optimum.opt_var.g = pickle.load(open(fname, 'rb'))