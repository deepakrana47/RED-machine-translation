import pickle
from util.Utility import init_weight
from util.Activations import *
from Autoencoder import autoencoder
from util.Utility import mini_batch
from util.Logger import logger

class stack_RAE:
    def __init__(self, input_size, hidden_size, optimization=None, wpresent=[], hidden_activation=tanh, hidden_deactivation=dtanh, output_activation=tanh, output_deactivation=dtanh, log_file='log.txt', log=1, vector=None, debug=1):
        self.model_type = 'RAE'
        self.i_size = input_size
        self.h_size = hidden_size
        self.optimum = optimization
        self.hidactivation = hidden_activation
        self.hiddeactivation = hidden_deactivation
        self.outactivation = output_activation
        self.outdeactivation = output_deactivation
        self.en_de_coder = autoencoder(hidden_activation, hidden_deactivation, output_activation, output_deactivation)
        self.w , self.b, self.dw, self.db, self.count = self.init_weights(input_size, hidden_size, wpresent)
        self.count2 = 0.0
        self.wpresent = wpresent
        self.log = log
        self.logg = logger('text', log_file)
        self.vector = vector
        self.debug= debug

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
        h={}
        for j in range(len(data['h_vect'])):
            tx = np.concatenate([vect[k] for k in data['h_vect'][j]], axis=0)
            twe = np.concatenate([self.w['e'][i] for i in data['wp'][j]], axis=1)
            t, _ = self.en_de_coder.encoder(tx, twe, self.b['e'][0])
            vect[wsize + j] = t
            h[j]=t
        return h

    def encoder(self, data):
        vect = data['vects']
        wsize = data['w_size']
        h_vect = data['h_vect']
        wp = data['wp']
        v = {}
        for j in range(len(h_vect)):
            to=0.0
            for i in range(len(h_vect[j])):
                to += np.dot(self.w['e'][wp[j][i]], vect[h_vect[j][i]])
            v[wsize + j] = to
            vect[wsize + j] = self.hidactivation(to + self.b['e'][0])
        data['vects'] = vect
        return vect, v

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
            cost = 0.0; vcount = 0
            for batch in range(len(batches)):
                for data in batches[batch]:
                    vect = data['vects']
                    wsize = data['w_size']
                    for j in range(len(data['h_vect'])):
                        wp = data['wp'][j]
                        hvect = data['h_vect'][j]
                        x = [vect[k] for k in hvect]
                        xsplit = np.cumsum([i.shape[0] for i in x])[:-1]
                        tx = np.concatenate([i for i in x], axis=0)
                        twe = np.concatenate([self.w['e'][i] for i in wp], axis=1)
                        twd = np.concatenate([self.w['d'][i] for i in wp], axis=0)
                        tbd = np.concatenate([self.b['d'][i] for i in wp], axis=0)

                        # encodeing
                        h, to = self.en_de_coder.encoder(tx, twe, self.b['e'][0])
                        # for i in np.argsort(np.abs(h), axis=0)[:int(h.shape[0]*2/5)]:
                        #     h[i] = 0.0
                        np.where(np.abs(h) < h[np.argsort(np.abs(h), axis=0)[int(h.shape[0]*2/5)]], 0.0, h)
                        vect[wsize+j] = h

                        # decoder
                        y, to_ = self.en_de_coder.decoder(h, twd, tbd)

                        # cost
                        cost += np.linalg.norm(y-tx)
                        # cost += np.sum(np.square(y-tx))
                        vcount += 1

                        # backpropogation
                        hgrad, tograd, tlgrad = self.en_de_coder.backward_pass(y-tx, twe, twd, to, to_)
                        ograd = np.split(tograd, xsplit, axis=0)
                        lgrad = np.split(tlgrad, xsplit, axis=0)

                        # modifing input vector by vary small
                        # self.update_vector(data['words'], hvect, lgrad, wsize)

                        # calculating weight update
                        for i in range(len(wp)):
                            self.dw['d'][wp[i]] += np.dot(ograd[i], h.T)
                            self.db['d'][wp[i]] += ograd[i]
                            self.dw['e'][wp[i]] += np.dot(hgrad, x[i].T)
                            self.count[wp[i]] += 1
                        self.db['e'][0] += hgrad
                        self.count[0] += 1

                # updating weight
                self.update_weights()

                if (batch+1)%1 == 0:
                    print "%d/%d batch error is : %f"%(batch+1, len(batches), cost/vcount)
                    if self.log == 1:
                        self.logg.log_text("%d/%d batch error is : %f\n"%(batch+1, len(batches), cost/vcount))
                    cost = 0.0
                    vcount = 0
            print "%d/%d epoch completed ...." % (ep + 1, epoch)

            if self.log == 1:
                self.logg.log_text("%d/%d epoch completed ....\n" % (ep + 1, epoch))

    def update_vector(self, words, hvect, grad, wsize, neta=0.0001):
        for hv in range(len(hvect)):
            if hvect[hv] < wsize:
                self.vector.set_word_vect(words[hvect[hv]], self.vector.get_word_vect(words[hvect[hv]]) + np.multiply(neta, grad[hv]))
        return


    def update_weights(self):
        for en in ['e', 'd']:
            self.w[en], self.b[en] = self.update_weight(self.w[en], self.dw[en], self.b[en], self.db[en], en=en)
            # if self.debug:
            #     print "Relative","encoding" if en == 'e' else "decoding","dw:"
            #     for wpi in sorted(self.dw[en]):
            #         print wpi,":",np.sum(np.divide(self.dw[en][wpi]/self.count[wpi] if self.count[wpi] else 1.0, self.w[en][wpi])),
            #     print
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

    def gradient_check(self, dataset, wpresent, eps=1e-6, tolerance=1e-7):
        # add eps
        count = 0
        data2check = []
        wp2check = []
        wpsize = len(wpresent)
        for data in dataset:
            check = 0
            for j in range(len(data['wp'])):
                wp = data['wp'][j]
                for i in wp:
                    if i in wpresent:
                        wp2check.append(wpresent.pop(wpresent.index(i)))
                        check = 1
            if check:
                data2check.append(data)

        wbelow=0
        for data in data2check:
            vect = data['vects']
            wsize = data['w_size']
            for j in range(len(data['h_vect'])):
                wp = data['wp'][j]
                hvect = data['h_vect'][j]
                for wp_index in range(len(wp)):
                    if wp[wp_index] in wp2check:
                        wbelow += 1
                        print 'weights', wp[wp_index],
                        for en in 'ed':
                            # print 'encoder' if en is 'e' else 'decoding', 'weights', wp[wp_index]
                            for r in range(self.w[en][wp[wp_index]].shape[0]):
                                cost_plus=cost_minus=0.0;pr=0
                                for c in range(self.w[en][wp[wp_index]].shape[1]):
                                    # eps plus calculation
                                    self.w[en][wp[wp_index]][r, c] += eps
                                    # encodeing
                                    to=0.0
                                    # try:
                                    for i in range(len(hvect)):
                                        to += np.dot(self.w['e'][wp[i]], vect[hvect[i]])
                                    vect[wsize+j] =  self.hidactivation(to + self.b['e'][0])
                                    # except KeyError:
                                    #     print
                                    # decoder
                                    vect_={}
                                    for i in range(len(hvect)):
                                        to_ = np.dot(self.w['d'][wp[i]], vect[wsize+j])
                                        vect_[hvect[i]] = self.outactivation(to_ + self.b['d'][wp[i]])
                                    # cost
                                    for i in range(len(hvect)):
                                        cost_plus += np.linalg.norm(vect_[hvect[i]] - vect[hvect[i]])

                                    # eps mines calculation
                                    self.w[en][wp[wp_index]][r, c] -= 2*eps
                                    # encodeing
                                    to = 0.0
                                    for i in range(len(hvect)):
                                        to += np.dot(self.w['e'][wp[i]], vect[hvect[i]])
                                    vo = self.hidactivation(to + self.b['e'][0])
                                    # decoder
                                    vect_ = {}
                                    for i in range(len(hvect)):
                                        to_ = np.dot(self.w['d'][wp[i]], vo)
                                        vect_[hvect[i]] = self.outactivation(to_ + self.b['d'][wp[i]])
                                    # cost
                                    for i in range(len(hvect)):
                                        cost_minus += np.linalg.norm(vect_[hvect[i]] - vect[hvect[i]])

                                    cost = abs(cost_plus-cost_minus)
                                    if cost > tolerance:
                                        # print [r,c],"=",cost,
                                        # pr = 1
                                        count += 1
                                    self.w[en][wp[wp_index]][r, c] += eps
                                # if pr:
                                #     print
                        wp2check.pop(wp2check.index(wp[wp_index]))
                    else:
                        # encodeing
                        to = 0.0
                        for i in range(len(hvect)):
                            to += np.dot(self.w['e'][wp[i]], vect[hvect[i]])
                        vect[wsize + j] = self.hidactivation(to + self.b['e'][0])
                        # decoder
                        vect_ = {}
                        for i in range(len(hvect)):
                            to_ = np.dot(self.w['d'][wp[i]], vect[wsize + j])
                            vect_[hvect[i]] = self.outactivation(to_ + self.b['d'][wp[i]])
                    if not wp2check:
                        print "\nNumber of weights not give good tolerance : %d/%d" % (wbelow,wpsize)
                        print "Number of neurons not give good tolerance : %d/%d" % (count,wpsize*50*50)
                        return
        print "\nNumber of weights not give good tolerance : %d/%d"%(wbelow,wpsize)
        print "Number of neurons not give good tolerance : %d/%d"%(count,wpsize*50*50)
        return