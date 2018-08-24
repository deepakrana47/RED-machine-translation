import numpy as np

class autoencoder:

    def __init__(self, hidden_activation, hidden_deactivation, output_activation, output_deactivation):
        self.hidactivation = hidden_activation
        self.hiddeactivation = hidden_deactivation
        self.outactivation = output_activation
        self.outdeactivation = output_deactivation
        pass

    def encoder(self, x, w, b):
        ###### Encoding
        v = np.dot(w, x) + b
        h = self.hidactivation(v)
        return h, v

    def decoder(self, h, w, b):
        ###### Decoding
        v = np.dot(w, h) + b
        y = self.outactivation(v)
        return y, v

    def backward_pass(self, grad, we, wd, to, to_):
        ograd = grad * self.outdeactivation(to_)
        hgrad = np.dot(wd.T, ograd) * self.hiddeactivation(to)
        lgrad = np.dot(we.T, hgrad)
        return hgrad, ograd, lgrad

    def decoder_backpass(self, ugrad, w, v):
        grad = ugrad * self.outdeactivation(v)
        lgrad = np.dot(w.T, grad)
        return grad, lgrad

    def encoder_backpass(self, ugrad, w, v):
        grad = ugrad * self.hiddeactivation(v)
        lgrad = np.dot(w.T, grad)
        return grad, lgrad