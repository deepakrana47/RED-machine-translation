from util.Activations import tanh, dtanh
from GRU import gru

class RNN_encoder_decoder:

    def __init__(self, isize, hsize, osize, rnn_unit='gru', outactivation=tanh, outdeactivation=dtanh, optmization=None):

        if not optmization:

        if rnn_unit == 'gru':
            gru = GRU_encoder_decoder(isize, hsize, osize, outactivation, outdeactivation)
            self.encoder = gru.encoder
            self.decoder = gru.decoder
        pass

class GRU_encoder_decoder:
    def __init__(self, isize, hsize, osize, outactivation, outdeactivation):
        self.en_gru = gru(isize, hsize, osize, outactivation, outdeactivation)
        pass
    def encoder(self, inputs):
        self.en_gru.forward_pass(inputs)
        pass
    def decoder(self):
        pass
    def encoder_back(self):
        pass
    def decoder_back(self):
        pass
