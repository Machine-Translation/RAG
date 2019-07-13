# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:01:58 2019

@author: Cory
"""
from keras.layers import Input, Dense, GRU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.activationxs
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils

class VRNN():

    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        #feature-extracting transformations
        self.phi_x = Sequential([
            Dense(x_dim, h_dim, activation='relu'),
            Dense(h_dim, h_dim, activation='relu')])

        self.phi_z = Sequential([Dense(z_dim, h_dim, activation='relu')])

        #encoder
        self.enc = Sequential([
            Dense(h_dim + h_dim, h_dim, activation='relu'),
            Dense(h_dim, h_dim, activation='relu')])

        self.enc_mean = Dense(h_dim, z_dim)
        self.enc_std = Sequential([
            Dense(h_dim, z_dim, activation='softplus')])

        #prior
        self.prior = Sequential([Dense(h_dim, h_dim, activation='relu')])

        self.prior_mean = Dense(h_dim, z_dim)
        self.prior_std = Sequential([Dense(h_dim, z_dim, activation='softplus')])

        #decoder
        self.dec = Sequential([
            Dense(h_dim + h_dim, h_dim, activation='relu'),
            Dense(h_dim, h_dim, activation='relu')])

        self.dec_std = Sequential([Dense(h_dim, x_dim, activation='softplus')])

        #self.dec_mean = Dense(h_dim, x_dim)
        self.dec_mean = Sequential([Dense(h_dim, x_dim, activation='sigmoid')])

        #recurrence
        #self.rnn = GRU(h_dim + h_dim, h_dim, n_layers, bias)
        self.rnn = GRU(h_dim + h_dim, use_bias = bias)


    def forward(self, x):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)