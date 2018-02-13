from keras.models import Sequential
from keras.layers import Dense, Reshape, Lambda
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.layers.core import Flatten
import tensorflow as tf
import numpy as np
from keras import backend as K

alpha = 0.2

def Dropout(p):
    layer = Lambda(lambda x: K.dropout(x, p), output_shape=lambda shape: shape)
    return layer

def encoder_model(layer_sizes, name = 'encoder', batch_norm = False, \
                 reshape_conv = False, dropout = False):
    model = Sequential()
    bias = (not batch_norm)
    if reshape_conv:
        model.add(Reshape((784,), input_shape = (28, 28, 1)))
    for l in xrange(len(layer_sizes)-1):
        model.add(Dense(input_dim=layer_sizes[l], \
                        output_dim=layer_sizes[l+1], \
                        name = name+'_l%d' % l,
                        bias = bias))
        if batch_norm:
            model.add(BatchNormalization(name = name + '_bn%d' % l, mode = 2))
        if l < len(layer_sizes) - 2:
            #model.add(LeakyReLU(alpha = alpha))
            model.add(Activation('relu'))
            #model.add(Activation('tanh'))
        if dropout:
            print "add in dropout"
            model.add(Dropout(0.2))
    return model

def decoder_model(layer_sizes, name = 'decoder', reshape_conv = False):
    model = Sequential()
    for l in xrange(len(layer_sizes)-1):
        model.add(Dense(input_dim=layer_sizes[l], \
                        output_dim=layer_sizes[l+1], \
                        name = name+'_l%d' % l,
                        bias = True))
        if l < len(layer_sizes) - 2:
            model.add(LeakyReLU(alpha = alpha))
            #model.add(Activation('tanh'))
        else:
            model.add(Activation('sigmoid'))
    if reshape_conv:
        model.add(Reshape((28, 28, 1)))
    return model

def discriminator_model(layer_sizes, name = 'discriminator', batch_norm = False):
    model = Sequential()
    bias = (not batch_norm)
    for l in xrange(len(layer_sizes)-1):
        model.add(Dense(input_dim=layer_sizes[l], \
                        output_dim=layer_sizes[l+1], \
                        name = name+'_l%d' % l,
                        bias = bias))
        if batch_norm:
            model.add(BatchNormalization(name = name + '_bn%d' % l, mode = 2))
        if l < len(layer_sizes) - 2:
            model.add(LeakyReLU(alpha = alpha))
    return model
    
