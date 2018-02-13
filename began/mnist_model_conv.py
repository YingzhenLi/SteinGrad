import numpy as np
import tensorflow as tf
from convnet import ConvNet, construct_filter_shapes

def init_weights(input_size, output_size, constant=1.0, seed=123): 
    """ Glorot and Bengio, 2010's initialization of network weights"""
    scale = constant*np.sqrt(6.0/(input_size + output_size))
    if output_size > 0:
        return tf.random_uniform((input_size, output_size), 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)
    else:
        return tf.random_uniform([input_size], 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed)

def mlp_layer(d_in, d_out, activation, name):
    W = tf.Variable(init_weights(d_in, d_out), name = name+'_W')
    b = tf.Variable(tf.zeros([d_out]), name = name+'_b')
    
    def apply_layer(x):
        a = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a  
            
    return apply_layer

def deconv_layer(output_shape, filter_shape, activation, strides, name):
    scale = 1.0 / np.prod(filter_shape[:3])
    seed = 123
    W = tf.Variable(tf.random_uniform(filter_shape, 
                             minval=-scale, maxval=scale, 
                             dtype=tf.float32, seed=seed), name = name+'_W')
    
    def apply(x):
        output_shape_x = (x.get_shape().as_list()[0],)+output_shape
        a = tf.nn.conv2d_transpose(x, W, output_shape_x, strides, 'SAME')
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a
        if activation == 'split':
            x1, x2 = split(a, axis=3, num_split=2)	# a is a 4-D tensor
            return tf.nn.sigmoid(x1), x2
            
    return apply

def DeConvNet(input_shape, dimH, dimZ, last_activation, name):
    # now construct a decoder
    filter_width = 3
    decoder_input_shape = [(4, 4, 32), (7, 7, 64), (14, 14, 64)]
    decoder_input_shape.append(input_shape)
    fc_layers = [dimZ, dimH, int(np.prod(decoder_input_shape[0]))]
    l = 0
    # first include the MLP
    mlp_layers = []
    N_layers = len(fc_layers) - 1
    for i in xrange(N_layers):
        name_layer = name + '_l%d' % l
        mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i+1], 'relu', name_layer))
        l += 1
    
    conv_layers = []
    N_layers = len(decoder_input_shape) - 1
    for i in xrange(N_layers):
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = last_activation
        name_layer = name + '_l%d' % l
        output_shape = decoder_input_shape[i+1]
        input_shape = decoder_input_shape[i]
        up_height = int(np.ceil(output_shape[0]/float(input_shape[0])))
        up_width = int(np.ceil(output_shape[1]/float(input_shape[1])))
        strides = (1, up_height, up_width, 1)       
        if activation == 'logistic_cdf' and i == N_layers - 1:	# ugly wrapping for logistic cdf likelihoods
            activation = 'split'
            output_shape = (output_shape[0], output_shape[1], output_shape[2]*2)
        filter_shape = (filter_width, filter_width, output_shape[-1], input_shape[-1])
        
        conv_layers.append(deconv_layer(output_shape, filter_shape, activation, \
                                            strides, name_layer))
        l += 1
    
    print name + ' Conv Net of size', decoder_input_shape
    
    def apply(z):
        x = z
        for layer in mlp_layers:
            x = layer(x)
        x = tf.reshape(x, (x.get_shape().as_list()[0],)+decoder_input_shape[0])
        for layer in conv_layers:
            x = layer(x)
        return x
        
    return apply

def construct_model(input_shape, dimZ, dimF, dimH, batch_norm = False):
    # test dropout model to start with
    # construct encoder
    weight_init = 'glorot_normal'
    layer_channels = [32, 64, 64]
    filter_width = 3
    filter_shapes = construct_filter_shapes(layer_channels, filter_width)
    fc_layer_sizes = [dimH, dimF]
    enc, conv_output_shape = ConvNet('encoder_conv', input_shape, filter_shapes, fc_layer_sizes, \
                                     'relu', batch_norm, last_activation = 'relu', \
                                     dropout = False) 
    print 'encoder' + ' network architecture:', \
            conv_output_shape, fc_layer_sizes

    # construct the decoder
    batch_size_ph = tf.placeholder(tf.int32, shape=(), name='batch_size_ph')
    dec = DeConvNet(input_shape, dimH, dimF, last_activation = 'sigmoid', name = 'decoder')
    
    # construct the generator
    generator = DeConvNet(input_shape, dimH, dimF, last_activation = 'sigmoid', name = 'generator')
    
    def gen(z):
        N = z.get_shape().as_list()[0]
        x = generator(z)
        return tf.reshape(x, (N,)+input_shape)
                           
    return enc, dec, gen, batch_size_ph                       
                           
