import numpy as np
import tensorflow as tf

"""
Bayesian neural nets
"""

def predict(X, theta, shapes = None, activation = 'relu'):
    assert shapes is not None
    if activation == 'relu':
        func = tf.nn.relu
    if activation == 'softplus':
        func = tf.nn.softplus
    if activation == 'sigmoid':
        func = tf.nn.sigmoid

    K = theta.get_shape().as_list()[0]	# shape (K, dimTheta)
    X_3d = tf.tile(tf.expand_dims(X, 0), [K, 1, 1])	# shape (K, N, dimX)	
    # First layer
    (d_in, d_out) = shapes[0]
    W = tf.reshape(theta[:, :d_in*d_out], [K, d_in, d_out])	# (K, d_in, d_out)
    b = tf.expand_dims(theta[:, d_in*d_out:(d_in*d_out+d_out)], 1)	# (K, 1, d_out)
    h = tf.matmul(X_3d, W) + b
    h = func(h)	# shape (K, N, d_out)
    dim = d_in * d_out + d_out    
    # Second layer
    (d_in, d_out) = shapes[1]
    W = tf.reshape(theta[:, dim:dim+d_in*d_out], [K, d_in, d_out])	# (K, d_in, d_out)
    b = tf.expand_dims(theta[:, dim+d_in*d_out:(dim+d_in*d_out+d_out)], 1)	# (K, 1, d_out)
    logit = tf.matmul(h, W) + b     
    # output the logit (prob vector before sigmoid)
    return tf.transpose(tf.squeeze(logit))	# shape (N, K)

def evaluate(X, y, theta, shapes = None, activation = 'relu'):  
    # we assume y in {0, 1} 
    N, dimY = y.get_shape().as_list()
    prob = tf.nn.sigmoid(predict(X, theta, shapes, activation))
    prob = tf.reduce_mean(prob, 1, keep_dims=True)	# shape (batch_size)
    prob = tf.clip_by_value(prob, 1e-8, 1.0 - 1e-8)
    label = tf.where(prob > 0.5, tf.ones(shape=(N,dimY)), tf.zeros(shape=(N,dimY)))
    acc = tf.reduce_mean(tf.cast(tf.equal(label, y), tf.float32))
    llh = tf.reduce_mean(y*tf.log(prob) + (1-y)*tf.log(1 - prob))
    
    return acc, llh   

def grad_bnn(X, y, theta, data_N, shapes = None):
    # use tensorflow automatic differentiation to compute the gradients
    #prob = tf.clip_by_value(predict(X, theta, shapes), 1e-8, 1.0 - 1e-8)
    logit = predict(X, theta, shapes)
    N, K = logit.get_shape().as_list()
    y_ = tf.tile(y, [1, K])
    # for tensorflow r0.11
    logprob = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logit)
    logprob = -tf.reduce_mean(logprob, 0)	# shape (K,)
    #logprob = tf.reduce_mean(tf.log(prob), 0)	# shape (K,)
    dtheta_data = tf.gradients(logprob, theta)[0]
    # assume prior is N(0, 1)
    dtheta_prior = - theta

    return dtheta_data * data_N + dtheta_prior
    
