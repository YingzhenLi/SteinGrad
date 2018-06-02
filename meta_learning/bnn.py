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
    print "calling predict, shape(Theta) =", theta.get_shape(),
    print "activation =", activation, func

    K = theta.get_shape().as_list()[0]	# shape (K, dimTheta)
    X_3d = tf.tile(tf.expand_dims(X, 0), [K, 1, 1])	# shape (K, N, dimX)

    N_layers = len(shapes)
    dim = 0
    h = X_3d
    for i in xrange(N_layers):
        (d_in, d_out) = shapes[i]
        W = tf.reshape(theta[:, dim:dim+d_in*d_out], [K, d_in, d_out])	# (K, d_in, d_out)
        b = tf.expand_dims(theta[:, dim+d_in*d_out:(dim+d_in*d_out+d_out)], 1)	# (K, 1, d_out)
        h = tf.matmul(h, W) + b
        if i + 1 < N_layers:	# not last layer
            h = func(h)	# shape (K, N, d_out)
        dim = d_in * d_out + d_out    
    return tf.transpose(tf.squeeze(h))	# shape (N, K)

def evaluate(X, y, theta, shapes = None, activation = 'relu'):  
    # we assume y in {0, 1} 
    print "calling evaluate, shape(Theta) =", theta.get_shape(),
    print "activation =", activation

    N, dimY = y.get_shape().as_list()
    prob = tf.nn.sigmoid(predict(X, theta, shapes, activation))
    if len(prob.get_shape().as_list()) > 1:	# K > 1, need to do averaging
        prob = tf.reduce_mean(prob, 1, keep_dims=True)	# shape (batch_size)
    else:
        prob = tf.expand_dims(prob, 1)
    prob = tf.clip_by_value(prob, 1e-8, 1.0 - 1e-8)
    label = tf.where(prob > 0.5, tf.ones(shape=(N,dimY)), tf.zeros(shape=(N,dimY)))
    acc = tf.reduce_mean(tf.cast(tf.equal(label, y), tf.float32))
    llh = tf.reduce_mean(y*tf.log(prob) + (1-y)*tf.log(1 - prob))
    
    return acc, llh, prob 

def logp_bnn(X, y, theta, data_N, shapes = None, add_prior = True, activation = 'relu'):
    # use tensorflow automatic differentiation to compute the gradients
    print "calling logp_bnn, shape(Theta) =", theta.get_shape(),
    print "activation =", activation

    logit = predict(X, theta, shapes, activation)
    if len(logit.get_shape().as_list()) == 1:
        logit = tf.expand_dims(logit, 1)
    N, K = logit.get_shape().as_list()
    y_ = tf.tile(y, [1, K])
    # for tensorflow r0.11
    logll = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logit)
    logll = -tf.reduce_mean(logll, 0)	# shape (K,)
    if add_prior:
        # assume N(0, 1) prior
        log_prior = -0.5 * tf.reduce_sum(theta**2, 1)	# up to a constant
        logprob = logll * data_N + log_prior
    else:
        logprob = logll	# no scale up with data_N

    return logprob

def grad_bnn(X, y, theta, data_N, shapes = None, activation = 'relu'):
    # use tensorflow automatic differentiation to compute the gradients
    #prob = tf.clip_by_value(predict(X, theta, shapes), 1e-8, 1.0 - 1e-8)
    #logit = predict(X, theta, shapes)
    #N, K = logit.get_shape().as_list()
    #y_ = tf.tile(y, [1, K])
    # for tensorflow r0.11
    #logprob = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logit)
    #logprob = -tf.reduce_mean(logprob, 0)	# shape (K,)
    #dtheta_data = tf.gradients(logprob, theta)[0]
    # assume prior is N(0, 1)
    #dtheta_prior = - theta

    #dtheta = dtheta_data * data_N + dtheta_prior
    print "calling grad_bnn, shape(Theta) =", theta.get_shape(),
    print "activation =", activation
    logp = logp_bnn(X, y, theta, data_N, shapes, activation = activation)
    dtheta = tf.gradients(logp, theta)[0]

    return dtheta

def grad_bnn_separate(X, y, theta, data_N, shapes = None, activation = 'relu'):
    print "calling grad_bnn_separate, shape(Theta) =", theta.get_shape(),
    print "activation =", activation
    logll = logp_bnn(X, y, theta, data_N, shapes, add_prior=False,
                    activation = activation)
    dtheta_ll = tf.gradients(logll, theta)[0]
    # assume prior is N(0, 1)
    dtheta_prior = -theta

    return dtheta_ll, dtheta_prior

def grad_bnn_per_sample(X, y, theta, data_N, shapes = None, activation = 'relu'):
    print "calling grad_bnn_per_sample, shape(Theta) =", theta.get_shape(),
    print "activation =", activation

    X_ = tf.split(X, X.get_shape().as_list()[0], axis=0)
    y_ = tf.split(y, y.get_shape().as_list()[0], axis=0)
    theta_ = [tf.identity(theta) for x in X_]	# list of theta copies of shape (K, dimTheta)
    N = len(theta_)
    logp = []
    for n in xrange(N):
        logp.append(logp_bnn(X_[n], y_[n], theta_[n], data_N, shapes, 
                    add_prior=False, activation = activation))
    dtheta_ll = tf.gradients(logp, theta_)
    dtheta_ll = tf.stack(dtheta_ll) # shape (N, K, dimTheta)
    # assume prior is N(0, 1)
    dtheta_prior = -theta
    
    return dtheta_ll, dtheta_prior 


    
