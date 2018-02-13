import numpy as np
import tensorflow as tf
from kernel import *

def entropy_gradient(z, method):
    K, dimZ = z.get_shape().as_list()
    # Epanechnikov kernel
    Kxy, h_square = Epanechnikov_kernel(z, K)
    dxkxy = -2 * K * (tf.reduce_mean(z, 0) - z) / dimZ
    lbd = 0.01
 
    # stein's method
    if method == 'stein':
        K_ = Kxy+lbd*tf.diag(tf.ones(shape=(K,)))
        # test U-statistic (didn't work well?)
        #K_ = K_ - Kxy * tf.diag(tf.ones(shape=(K,)))
        entropy_grad = tf.matrix_solve(K_, dxkxy)
        entropy_loss = tf.reduce_mean(tf.stop_gradient(entropy_grad) * z)	# an estimation for the entropy gradient
    
    # plug-in estimates
    if method == 'kde':
        entropy_grad = dxkxy / (tf.reduce_sum(Kxy, 1, keep_dims=True)+lbd)
        entropy_loss = tf.reduce_mean(tf.stop_gradient(entropy_grad) * z)
    
    # Score matching
    if method == 'score':
        z_ = tf.expand_dims(z, 1)
        z_mean = tf.reduce_mean(z, 0)
        T = tf.reduce_mean(z * z_, 2) + tf.reduce_mean(z**2) - tf.reduce_mean((z + z_)*z_mean, 2)
        a = tf.matrix_solve(T+lbd*tf.diag(tf.ones(shape=(K,))), dimZ*tf.ones(shape=(K,1))) * 0.5
        entropy_grad = -2.0 /dimZ * (tf.reduce_sum(a*z, 0) - tf.reduce_sum(a) * z)
        entropy_loss = tf.reduce_mean(tf.stop_gradient(entropy_grad) * z)
   
    return entropy_loss, Kxy, h_square

    
def image_diversity(sess, image, X_train, y_train, power = 1):
    # then compute nearest neighbour in X_train
    print "compute pairwise distance..."
    batch_size = image.shape[0] * 2
    image_ph = tf.placeholder(tf.float32, shape=image.shape)
    X_ph = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
    X_ph_ = tf.expand_dims(X_ph, 1)
    dist_tf = tf.reduce_sum(tf.abs(X_ph_ - image_ph) ** power, [2, 3, 4]) # (K, K) matrix
    for k in xrange(X_train.shape[0] / batch_size):
        X_batch = X_train[k*batch_size:(k+1)*batch_size]
        dist_now = sess.run(dist_tf, feed_dict={X_ph:X_batch, image_ph: image})
        if k == 0:
            dist = dist_now
        else:
            dist = np.concatenate((dist, dist_now), 0)
    index = np.argmin(dist, axis=0)
    dist = np.min(dist, axis=0)
    X_neighbour = X_train[index]
    y_neighbour = y_train[index] # (K, 10), K is number of images K = x.shape[0]
    # compute entropy
    prob = np.mean(y_neighbour, axis=0)	# a (10) vector
    print prob
    eps = 10e-8
    entropy = -np.sum(prob * np.log(prob+eps)) # a scalar
    return prob, entropy, np.mean(dist), X_neighbour
    
