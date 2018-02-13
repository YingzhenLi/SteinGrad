import numpy as np
import tensorflow as tf

"""
Compute the KSD divergence using samples, adapted from the theano code
"""

def KSD(z, Sqx):

    # compute the rbf kernel
    K, dimZ = z.get_shape().as_list()
    z_ = tf.stop_gradient(tf.expand_dims(z, 1))	# shape (K, 1, dimZ)
    pdist_square = tf.reduce_sum((z - z_)**2, -1)	# shape (K, K)
    # use median
    pdist_vec = tf.reshape(pdist_square, (-1,))
    top_k, _ = tf.nn.top_k(pdist_vec, k = max(int(K**2/2), 1))
    median = tf.gather(top_k, max(int(K**2/2)-1, 0)) + 0.1
    h_square = tf.stop_gradient(0.5 * median / np.log(K+1.0))
    Kxy = tf.exp(- pdist_square / h_square / 2.0)

    # now compute KSD
    Sqxdy = tf.matmul(Sqx, tf.transpose(z)) \
        - tf.tile(tf.reduce_sum(Sqx * z, 1, keep_dims=True), (1, K))
    Sqxdy = -Sqxdy / h_square

    dxSqy = tf.transpose(Sqxdy)
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square
    # M is a (K, K) tensor
    M = (tf.matmul(Sqx, tf.transpose(Sqx)) + Sqxdy + dxSqy + dxdy) * Kxy
    #return tf.reduce_mean(M)	# V-statistic

    # the following for U-statistic
    M2 = M - M*tf.diag(tf.ones(shape=(K,)))#tf.diag(tf.diag(M))
    return tf.reduce_sum(M2) / (K * (K - 1))
