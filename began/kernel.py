import tensorflow as tf

def Epanechnikov_kernel(z, K):
    z_ = tf.expand_dims(z, 1)
    pdist_square = (z - tf.stop_gradient(z_))**2
    kzz = tf.reduce_mean(1 - pdist_square, -1)

    return kzz, tf.constant(1.0)
   
