import numpy as np
import tensorflow as tf

def f_net(dimH, name):
    print 'add in 1 hidden layer mlp...'
    W1 = tf.Variable(tf.random_normal(shape=(dimH, ))*0.01, name = name + '_W1')
    b1 = tf.Variable(tf.random_normal(shape=(dimH, ))*0.01, name = name + '_b1')
    #W2 = tf.Variable(tf.random_normal(shape=(dimH, dimH))*0.01, name = name + '_W2')
    #b2 = tf.Variable(tf.random_normal(shape=(dimH, ))*0.01, name = name + '_b2')
    W3 = tf.Variable(tf.random_normal(shape=(dimH,))*0.01, name = name + '_W3')
    b3 = tf.Variable(tf.random_normal(shape=())*0.01, name = name + '_b3')

    def apply(z):
        x = tf.expand_dims(z, 2)
        x = tf.nn.relu(x * W1 + b1)	# (K, dimX, dimH)
        #x = tf.expand_dims(x, 3)
        #x = tf.nn.relu(tf.reduce_sum(x * W2, 2) + b2)
        x = tf.reduce_sum(x * W3, 2) + b3 	# (K, dimX)
        return x

    return apply

def g_net(dimH, name):
    print 'add in 1 hidden layer mlp...'
    W1 = tf.Variable(tf.random_normal(shape=(dimH, ))*0.01, name = name + '_W1')
    b1 = tf.Variable(tf.random_normal(shape=(dimH, ))*0.01, name = name + '_b1')
    #W2 = tf.Variable(tf.random_normal(shape=(dimH, dimH))*0.01, name = name + '_W2')
    #b2 = tf.Variable(tf.random_normal(shape=(dimH, ))*0.01, name = name + '_b2')
    W3 = tf.Variable(tf.random_normal(shape=(dimH,))*0.01, name = name + '_W3')
    b3 = tf.Variable(tf.random_normal(shape=())*0.01, name = name + '_b3')

    def apply(z):
        x = tf.expand_dims(z, 2)
        x = tf.nn.relu(x * W1 + b1)	# (K, dimX, dimH)
        x = tf.reduce_sum(x, 1)	# (K, dimH)
        #x = tf.expand_dims(tf.reduce_sum(x, 1), 2)	# (K, dimH, 1)
        #x = tf.nn.relu(tf.reduce_sum(x * W2, 1) + b2)	# (K, dimH)
        x = tf.reduce_sum(x * W3, 1) + b3 	# (K, )
        return tf.expand_dims(x, 1)

    return apply

def init_nn_sampler(dimH, name = 'nn_sampler'):
    # parameters
    print 'construct two MLPs with %d units...' % dimH
    f = f_net(dimH, name=name+'_f_net')
    g = g_net(dimH, name=name+'_g_net')        

    def network_transform(z, grad_z, eps, data_N, compute_gamma=True):
        # compute D matrix
        f_out = f(z)	# (K, dimX)       
        g_out = g(z)	# (K, 1)
        mean_g = (grad_z + z) / data_N	# assume gaussian prior
        square_grad = tf.reduce_sum(mean_g ** 2, 1, keep_dims=True)	# (K, 1)
        h_out = 1e-5 + tf.sqrt(square_grad)
        D_matrix = tf.nn.relu(f_out + g_out) / h_out
        direction = D_matrix * grad_z
        if compute_gamma:
            # now compute gamma vector
            print "use gamma vector"
            df = tf.gradients(f_out, z)[0]
            dg = tf.gradients(g_out, z)[0]
            dlogh = tf.gradients(tf.log(h_out), z)[0]
            gamma_vector = (df + dg - dlogh * (f_out + g_out)) / h_out
            gamma_vector = tf.where(D_matrix > 0, gamma_vector, tf.zeros(z.get_shape()))
            direction += gamma_vector
        # compute output 
        noise = tf.random_normal(z.get_shape())
        delta = eps * direction + tf.sqrt(2 * eps * D_matrix) * noise
        
        return delta
    
    def nn_sampler(z, X, y, data_N, grad_logp_func, shapes = None, 
                   eps = 1e-5, compute_gamma = True):
        print "calling the sampler, shape(Theta)=", z.get_shape()
        noise = tf.random_normal(z.get_shape())
        grad_logp = grad_logp_func(X, y, z, data_N, shapes)
        delta = network_transform(z, grad_logp, eps, data_N, compute_gamma)
        z = z + delta
            
        return z
        
    return nn_sampler
    
