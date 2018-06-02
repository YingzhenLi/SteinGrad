import numpy as np
import tensorflow as tf
import pickle, os

def load_params(sess, path, filename, sampler_name = 'nn_sampler'):
    t_vars = tf.trainable_variables()
    params = [v for v in t_vars if sampler_name in v.name]
    assert os.path.isdir(path)
    g = open(path + filename, 'r')
    params_val = pickle.load(g)
    assign_ops = []
    for val, par in zip(params_val, params):
        assign_ops.append(tf.assign(par, val))
    sess.run(assign_ops)
    print 'load '+ path + filename

def save_params(sess, path, filename, sampler_name = 'nn_sampler'):
    t_vars = tf.trainable_variables()
    params = [v for v in t_vars if sampler_name in v.name]
    params_eval = sess.run(params)
    if not os.path.isdir(path):
        os.mkdir(path)
        print 'create path', path
    g = open(path + filename, 'w')
    pickle.dump(params_eval, g)
    print "model params saved in " + path + filename

def init_theta(a0=1, b0=0.1, n_particle=20, dim = 10, seed = None):
    if seed is not None: np.random.seed(seed)
    theta = np.random.normal(loc=np.zeros((n_particle, 1)), scale=1.0,
                size=(n_particle, dim))
    return theta

def init_theta_glorot_uniform(shapes, n_particle=20, seed = None):
    if seed is not None: np.random.seed(seed)
    theta = []
    for shape in shapes:
        dim = np.prod(shape)
        tmp = np.random.uniform(size=[n_particle, dim]) 
        tmp = (2 * tmp - 1) * np.sqrt(6.0 / np.sum(shape))
        theta.append(tmp)
        tmp = (np.random.uniform(size=[n_particle, shape[1]]) - 0.5) * 0.05	# for bias
        theta.append(tmp)
    theta = np.concatenate(theta, axis=1)
    return theta

def init_theta_glorot_normal(shapes, n_particle=20, seed = None):
    if seed is not None: np.random.seed(seed)
    theta = []
    for shape in shapes:
        dim = np.prod(shape)
        tmp = np.random.normal(size=[n_particle, dim]) * np.sqrt(2.0 / np.sum(shape))
        theta.append(tmp)
        tmp = (np.random.uniform(size=[n_particle, shape[1]]) - 0.5) * 0.05	# for bias
        theta.append(tmp)
    theta = np.concatenate(theta, axis=1)
    return theta

def init_theta_he_normal(shapes, n_particle=20, seed = None):
    if seed is not None: np.random.seed(seed)
    theta = []
    for shape in shapes:
        dim = np.prod(shape)
        tmp = np.random.normal(size=[n_particle, dim]) * np.sqrt(2.0 / shape[0])
        theta.append(tmp)
        tmp = (np.random.uniform(size=[n_particle, shape[1]]) - 0.5) * 0.05	# for bias
        theta.append(tmp)
    theta = np.concatenate(theta, axis=1)
    return theta

def make_shapes(dimX, dimH, dimY, n_hidden_layers=1):
    shapes = [(dimX, dimH)]
    if n_hidden_layers > 1:
        shapes = shapes + [(dimH, dimH) for i in xrange(n_hidden_layers-1)]
    shapes = shapes + [(dimH, dimY)]
    dimTheta = 0
    for shape in shapes:
        dimTheta += np.prod(shape) + shape[1] 
    return shapes, dimTheta
