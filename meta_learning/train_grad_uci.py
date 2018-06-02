import sys, random, os, scipy.io, pickle, argparse, time
sys.path.extend(['sampler/', 'utils/'])
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.utils import shuffle
from bnn import evaluate, grad_bnn, logp_bnn
from nn_sampler import init_nn_sampler
from load_uci import load_uci_data
from ksd import KSD
from utils import *
from gradients import amortised_loss

def main(task='crabs', dimH=20, dimH_nn=20, T_unroll=10, seed=42, hsquare=-1.0, lbd=0.01, 
         batch_size=32, n_particle=50, lr=0.001, n_iter=500, method='map'):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # load data
    datapath = 'data/'
    X_train, y_train = load_uci_data(datapath, task, merge = True)
    N_train, dimX = X_train.shape
    _, dimY = y_train.shape
    print "size of training=%d" % (N_train)

    # now define ops
    activation = 'relu'
    def logp_func(X, y, theta, data_N, shapes):
        return logp_bnn(X, y, theta, data_N, shapes, activation=activation)
    def grad_logp_func(X, y, theta, data_N, shapes):
        return grad_bnn(X, y, theta, data_N, shapes, activation=activation)

    # placeholders for training
    print "settings:"
    print "T_unroll", T_unroll
    print "N_train", N_train
    print "batch_size_train", batch_size

    X_ph = tf.placeholder(tf.float32, shape=(batch_size*T_unroll, dimX), name = 'X_ph')
    y_ph = tf.placeholder(tf.float32, shape=(batch_size*T_unroll, dimY), name = 'y_ph')
    N_ph = tf.placeholder(tf.float32, shape=())
    # train the sampler on small network
    shapes, dimTheta = make_shapes(dimX, dimH, dimY)
    theta_ph = tf.placeholder(tf.float32, shape=(n_particle, dimTheta), name = 'theta_ph')

    # ops for training the langevin sampler
    q_sampler = init_nn_sampler(dimH_nn, name = 'nn_sampler')
    q_stepsize_ph = tf.placeholder(tf.float32, shape=(), name='sampler_stepsize')
    loss, theta_q_train = amortised_loss(theta_ph, X_ph, y_ph, N_ph, q_sampler, 
                                         grad_logp_func, method, shapes, T_unroll, 
                                         hsquare, lbd, q_stepsize_ph)
    t_vars = tf.trainable_variables()
    params = [v for v in t_vars if 'nn_sampler' in v.name]
    grad = tf.gradients(loss, params)
    # clip the norm of the gradient
    grad_norm = 10.0
    #grad = [v / tf.sqrt(tf.reduce_sum(v**2)) for v in grad]
    grad = [tf.clip_by_value(v, -grad_norm, grad_norm) for v in grad]
    grad = zip(grad, params)
    lr_ph = tf.placeholder(tf.float32, shape=(), name = 'lr_ph')
    opt = tf.train.AdamOptimizer(learning_rate = lr_ph).apply_gradients(grad)
    #opt = tf.train.AdamOptimizer(learning_rate = lr_ph).minimize(loss)

    # now check init
    print "initialise tensorflow variables..."
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.set_random_seed(seed)
    initialised_var = set([])
    init_var_list = set(tf.all_variables()) - initialised_var
    if len(init_var_list) > 0:
        init = tf.initialize_variables(var_list = init_var_list)
        sess.run(init)

    filename = "%s_H%d_T%d_N%d_%s_hsquare%.2f_lbd%.2f.pkl" % (task, dimH_nn, T_unroll, n_particle, method, hsquare, lbd)

    # first training the network
    print "Start Training the NN Sampler"
    np.random.seed(seed)
    sigma = 10.0
    theta = init_theta(dim = dimTheta, n_particle=n_particle)

    # memory for replay
    N_samples_memory = 40 * n_particle
    N_sample_update = 5 * n_particle
    prob_replay = 0.0
    replay_theta = init_theta(dim = dimTheta, n_particle=N_samples_memory)
    T_update_replay = 5
    T_restart = 100 / T_unroll
    accum = 0
    stepsize = 1e-5

    for iteration in tqdm(range(1, n_iter+1)):
        ind = np.random.randint(0, N_train, T_unroll*batch_size)
        _, theta, cost = sess.run((opt, theta_q_train, loss), 
                                                  {X_ph: X_train[ind], y_ph: y_train[ind], \
                                                   lr_ph: lr, N_ph: N_train, theta_ph: theta,
                                                   q_stepsize_ph: stepsize})

        if iteration % T_update_replay == 0:
            # update replay memory
            replay_theta[accum:accum+n_particle] = theta
            accum = (accum + n_particle) % N_samples_memory

        if iteration % T_restart == 0:
            # start from new locations from replay or random init
            coin = np.asarray(np.random.uniform(size=(n_particle, 1)) < prob_replay, dtype='f')
            ind = np.random.permutation(range(replay_theta.shape[0]))[:n_particle]
            tmp1 = replay_theta[ind]
            # resample randomness
            tmp2 = init_theta(dim = dimTheta, n_particle=n_particle) #* coin2
            # combine
            theta = coin * tmp1 + (1 - coin) * tmp2
        if iteration % 100 == 0:
            print cost, stepsize, N_train

    save_params(sess, 'save/', filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=32)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--dimH', '-H', type=int, default=20)
    parser.add_argument('--dimH_nn', '-J', type=int, default=20)
    parser.add_argument('--n_particle', '-n', type=int, default=50)
    parser.add_argument('--lbd', type=float, default=0.01)
    parser.add_argument('--T_unroll', '-T', type=int, default=10)
    parser.add_argument('--hsquare', type=float, default=-1.0)
    parser.add_argument('--lr', '-l', type=float, default=0.001)
    parser.add_argument('--n_iter', '-i', type=int, default=500)
    parser.add_argument('--task', type=str, default='crabs')
    parser.add_argument('--method', type=str, default='map')

    args = parser.parse_args()
    main(task = args.task, dimH = args.dimH, dimH_nn = args.dimH_nn, T_unroll = args.T_unroll, 
         hsquare = args.hsquare, lbd = args.lbd, seed = args.seed, 
         batch_size = args.batch_size, n_particle = args.n_particle, 
         lr = args.lr, n_iter = args.n_iter, method = args.method)

