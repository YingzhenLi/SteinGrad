import sys, random, os, scipy.io, pickle, argparse, time
sys.path.extend(['sampler/', 'utils/'])
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.utils import shuffle
from bnn import evaluate, grad_bnn, logp_bnn
from load_uci import load_uci_data
from ksd import KSD
from utils import *

def main(task='pima', dimH=50, seed=42, batch_size_test=32, n_particle_test=500, lr=0.001, 
         method='sgld', dimH_nn=20, n_particle_train=50, T_unroll=10, power=2.0, 
         hsquare=-1.0, lbd=0.01, train_task='crabs'):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # load data
    datapath = 'data/'
    total_dev, total_test = load_uci_data(datapath, task, merge = False)

    # now define ops
    N_ph = tf.placeholder(tf.float32, shape=(), name='N_ph')
    lr_ph = tf.placeholder(tf.float32, shape=(), name='lr_ph')

    # ops for testing
    dimX = total_dev[0][0].shape[1]
    dimY = total_dev[0][1].shape[1]

    print "======================="
    print "define test time model:"
    shapes, dimTheta = make_shapes(dimX, dimH, dimY, n_hidden_layers=1)
    print "network shape:", shapes
    activation_test = 'sigmoid'
    def grad_logp_func_test(X, y, theta, data_N, shapes):
        return grad_bnn(X, y, theta, data_N, shapes, activation=activation_test)

    X_ph_test = tf.placeholder(tf.float32, shape=(batch_size_test, dimX), name = 'X_ph_test')
    y_ph_test = tf.placeholder(tf.float32, shape=(batch_size_test, dimY), name = 'y_ph_test')
    theta_ph_test = tf.placeholder(tf.float32, shape=(n_particle_test, dimTheta), name = 'theta_ph_test')
    if method == 'meta_mcmc':
        from meta_mcmc_sampler import init_nn_sampler
        dimH_nn = 20
        q_sampler = init_nn_sampler(dimH_nn, name = 'nn_sampler')
        theta_q_test = q_sampler(theta_ph_test, X_ph_test, y_ph_test, N_ph, grad_logp_func_test, 
                                 shapes, lr_ph, compute_gamma = True)
    elif method in ['map', 'kde', 'score', 'stein', 'random']:
        from nn_sampler import init_nn_sampler
        dimH_nn = 20
        q_sampler = init_nn_sampler(dimH_nn, name = 'nn_sampler')
        theta_q_test = q_sampler(theta_ph_test, X_ph_test, y_ph_test, N_ph, grad_logp_func_test, 
                                 shapes, lr_ph)
    elif method == 'sgld':
        from mcmc import one_step_dynamics
        theta_q_test = one_step_dynamics(X_ph_test, y_ph_test, theta_ph_test, N_ph, 
                                         grad_logp_func_test, 'sgld', lr_ph, shapes) 
    else:
        raise ValueError('sampler %s not implemented' % method)

    acc_op, ll_op, prob_op = evaluate(X_ph_test, y_ph_test, theta_ph_test, shapes, 
                                      activation = activation_test)
    grad_test = grad_logp_func_test(X_ph_test, y_ph_test, theta_ph_test, N_ph, shapes)
    ksd_op = KSD(theta_ph_test, grad_test)

    def _chunck_eval(sess, X_test, y_test, theta, N):
        N_test = y_test.shape[0]
        acc_total = 0.0; ll_total = 0.0; ksd_total = 0.0
        N_batch = int(N_test / batch_size_test)
        print N_test, batch_size_test, N_batch
        for i in xrange(N_batch):
            X_batch = X_test[i*batch_size_test:(i+1)*batch_size_test]
            y_batch = y_test[i*batch_size_test:(i+1)*batch_size_test]
            acc, ll, prob, ksd_val = sess.run((acc_op, ll_op, prob_op, ksd_op), \
                      feed_dict={X_ph_test: X_batch, y_ph_test: y_batch, \
                                 theta_ph_test: theta, N_ph: N})
            acc_total += acc / N_batch; ll_total += ll / N_batch; ksd_total += ksd_val / N_batch
        print y_batch[:5, 0], prob[:5, 0]
        return acc_total, ll_total, ksd_total / theta.shape[1]

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
    # load file
    if method == 'amc':
        filename = "k%s_H%d_T%d_N%d_p%.1f.pkl" \
                   % (train_task, dimH_nn, T_unroll, n_particle_train, power)
        load_params(sess, 'save/', filename)
    elif method in ['map', 'kde', 'score', 'stein']:
        filename = "%s_H%d_T%d_N%d_%s_hsquare%.2f_lbd%.2f.pkl" \
                   % (train_task, dimH_nn, T_unroll, n_particle_train, method, hsquare, lbd)
        load_params(sess, 'save/', filename)
    else:
        pass

    # now start testing
    total_entropy_acc = []
    total_entropy_ll = []

    print "Start testing on tasks %s with stepsize %.4f, seed %d" % (task, lr, seed)
    T_test = 2000
    T_report = 50
    if task == 'sonar':
        T_test = 5000
        T_report = 100
        print "for sonar dataset, run longer T=%d ..." % T_test 
    results = {'acc':[], 'll':[], 'ksd':[], 'time': []}

    for data_i in range(0, len(total_dev)):
        # For each dataset training on dev and testing on test dataset

        X_dev, y_dev = total_dev[data_i]
        X_test, y_test = total_test[data_i]
        X_dev, y_dev = shuffle(X_dev, y_dev)
        X_test, y_test = shuffle(X_test, y_test)
        dev_N = X_dev.shape[0]
        print X_dev.shape, y_dev.shape
        total_m_acc = []
        total_m_ll = []
        total_m_ksd = []
        total_time = []
    
        # nn sampler
        theta = init_theta(dim = dimTheta, n_particle = n_particle_test, seed=seed)
        # evaluate the start point
        acc, ll, ksd = _chunck_eval(sess, X_test, y_test, theta, dev_N)
        print acc, ll, ksd, 0, theta[0, 0], theta[0, 1], dev_N
        total_m_acc.append(acc)
        total_m_ll.append(ll)
        total_m_ksd.append(ksd)
        total_time.append(0.0)

        start = time.time()
        lr_test = lr #/ dev_N
        for t in tqdm(range(T_test)):
            ind = np.random.randint(0, dev_N, batch_size_test)
            theta = sess.run(theta_q_test, {X_ph_test: X_dev[ind], y_ph_test: y_dev[ind], \
                                            N_ph: dev_N, theta_ph_test: theta, lr_ph: lr})
            if (t+1) % T_report == 0:
                end = time.time()
                acc, ll, ksd = _chunck_eval(sess, X_test, y_test, theta, dev_N)
                print acc, ll, ksd, t+1, theta[0, 0], theta[0, 1], dev_N
                total_m_acc.append(acc)
                total_m_ll.append(ll)
                total_m_ksd.append(ksd)
                total_time.append(end - start)
                start = time.time()
    
        results['acc'].append(total_m_acc)
        results['ll'].append(total_m_ll)
        results['ksd'].append(total_m_ksd)
        results['time'].append(total_time)

        print "Evaluation of dataset=%d" %(data_i)
        print "Evaluation of our methods, ", acc, ll, ksd
        print "\n"


    print "Final results"
    results['acc'] = np.array(results['acc'])
    results['ll'] = np.array(results['ll'])
    results['ksd'] = np.array(results['ksd'])

    print "\nOur methods----"
    print "acc", np.mean(results['acc'][:, -1]), np.std(results['acc'][:, -1])
    print "ll", np.mean(results['ll'][:, -1]), np.std(results['ll'][:, -1])
    print "ksd", np.mean(results['ksd'][:, -1]), np.std(results['ksd'][:, -1])

    ## save results
    if method in ['kde', 'score', 'stein']:
        method = method + '_hsquare%.2f_lbd%.2f' % (hsquare, lbd)

    filename = "bnn_%s_%s.pkl" % (task, method)
    savepath = 'results/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
        print 'create path ' + savepath
    f = open(savepath+filename, 'w')
    pickle.dump(results, f)
    print "results saved in results/%s" % (savepath+filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RVAE experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=32)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--dimH', '-H', type=int, default=50)
    parser.add_argument('--n_particle_test', type=int, default=500)
    parser.add_argument('--lr', '-l', type=float, default=1e-5)
    parser.add_argument('--method', '-m', type=str, default='amc')
    parser.add_argument('--dimH_nn', '-J', type=int, default=20)
    parser.add_argument('--n_particle_train', type=int, default=50)
    parser.add_argument('--T_unroll', '-T', type=int, default=10)
    parser.add_argument('--power', '-p', type=float, default=2.0)
    parser.add_argument('--hsquare', type=float, default=-1.0)
    parser.add_argument('--lbd', type=float, default=0.01)
    parser.add_argument('--task', type=str, default='pima')
    parser.add_argument('--train_task', type=str, default='crabs')

    args = parser.parse_args()
    main(task = args.task, dimH = args.dimH, seed = args.seed, batch_size_test = args.batch_size, 
         n_particle_test = args.n_particle_test, lr = args.lr, method = args.method,
         dimH_nn = args.dimH_nn, n_particle_train = args.n_particle_train, 
         T_unroll = args.T_unroll, power = args.power, hsquare = args.hsquare, lbd = args.lbd,
         train_task = args.train_task)

