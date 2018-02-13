import sys, random, os, scipy.io
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.utils import shuffle
from bnn import evaluate, grad_bnn
from utils import load_uci_data
from mcmc import one_step_dynamics
from ksd import KSD
import pickle

test_name = str(sys.argv[1])
seed = int(sys.argv[2])

np.random.seed(seed)
tf.set_random_seed(seed)

# load data
datapath = 'data/'
total_dev, total_test = load_uci_data(datapath, test_name, merge = False)

# experiment setup
batch_size = 32

# now define ops
grad_logp_func = grad_bnn

y_ph = tf.placeholder(tf.float32, shape=(batch_size, 1), name = 'y_ph')
N_ph = tf.placeholder(tf.float32, shape=())
lr_ph = tf.placeholder(tf.float32, shape=(), name = 'lr_ph')

# ops for testing
n_particle_test = 500
dimX = total_dev[0][0].shape[1]
dimH = 50
shapes = [(dimX, dimH), (dimH, 1)]
dimTheta = int(np.prod(shapes[0]) + shapes[0][-1] + np.prod(shapes[1]) + shapes[1][-1])
theta_ph_test = tf.placeholder(tf.float32, shape=(n_particle_test, dimTheta), name = 'theta_test')
X_ph_test = tf.placeholder(tf.float32, shape=(batch_size, dimX), name = 'X_ph_test')
theta_new_sgld = one_step_dynamics(X_ph_test, y_ph, theta_ph_test, N_ph, grad_logp_func, 'sgld', lr_ph, shapes) 

# init the session and all random variables
def init_theta(a0=1, b0=0.1, n_particle=20, dim = 10, seed = None):
    if seed is not None: np.random.seed(seed)
    #alpha0 = np.transpose([np.random.gamma(a0, b0, n_particle)])
    theta0 = np.random.normal(loc=np.zeros((n_particle, 1)), scale=1.0,
                size=(n_particle, dim))
    return theta0

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

acc_op, ll_op = evaluate(X_ph_test, y_ph, theta_ph_test, shapes, activation = 'sigmoid')
grad_test = grad_logp_func(X_ph_test, y_ph, theta_ph_test, N_ph, shapes)
ksd_op = KSD(theta_ph_test, grad_test)

def _chunck_eval(sess, X_test, y_test, theta, N):
    N_test = y_test.shape[0]
    acc_total = 0.0; ll_total = 0.0; ksd_total = 0.0
    N_batch = int(N_test / batch_size)
    for i in xrange(N_batch):
        X_batch = X_test[i*batch_size:(i+1)*batch_size]
        y_batch = y_test[i*batch_size:(i+1)*batch_size]
        acc, ll, ksd_val = sess.run((acc_op, ll_op, ksd_op), \
                      feed_dict={X_ph_test: X_batch, y_ph: y_batch, theta_ph_test: theta, N_ph: N})
        acc_total += acc / N_batch; ll_total += ll / N_batch; ksd_total += ksd_val / N_batch

    return acc_total, ll_total, ksd_total / theta.shape[1]

print "Start testing on separete data set"
T_test = 5000
results = {'acc':[], 'll':[], 'ksd':[]}

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

    # langevin
    seed = 0
    theta = init_theta(dim = dimTheta, n_particle=n_particle_test, seed=seed)
    for t in tqdm(range(T_test)):
        ind = np.random.permutation(range(dev_N))[:batch_size] 
        stepsize = 1e-5 #* (2 ** 11) * (1 + i) ** -0.55
        theta = sess.run(theta_new_sgld, feed_dict={X_ph_test: X_dev[ind], y_ph: y_dev[ind], \
                                                    N_ph: dev_N, theta_ph_test: theta, \
                                                    lr_ph: stepsize})
        if (t+1) % 100 == 0:
            acc, ll, ksd = _chunck_eval(sess, X_test, y_test, theta, dev_N)
            print acc, ll, ksd, t+1, theta[0, 0], theta[0, 1]
            total_m_acc.append(acc)
            total_m_ll.append(ll)
            total_m_ksd.append(ksd)
    
    results['acc'].append(total_m_acc)
    results['ll'].append(total_m_ll)
    results['ksd'].append(total_m_ksd)

    print "Evaluation of dataset=%d" %(data_i)
    print "Evaluation of our methods, ", acc, ll, ksd
    print "\n"


print "Final results"
results['acc'] = np.array(results['acc'])
results['ll'] = np.array(results['ll'])
results['ksd'] = np.array(results['ksd'])

print "\nOur methods----"
print "our acc", np.mean(results['acc'][:, -1]), np.std(results['acc'][:, -1])
print "our ll", np.mean(results['ll'][:, -1]), np.std(results['ll'][:, -1])
print "our ksd", np.mean(results['ksd'][:, -1]), np.std(results['ksd'][:, -1])

## save results
filename = "bnn_%s_sgld.pkl" % test_name
savepath = 'results/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)
    print 'create path ' + savepath
f = open(savepath+filename, 'w')
pickle.dump(results, f)
print "results saved in %s" % (savepath+filename)

