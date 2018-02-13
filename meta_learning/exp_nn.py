import sys, random, os, scipy.io
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.utils import shuffle
from bnn import evaluate, grad_bnn
from utils import load_uci_data
from gradients import amortised_loss
from nn_sampler import init_nn_sampler
from mcmc import one_step_dynamics
from ksd import KSD
import pickle, os

method = str(sys.argv[1])
train_name = str(sys.argv[2])
test_name = str(sys.argv[3])
seed = int(sys.argv[4])
hsquare = float(sys.argv[5])
lbd = float(sys.argv[6])

np.random.seed(seed)
tf.set_random_seed(seed)

# load data
datapath = # DATA_PATH
X_train, y_train = load_uci_data(datapath, train_name, merge = True)
total_dev, total_test = load_uci_data(datapath, test_name, merge = False)
N_train, dimX = X_train.shape
print "size of training=%d" % (N_train)
dimH = 20
shapes = [(dimX, dimH), (dimH, y_train.shape[1])]
dimTheta = int(np.prod(shapes[0]) + shapes[0][-1] + np.prod(shapes[1]) + shapes[1][-1])

# experiment setup
batch_size = 20
n_particle = 50
lr_train = 0.001

# now define ops
grad_logp_func = grad_bnn

# placeholders for training
dimH_nn = 20
T_unroll = 10
X_ph = tf.placeholder(tf.float32, shape=(batch_size*T_unroll, dimX), name = 'X_ph')
y_ph = tf.placeholder(tf.float32, shape=(batch_size*T_unroll, 1), name = 'y_ph')
theta_ph = tf.placeholder(tf.float32, shape=(n_particle, dimTheta), name = 'theta_ph')
h_ph = tf.placeholder(tf.float32, shape=(n_particle, dimTheta, dimH_nn), name = 'h_ph')
c_ph = tf.placeholder(tf.float32, shape=(n_particle, dimTheta, dimH_nn), name = 'c_ph')
N_ph = tf.placeholder(tf.float32, shape=())

# ops for training the langevin sampler
q_sampler = init_nn_sampler(dimTheta, dimH_nn, grad_logp_func, name = 'lstm_sampler')
loss, theta_q_train, h_q_train, c_q_train = \
    amortised_loss(theta_ph, X_ph, y_ph, h_ph, c_ph, N_ph, 
                   q_sampler, grad_logp_func, method, shapes, T_unroll, hsquare, lbd)
lr_ph = tf.placeholder(tf.float32, shape=(), name = 'lr_ph')
opt = tf.train.AdamOptimizer(learning_rate = lr_ph).minimize(loss)

# ops for testing
n_particle_test = 500
dimX = total_dev[0][0].shape[1]
dimH = 50
shapes = [(dimX, dimH), (dimH, y_train.shape[1])]
dimTheta_train = dimTheta
dimTheta = int(np.prod(shapes[0]) + shapes[0][-1] + np.prod(shapes[1]) + shapes[1][-1])
X_ph_test = tf.placeholder(tf.float32, shape=(batch_size, dimX), name = 'X_ph_test')
y_ph_test = tf.placeholder(tf.float32, shape=(batch_size, 1), name = 'y_ph_test')
theta_ph_test = tf.placeholder(tf.float32, shape=(n_particle_test, dimTheta), name = 'theta_ph_test')
h_ph_test = tf.placeholder(tf.float32, shape=(n_particle_test, dimTheta, dimH_nn), name = 'h_ph_test')
c_ph_test = tf.placeholder(tf.float32, shape=(n_particle_test, dimTheta, dimH_nn), name = 'c_ph_test')

theta_q_test, h_q_test, c_q_test = q_sampler(theta_ph_test, X_ph_test, y_ph_test, \
                                             N_ph, h_ph_test, c_ph_test, shapes)

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

# init the session and all random variables
def init_theta(a0=1, b0=0.1, n_particle=20, dim = 10, seed = None):
    if seed is not None: np.random.seed(seed)
    theta = np.random.normal(loc=np.zeros((n_particle, 1)), scale=1.0,
                size=(n_particle, dim))
    h = np.zeros([n_particle, dim, dimH_nn])
    c = np.zeros([n_particle, dim, dimH_nn])
    return theta, h, c

# load file
n_iter = 1000
filename = "model_T%d_%s_%s_seed%d_hsquare%.2f_lbd%.1f.pkl" % (n_iter, train_name, method, seed, hsquare, lbd)

load_file = False 
if load_file:
    t_vars = tf.trainable_variables()
    params = [v for v in t_vars if 'nn_sampler' in v.name]
    g = open('models/' + filename, 'r')
    print 'load models/'+filename
    params_val = pickle.load(g)
    assign_ops = []
    for val, par in zip(params_val, params):
        assign_ops.append(tf.assign(par, val))
    sess.run(assign_ops)

else:

    # first training the network
    print "Start Training the NN Sampler"
    # langevin sampler

    np.random.seed(seed)
    theta, h, c = init_theta(dim = dimTheta_train, n_particle=n_particle)
    for iteration in tqdm(range(1, n_iter+1)):
        ind = np.random.permutation(range(N_train))[:T_unroll*batch_size]
        _, theta, h, c = sess.run((opt, theta_q_train, h_q_train, c_q_train), \
                               feed_dict={X_ph: X_train[ind], y_ph: y_train[ind], \
                                          lr_ph: lr_train, N_ph: N_train, \
                                          theta_ph: theta, h_ph: h, c_ph: c})
        if (iteration+1) % 100 == 0:
            theta, h, c = init_theta(dim = dimTheta_train, n_particle=n_particle)

# now start testing
total_entropy_acc = []
total_entropy_ll = []

acc_op, ll_op = evaluate(X_ph_test, y_ph_test, theta_ph_test, shapes, activation = 'sigmoid')
grad_test = grad_logp_func(X_ph_test, y_ph_test, theta_ph_test, N_ph, shapes)
ksd_op = KSD(theta_ph_test, grad_test)

def _chunck_eval(sess, X_test, y_test, theta, N):
    N_test = y_test.shape[0]
    acc_total = 0.0; ll_total = 0.0; ksd_total = 0.0
    N_batch = int(N_test / batch_size)
    for i in xrange(N_batch):
        X_batch = X_test[i*batch_size:(i+1)*batch_size]
        y_batch = y_test[i*batch_size:(i+1)*batch_size]
        acc, ll, ksd_val = sess.run((acc_op, ll_op, ksd_op), \
                      feed_dict={X_ph_test: X_batch, y_ph_test: y_batch, \
                                 theta_ph_test: theta, N_ph: N})
        acc_total += acc / N_batch; ll_total += ll / N_batch; ksd_total += ksd_val / N_batch

    return acc_total, ll_total, ksd_total / theta.shape[1]

print "Start testing on separete data set"
T_test = 2000
results = {'acc':[], 'll':[], 'ksd':[]}

for data_i in range(0, len(total_dev)):
    # For each dataset training on dev and testing on test dataset

    X_dev, y_dev = total_dev[data_i]
    X_test, y_test = total_test[data_i]
    X_dev, y_dev = shuffle(X_dev, y_dev)
    X_test, y_test = shuffle(X_test, y_test)
    dev_N = X_dev.shape[0]
    print X_dev.shape, y_dev.shape, X_train.shape, y_train.shape
    total_m_acc = []
    total_m_ll = []
    total_m_ksd = []
    
    # nn sampler
    theta, h, c = init_theta(dim = dimTheta, n_particle = n_particle_test, seed=0)
    ops = (theta_q_test, h_q_test, c_q_test)
    for t in tqdm(range(T_test)):
        ind = np.random.permutation(range(dev_N))[:batch_size] 
        theta, h, c = sess.run(ops, feed_dict={X_ph_test: X_dev[ind], y_ph_test: y_dev[ind], \
                                                   N_ph: dev_N, theta_ph_test: theta, \
                                                   h_ph_test: h, c_ph_test: c})
        if (t+1) % 50 == 0:
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
print "acc", np.mean(results['acc'][:, -1]), np.std(results['acc'][:, -1])
print "ll", np.mean(results['ll'][:, -1]), np.std(results['ll'][:, -1])
print "ksd", np.mean(results['ksd'][:, -1]), np.std(results['ksd'][:, -1])

# save model
if not load_file:
    t_vars = tf.trainable_variables()
    params = [v for v in t_vars if 'langevin_sampler' in v.name]
    params_eval = sess.run(params)
    g = open('models/' + filename, 'w')
    pickle.dump(params_eval, g)
    print "model params saved in models/%s" % filename

## save results
filename = "bnn_%s_%s_hsquare%.2f_lbd%.1f.pkl" % (test_name, method, hsquare, lbd)
savepath = 'results/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)
    print 'create path ' + savepath
f = open(savepath+filename, 'w')
pickle.dump(results, f)
print "results saved in results/%s" % (savepath+filename)

