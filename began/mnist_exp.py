import sys
sys.path.append('..')
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import pickle
from utils import *
import shutil
import tensorflow as tf
from mnist_model_conv import construct_model
from helper_functions import entropy_gradient, image_diversity

"""
Training energy models
"""
#np.random.seed(1)
#tf.set_random_seed(1)

method = str(sys.argv[1])
entropy_option = str(sys.argv[2])
gamma = float(sys.argv[3])
eta = float(sys.argv[4])
if len(sys.argv) > 5:
    extra = '_'+str(sys.argv[5])
else:
    extra = ''
assert method in ['kde', 'stein', 'score', 'original']
if method == 'original':
    eta = 0.0
    
# settings
lr = 0.0002
k_t = 0.0
lbd = 0.001

path = # SAVE_PATH
desc = 'mnist_conv_' + method
string = '%s_gamma%.2f_eta%.2f_%s'%(desc, gamma, eta, entropy_option)
models_dir = path + 'models/' + string + extra
results_dir = path + 'results/' + string + extra
images_dir = path + 'images/' + string + extra
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# copy current file to model_dir
file_name = 'mnist_exp.py'
dist_dir = os.path.join(models_dir, file_name)
shutil.copy(file_name, dist_dir)

# construct model  
print "construct model..."
dimZ = 64
dimH = 512
dimF = 64
input_shape = (28, 28, 1)
enc, dec, gen, batch_size_ph = construct_model(input_shape, dimZ, dimF, dimH)
t_vars = tf.trainable_variables()
gen_params = [var for var in t_vars if 'generator' in var.name]
enc_params = [var for var in t_vars if 'encoder' in var.name]
dec_params = [var for var in t_vars if 'decoder' in var.name]
ae_params = enc_params + dec_params

def recon_loss_l2(x):
    x_recon = dec(enc(x))
    return tf.reduce_mean((x - x_recon)**2)

print "define ops..."
batch_size = 100
eta_ph = tf.placeholder(tf.float32, shape=())
X_ph = tf.placeholder(tf.float32, shape=(batch_size,)+input_shape)

# first compute generator images
x_gen = gen(tf.random_uniform(shape=(batch_size, dimZ)) * 2.0 - 1.0)

# objective function for the discriminator (auto-encoder)
k_t_ph = tf.placeholder(tf.float32, shape=())
L_real = recon_loss_l2(X_ph)
L_fake = recon_loss_l2(x_gen)
obj_ae = L_real - k_t_ph * L_fake
obj_gen = L_fake

# compute entropy loss on x
if method in ['kde', 'stein', 'score']:
    x_flat = tf.clip_by_value(tf.reshape(x_gen, [batch_size, -1]), 1e-8, 1.0 - 1e-8)
    entropy_loss, Kxy, h_square = entropy_gradient(x_flat, method)
    obj_gen -= eta_ph * entropy_loss	# i.e. minimise entropy
    if entropy_option == 'kdeH':
        print 'using the KDE entropy estimates...'
        logq = tf.log(tf.reduce_mean(Kxy * 0.75, 1))
        entropy_estimates = -tf.reduce_mean(logq)
    if entropy_option == 'proxyH':
        print 'using the proxy entropy estimates...'
        entropy_estimates = entropy_loss
else:
    entropy_estimates = tf.constant(0.0)	# the original method
    entropy_loss = tf.constant(0.0)

grad_ae = zip(tf.gradients(obj_ae, ae_params), ae_params)
grad_gen = zip(tf.gradients(obj_gen, gen_params), gen_params)
grad = grad_ae + grad_gen

loss_track = L_real + tf.abs(obj_ae)
    
lr_ph = tf.placeholder(tf.float32, shape=())
opt = tf.train.AdamOptimizer(learning_rate=lr_ph, beta1=0.5).apply_gradients(grad)
    
def updateParams(sess, X, lr = 0.0005, k_t = 1.0, eta = 1.0):
    _, l_real, l_fake, entropy = sess.run((opt, L_real, L_fake, entropy_estimates), \
                                  feed_dict={X_ph: X, lr_ph:lr, k_t_ph: k_t,
                                      eta_ph: eta, batch_size_ph: batch_size})
    return l_real, l_fake, entropy

# ops for generating samples
z_sample = tf.placeholder(tf.float32, shape = (500, dimZ))
x_sample = gen(z_sample)

print "loading data..."
path = # DATA_PATH
X_train, X_test, y_train, y_test = mnist_aug(path)
ntrain, ntest = len(X_train), len(X_test)
print X_train.mean(), X_test.mean()

# now check init
print "initialise tensorflow variables..."
sess = tf.Session()
initialised_var = set([])
init_var_list = set(tf.all_variables()) - initialised_var
if len(init_var_list) > 0:
    init = tf.initialize_variables(var_list = init_var_list)
    sess.run(init)

t = time()
np.random.seed(0)
z_random = np.random.randn(batch_size, dimZ)

# training
niter = 100
N_data = X_train.shape[0]
color_plot_images(X_test[:100], (28, 28), images_dir+'/', 'test_data', color=False)

label_entropy_list = []
pixel_entropy_list = []
prob_list = []
dist_list = []

for epoch in range(1, niter+1):
    X_train_now = X_train[np.random.permutation(X_train.shape[0])]

    m_loss_total = 0.0
    l_real_total = 0.0
    l_fake_total = 0.0
    entropy_total = 0.0
    start = time()
    for X_batch in tqdm(iter_data(X_train_now, size=batch_size), total=ntrain/batch_size): 
        l_real, l_fake, entropy = updateParams(sess, X_batch, lr, k_t, eta/gamma)
        m_loss = l_real + np.abs(gamma * l_real - l_fake)
        m_loss_total += m_loss * X_batch.shape[0]
        l_real_total += l_real * X_batch.shape[0]
        l_fake_total += l_fake * X_batch.shape[0]
        entropy_total += entropy * X_batch.shape[0]
        target = gamma * l_real - l_fake
        if method != 'original':
            target += eta * entropy
        k_t += lbd * target
        k_t = min(max(k_t, 0.0), 1.0)

    end = time()
    print 'training for %.3f seconds...' % (end - start)
    print epoch, 'M loss', m_loss_total / N_data, 'recon loss', l_real_total / N_data
    print 'l_fake', l_fake_total / N_data, 'entropy', entropy_total / N_data, 'k_t', k_t
    if epoch % 10 == 0:
        lr *= 0.9

    if epoch == 1 or epoch % 5 == 0:
        z_random = np.random.random([500, dimZ]) * 2.0 - 1.0
        samples = sess.run(x_sample, feed_dict={z_sample:z_random, batch_size_ph:z_random.shape[0]})
        color_plot_images(samples[:100], (28, 28), images_dir+'/', 'samples_%d'%epoch, color=False)
        # compute entropy values
        prob_gen, entropy_gen, dist_gen, neighbours = image_diversity(sess, samples, X_train, y_train)
        color_plot_images(neighbours[:100], (28, 28), images_dir+'/', 'neighbours_%d'%epoch, color=False)
        pixel_entropy = samples.mean(0) + 10e-8
        pixel_entropy = -np.mean(pixel_entropy*np.log(pixel_entropy) + (1-pixel_entropy)*np.log(1-pixel_entropy))
        print "label entropy", entropy_gen, "pixel entropy", pixel_entropy, "avg distance", dist_gen
        label_entropy_list.append(entropy_gen)
        prob_list.append(prob_gen)
        pixel_entropy_list.append(pixel_entropy)
        dist_list.append(dist_gen)

    if epoch == 1 or epoch % 100 == 0:
        params_val = sess.run(gen_params)
        joblib.dump(params_val, models_dir+'/%d_gen_params.jl'%epoch)
        pickle.dump(params_val, open(models_dir+'/%d_gen_params.pkl'%epoch, 'w'))

# save the tracking lists
f = open(results_dir + '/entropy.pkl', 'w')
pickle.dump([label_entropy_list, prob_list, pixel_entropy_list, dist_list], f)
f.close()


