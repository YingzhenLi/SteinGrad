import numpy as np
import tensorflow as tf

"""
Define the gradients and the amortisation loss
"""

def RBF_kernel(z, K, h_square = -1.0):
    z_ = tf.expand_dims(z, 1)   # (K, 1, dimZ) 
    pdist_square = tf.reduce_sum((z - tf.stop_gradient(z_))**2, -1)	# shape (K, K)
        
    # use median
    if h_square < 10e-3:
        pdist_vec = tf.reshape(pdist_square, [K * K])
        top_k, _ = tf.nn.top_k(pdist_vec, k = int(K**2/2))
        top_k = tf.transpose(top_k)
        median = tf.squeeze(tf.gather(top_k, int(K**2/2)-1)) + 0.01        
        h_square = tf.stop_gradient(0.5 * median / tf.log(K+1.0))
    
    kzz = tf.exp(-pdist_square / h_square / 2.0)  # shape (K, K)
    
    return kzz, h_square

def kde_approx_gradient(grad_logp, theta, hsquare = -1.0, lbd = 0.01):
    K = theta.get_shape().as_list()[0]
    Kxy, h_square = RBF_kernel(theta, K, hsquare)
    sumkxy = tf.reduce_sum(Kxy, -1, keep_dims=True)
    dxkxy = (-tf.matmul(Kxy, theta) + theta * sumkxy) / h_square
    entropy_grad = dxkxy / sumkxy
    gamma = 1.0                  
    return grad_logp + gamma * entropy_grad
    
def score_approx_gradient(grad_logp, z, hsquare = -1.0, lbd = 0.1):
    K, dimZ = z.get_shape().as_list()
    Kxy, h_square = RBF_kernel(z, K, hsquare)
    b = -tf.reduce_mean(Kxy * (dimZ + tf.log(Kxy+10e-8)*2), 1, keep_dims=True)
    # now compute C
    Kxy_ = tf.expand_dims(Kxy, 2)
    Kxy_Dx = tf.transpose(Kxy_ * z, [2, 0, 1])
    Dx_Kxy = tf.transpose(tf.transpose(Kxy_, [1, 0, 2]) * z, [2, 1, 0])
    C = tf.matmul(Dx_Kxy - Kxy_Dx, Kxy_Dx - Dx_Kxy)
    C = tf.reduce_mean(C, 0)	# shape (K, K)
    delta = tf.diag(tf.ones(shape=(K,))) * lbd
    print 'delta', lbd
    a = -h_square * tf.matrix_solve(C + delta, b)
    grad = tf.matmul(Kxy, a * z) - tf.reduce_sum(Kxy * tf.squeeze(a), 1, keep_dims=True) * z
    entropy_grad = -grad
    gamma = 1.0                         
    return grad_logp + gamma * entropy_grad

def stein_approx_gradient(grad_logp, theta, hsquare = -1.0, lbd = 1.0):
    K = theta.get_shape().as_list()[0]
    Kxy, h_square = RBF_kernel(theta, K, hsquare)
    sumkxy = tf.reduce_sum(Kxy, -1, keep_dims=True)
    delta = tf.diag(tf.ones(shape=(K,))) * lbd
    entropy_grad = (-theta + tf.matrix_solve(Kxy+delta, theta * sumkxy)) / h_square
    gamma = 1.0
    return grad_logp + gamma * entropy_grad

def amortised_loss(theta_init, X, y, data_N, sampler, grad_logp_func, 
                   method = 'map', shapes = None, T = 10, hsquare=-1.0, 
                   lbd = 0.01, stepsize = 1e-5):
    # first get samples from the langevin sampler
    loss = 0.0
    batch_size = y.get_shape().as_list()[0] / T
    
    for t in xrange(T):
        X_batch = X[t*batch_size:(t+1)*batch_size]
        y_batch = y[t*batch_size:(t+1)*batch_size]
        theta = sampler(theta_init, X_batch, y_batch, data_N, grad_logp_func, shapes, stepsize)
        # second compute moving direction
        grad_logp = grad_logp_func(X_batch, y_batch, theta, data_N, shapes)
        # stop gradient for grad_logp, see deepmind paper
        grad_logp = tf.stop_gradient(grad_logp)
        if method == 'kde':
            grad_theta = kde_approx_gradient(grad_logp, theta, hsquare, lbd)
        if method == 'score':
            print 'hsquare', hsquare, 'lbd', lbd
            grad_theta = score_approx_gradient(grad_logp, theta, hsquare, lbd)
        if method == 'stein':
            grad_theta = stein_approx_gradient(grad_logp, theta, hsquare, lbd)
        if method == 'map':
            grad_theta = grad_logp
        loss -= tf.reduce_mean(tf.stop_gradient(grad_theta) * theta)
        theta_init = theta
        if (t+1) % 10 == 0:
            theta_init = tf.stop_gradient(theta_init)
 
    return loss, theta
    
