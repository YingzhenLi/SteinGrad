import numpy as np
import tensorflow as tf
from gradients import *

def one_step_dynamics(X, y, theta, data_N, grad_logp_func, method = 'svgd', lr=1e-2, shapes = None):
    # run one step stochastic/deterministic dynamics
    # adagrad with momentum
    b = 1e-6
    a = 0.9

    grad_logp = grad_logp_func(X, y, theta, data_N, shapes)
    if method == 'svgd':
        grad_theta = svgd_gradient(grad_logp, theta)
        theta_new = theta + (lr / tf.sqrt((1 - a) * grad_theta**2  + b)) * grad_theta
    if method == 'entropy':
        grad_theta = entropy_approx_gradient(grad_logp, theta)
        theta_new = theta + (lr / tf.sqrt((1 - a) * grad_theta**2  + b)) * grad_theta
    if method == 'sgld':
        theta_new = theta + lr * grad_logp / 2 + tf.sqrt(lr) * tf.random_normal(theta.get_shape())

    return theta_new

def mcmc(z, logp_func, eps, T, alg = 'hmc'):
    assert alg in ['hmc', 'mala']
    print 'sampling with %s...' % alg
    if alg == 'hmc':
        return hmc(z, logp_func, eps, T)
    if alg == 'mala':
        return mala(z, logp_func, eps, T)

def mala(z, logp_func, eps, T = 1):
    """
    Metropolis Adjusted Langevin Algorithm (MALA)
    """
    
    z_sample = z
    accept_total = 0.0
    logp = logp_func(z_sample)
    grad_logp = tf.gradients(logp, z_sample)[0]
    
    for t in xrange(T):
        noise = tf.random_normal(z_sample.get_shape())
        z_tmp = tf.stop_gradient(z_sample + eps * grad_logp + tf.sqrt(2 * eps) * noise)
        # add rejection step
        log_denominator = logp - 0.5 * tf.reduce_sum(noise**2, 1)
        logp_tmp = logp_func(z_tmp)
        grad_logp_tmp = tf.gradients(logp_tmp, z_tmp)[0]
        log_numerator = logp_tmp - 0.25 / eps * tf.reduce_sum( \
            (z_sample - z_tmp - eps * grad_logp_tmp)**2, 1)
        acceptance_rate = tf.clip_by_value(tf.exp(log_numerator \
                                                  - log_denominator), 0.0, 1.0)
        # accept samples and update related quantities
        u = tf.random_uniform(acceptance_rate.get_shape())
        accept = tf.less_equal(u, acceptance_rate)
        z_sample = tf.select(accept, z_tmp, z_sample)
        logp = tf.select(accept, logp_tmp, logp)
        accept_total += acceptance_rate / T
        if t < T - 1:
            grad_logp = tf.select(accept, grad_logp_tmp, grad_logp)
            
    return z_sample, logp, acceptance_rate

def hmc(z, logp_func, eps, T = 1):
    """
    HMC with T LeapFrog step and rejection
    """

    z_sample = z
    p = tf.random_normal(z_sample.get_shape())

    def leapfrog(z, p, grad_logp):
        # compute the leapfrog step
        p_tmp = p + eps * grad_logp / 2.0
        z_tmp = tf.stop_gradient(z + eps * p_tmp)
        logp_tmp = logp_func(z_tmp)
        grad_logp_tmp = tf.gradients(logp_tmp, z_tmp)[0]
        p_tmp += eps * grad_logp_tmp / 2.0
        return z_tmp, p_tmp, grad_logp_tmp, logp_tmp

    logp = logp_func(z_sample)
    p_tmp = p; z_tmp = z_sample; logp_tmp = logp
    grad_logp_tmp = tf.gradients(logp_tmp, z_tmp)[0]
    # first run leapfrog step
    for t in xrange(T):
        z_tmp, p_tmp, grad_logp_tmp, logp_tmp = leapfrog(z_tmp, p_tmp, grad_logp_tmp)
    # then test acceptance
    acceptance_rate = tf.exp(logp_tmp - 0.5 * tf.reduce_sum(p_tmp ** 2, 1) \
                             - logp + 0.5 * tf.reduce_sum(p ** 2, 1))
    acceptance_rate = tf.minimum(tf.ones(acceptance_rate.get_shape()), acceptance_rate)
    u = tf.random_uniform(acceptance_rate.get_shape())
    accept = tf.less_equal(u, acceptance_rate)
    z_sample = tf.select(accept, z_tmp, z_sample)
    logp = tf.select(accept, logp_tmp, logp)

    return z_sample, logp, acceptance_rate
    
