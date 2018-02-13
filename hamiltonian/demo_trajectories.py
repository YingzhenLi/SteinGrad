
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian,\
    sample_gaussian
from plotting import visualise_trajectory, pdf_grid, compare_grad, density, \
                     compute_ksd, expectation, bootstrap
from kernel_hmc.proposals.kmc import KMCStatic
from kernel_hmc.proposals.hmc import HMCBase
import matplotlib.pyplot as plt

import pickle, sys, os
sys.path.append('estimators/')
from stein import KernelExpStein
from stein_nonparam import KernelExpStein as KernelExpSteinNonparam
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Metrics_alt import RCodaTools, autocorr

# banana gradient depends on theano, which is an optional dependency
from banana import Banana, sample_banana
banana_available = True

if __name__ == '__main__':
    """
    Example that visualises trajectories of KMC lite and finite on a simple target.
    C.f. Figures 1 and 2 in the paper.
    """
    
    # for D=2, the fitted log-density is plotted, otherwise trajectory only
    D = 2
    N = 200
    seed = int(sys.argv[1])
    np.random.seed(seed)
    # target is banana density, fallback to Gaussian if theano is not present
    if banana_available:
        b = 0.03; V = 100
        target = Banana(bananicity=b, V=V)
        X = sample_banana(5000, D, bananicity=b, V=V)
        ind = np.random.permutation(range(X.shape[0]))[:N]
        X = X[ind]
        print 'sampling from banana distribution...', X.shape
    else:
        target = IsotropicZeroMeanGaussian(D=D)
        X = sample_gaussian(N=N)
        print 'sampling from gaussian distribution'
        
    # compute sigma
    sigma = np.median(squareform(pdist(X))**2) / np.log(N+1.0) * 2
    M = 200
    if N < M:
        start_samples = np.tile(X, [int(M/N)+1, 1])[:M] + np.random.randn(M, 2) * 2
    else:
        start_samples = X[:M] + np.random.randn(M, 2) * 2
    #start_samples[:, 0] *= 4; start_samples[:, 1] *= 2
    
    # plot trajectories for both KMC lite and finite, parameters are chosen for D=2
    results = []
    num_steps = 2000
    step_size = 0.1
        
    for surrogate in [
                        KernelExpLiteGaussian(sigma=25*sigma, lmbda=0.01, D=D, N=N),
                        KernelExpStein(sigma=16*sigma, lmbda=0.01, D=D, N=N),
                        KernelExpSteinNonparam(sigma=9*sigma, lmbda=0.01, D=D, N=N)
                        
                      ]:
        surrogate.fit(X)
        
        
        # HMC parameters
        momentum = IsotropicZeroMeanGaussian(D=D, sigma=1.0)
        
        # kmc sampler instance
        kmc = KMCStatic(surrogate, target, momentum, num_steps, num_steps, step_size, step_size)
        
        # simulate trajectory from starting point, note _proposal_trajectory is a "hidden" method
        Qs_total = []
        acc_probs_total = []
        accor_total = []
        ksd_total = []
        ess_total = []
        mean_x1_total = []
        np.random.seed(seed+1)
        for i in xrange(M):
            current = start_samples[i]
            current_log_pdf = target.log_pdf(current)
            Qs, acc_probs, log_pdf_q = kmc._proposal_trajectory(current, current_log_pdf)
            # compute auto correlation on first dim
            accor = autocorr(Qs[:, 0])
            accor_total.append(accor)
            Qs_total.append(Qs)
            # compute min ESS
            T_ess = 1800
            ess = RCodaTools.ess_coda_vec(Qs[T_ess+1:])
            ess = np.minimum(ess, Qs[T_ess+1:].shape[0])
            min_ess = np.min(ess)
            ess_total.append(min_ess)
            # compute acceptance prob
            acc_probs_total.append(acc_probs)
            # compute E[x1] estimates for different time t
            e_x1 = expectation(Qs[1:, 0])
            mean_x1_total.append(e_x1)
            
        Qs_total = np.asarray(Qs_total)
        acc_probs_total = np.asarray(acc_probs_total)
        accor_total = np.asarray(accor_total)
        ess_total = np.asarray(ess_total)
        mean_x1_total = np.asarray(mean_x1_total)
        #ksd_total = np.asarray(ksd_total)
        print bootstrap(ess_total), ess_total.mean(), ess_total.max(), ess_total.min()
        # now compute ksd
        T = 100
        k = 0
        while k < num_steps + 1:
            # KSD
            Qs_batch = Qs_total[:, k, :]
            ksd_total.append(compute_ksd(Qs_batch, target))
            k += T
        ksd_total = np.asarray(ksd_total)
        results.append([Qs_total, acc_probs_total, accor_total, ksd_total, ess_total, mean_x1_total])
    
    # now run HMC
    momentum = IsotropicZeroMeanGaussian(D=D, sigma=1.0)
    hmc = HMCBase(target, momentum, num_steps, num_steps, step_size, \
                  step_size, adaptation_schedule=None)
    
    # simulate trajectory from starting point, note _proposal_trajectory is a "hidden" method
    Qs_total = []
    acc_probs_total = []
    accor_total = []
    ksd_total = []
    ess_total = []
    mean_x1_total = []
    np.random.seed(seed+1)
    for i in xrange(M):
        current = start_samples[i]
        current_log_pdf = target.log_pdf(current)
        Qs, acc_probs, log_pdf_q = hmc._proposal_trajectory(current, current_log_pdf)
        # compute auto correlation on first dim
        accor = autocorr(Qs[:, 0])
        accor_total.append(accor)
        Qs_total.append(Qs)
        # compute min ESS
        ess = RCodaTools.ess_coda_vec(Qs[T_ess+1:])
        ess = np.minimum(ess, Qs[T_ess+1:].shape[0])
        min_ess = np.min(ess)
        ess_total.append(min_ess)
        # compute acceptance prob
        acc_probs_total.append(acc_probs)
        # compute E[x1] estimates for different time t
        e_x1 = expectation(Qs[1:, 0])
        mean_x1_total.append(e_x1)
        
    Qs_total = np.asarray(Qs_total)
    acc_probs_total = np.asarray(acc_probs_total)
    accor_total = np.asarray(accor_total)
    ess_total = np.asarray(ess_total)
    mean_x1_total = np.asarray(mean_x1_total)
    #ksd_total = np.asarray(ksd_total)
    print bootstrap(ess_total), ess_total.mean(), ess_total.max(), ess_total.min()
    # now compute ksd
    k = 0
    T = 100
    while k < num_steps + 1:
        Qs_batch = Qs_total[:, k, :]
        ksd_total.append(compute_ksd(Qs_batch, target))
        k += T
    ksd_total = np.asarray(ksd_total)
    results.append([Qs_total, acc_probs_total, accor_total, ksd_total, ess_total, mean_x1_total])
    
    # save results
    results.append(X)
    path = 'results/'
    if not os.path.isdir(path):
        os.mkdir(path)
        print 'crate path ' + path
    filename = path + 'banana_b%.2f_V%d_M%d_num_step%d_N%d_seed%d.pkl' % \
        (b, V, M, num_steps, N, seed)
    pickle.dump(results, open(filename, 'w'))
    print 'results saved in', filename
        
