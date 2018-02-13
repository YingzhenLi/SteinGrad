from kernel_hmc.tools.mcmc_convergence import autocorr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ksd import KSD
from rpy2 import robjects


def geweke(z):
    
    def score(x, first, last, intervals):

        # Initialize list of z-scores
        zscores = []

        # Last index value
        end = x.shape[-1] - 1

        # Start intervals going up to the <last>% of the chain
        last_start_idx = (1 - last) * end

        # Calculate starting indices
        start_indices = np.arange(0, int(last_start_idx), step=int(
            (last_start_idx) / (intervals - 1)))

        # Loop over start indices
        for start in start_indices:
            # Calculate slices
            first_slice = x[:, start: start + int(first * (end - start))]
            last_slice = x[:, int(end - last * (end - start)):]

            z = first_slice.mean(1) - last_slice.mean(1)
            z /= np.sqrt(first_slice.var(1) + last_slice.var(1))

            zscores.append(z)
        
        zscores = np.asarray(zscores)
        print 'geweke', zscores.mean(1).mean(), zscores.var(1).mean()
        
        return zscores.mean(0)
    
    res = []
    first = 0.1; last = 0.5; intervals = 20
    for d in xrange(z.shape[-1]):
        res.append(score(z[:, :, d], first, last, intervals))
    return np.asarray(res).mean(0)

def ess_coda(z):
    # z of shape (M, T, dimZ)
    M, N, dimZ = z.shape
    print z.shape, M, N
    M = float(M); N = float(N)
    
    def ess(z):
    
        B = N * np.var(np.mean(z, axis=1), axis=0, ddof=1)
        W = np.mean(np.var(z, axis=1, ddof=1), axis=0)
        Vhat = (1 - 1 / N) * W + B / N
    
        rho = np.ones(int(N))
        negative_autocorr = False
        t = 1
        while not negative_autocorr and (t < N):
            variogram = np.mean((z[:, t:] - z[:, :-t])**2)
            rho[t] = 1. - variogram / (2. * Vhat)
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0
            t += 1

        if t % 2: t -= 1
        return min(M*N, M * N / (1. + 2 * rho[1:t-1].sum()))
        
    ess0 = ess(z[:, :, 0])
    ess1 = ess(z[:, :, 1])
    return np.asarray([ess0, ess1])

def expectation(Xs):
    # compute EX for different time t
    T = Xs.shape[0]
    mean = np.zeros(Xs.shape)
    mean_old = 0.0
    for t in xrange(T):
        mean[t] = (mean_old * float(t) + Xs[t]) / float(t+1)
        mean_old = mean[t]
        
    return mean

def bootstrap(Xs, K = 10):
    # compute bootstrap mean and std estimates
    X_mean_list = []
    for k in xrange(K):
        Xb = np.zeros(Xs.shape)
        for j in xrange(Xs.shape[0]):
            ind = int(np.random.choice(Xs.shape[0]))
            Xb[ind] = Xs[ind]
        X_mean_list.append(Xb.mean(0))
    X_mean_list = np.asarray(X_mean_list)
    return X_mean_list.mean(0), X_mean_list.std(0)
    
def pdf_grid(Xs, Ys, est):
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)

    # this is in-efficient, log_pdf_multiple on a 2d array is faster
    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
            D[j, i] = est.log_pdf(point)
            G[j, i] = np.linalg.norm(est.grad(point))

    return D, G

def density(Xs, Ys, target):
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)

    # this is in-efficient, log_pdf_multiple on a 2d array is faster
    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
            D[j, i] = target.log_pdf(point)

    return D
    
def compare_grad(Xs, Ys, est, target):
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)

    # this is in-efficient, log_pdf_multiple on a 2d array is faster
    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
            grad_approx = est.grad(point)
            grad_exact = target.grad(point)
            norm_exact = np.linalg.norm(grad_exact)
            
            # l2 distance
            D[j, i] = np.sum((grad_approx - grad_exact)**2) / norm_exact
            # cosine value
            norm_approx = np.linalg.norm(grad_approx)           
            G[j, i] = np.sum(grad_approx*grad_exact) / norm_approx / norm_exact

    return D, G

def compute_R_hat(z):
    # z of shape (M, T, dimZ)
    M, N, dimZ = z.shape
    M = float(M); N = float(N)
    z0 = z[:, :, 0]; z1 = z[:, :, 1]
    
    mean_chain0 = np.mean(z0, axis=1)  
    mean0 = np.mean(mean_chain0)
    B = np.sum((mean_chain0 - mean0)**2) * N / (M-1)    
    #B2 = N * np.var(np.mean(z0, axis=1), axis=0, ddof=1)
    
    var_chain0 = np.sum((z0 - mean_chain0[:, np.newaxis])**2, axis=1) / (N-1)
    W = np.sum(var_chain0) / M    
    #W2 = np.mean(np.var(z0, axis=1, ddof=1), axis=0)

    
    V = (1 - 1 / N) * W + B / N
    R = np.sqrt(V / W)
    print R
    return R
    
def compute_R_hat2(z):
    num_samples = z.shape[1]
    x = z[:, :, 0]
    # Calculate between-chain variance
    B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

    # Calculate within-chain variance
    W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

    # Estimate of marginal posterior variance
    Vhat = W * (num_samples - 1) / num_samples + B / num_samples

    Rhat = np.sqrt(Vhat / W)
    
    print Rhat, Vhat, W, B / num_samples
    return Rhat

def compute_ksd(z, target):
    # assume z is a (K, dimZ) object
    # first compute gradient (slow!)
    grad = z * 0.0
    for i in xrange(z.shape[0]):
        grad[i] = target.grad(z[i])
    return KSD(z, grad)

def visualise_array(Xs, Ys, A, samples=None):
    im = plt.imshow(A, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('Greens')
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], 'bx')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])

def visualise_trajectory(Qs, acc_probs, log_pdf_q, D, log_pdf=None):
    assert Qs.ndim == 2
    
    plot_density = log_pdf is not None and D==2
    
    plt.figure(figsize=(10, 12))
    plt.subplot(411)
    
    # plot density if given and dimension is 2
    if plot_density:
        Xs = np.linspace(-30, 30, 75)
        Ys = np.linspace(-10, 20, len(Xs))
        D, G = pdf_grid(Xs, Ys, log_pdf)
        visualise_array(Xs, Ys, D)
    
    plt.plot(Qs[:, 0], Qs[:, 1], linewidth=3)
    plt.plot(Qs[0, 0], Qs[0, 1], 'r*', markersize=20)
    plt.title("Log-pdf surrogate")
    
    plt.subplot(412)
    if plot_density:
        visualise_array(Xs, Ys, G)
    plt.plot(Qs[:, 0], Qs[:, 1], linewidth=3)
    plt.plot(Qs[0, 0], Qs[0, 1], 'r*', markersize=20)
    plt.title("Gradient norm surrogate")
    
#    plt.subplot(413)
#    plt.title("Acceptance probability")
#    plt.xlabel("Leap frog iteration")
#    plt.plot(acc_probs)
#    plt.plot([0, len(acc_probs)], [np.mean(acc_probs) for _ in range(2)], 'r--')
#    plt.xlim([0, len(acc_probs)])
#    
#    plt.subplot(414)
#    plt.title("Target log-pdf")
#    plt.xlabel("Leap frog iteration")
#    plt.plot(log_pdf_q)
#    plt.xlim([0, len(log_pdf_q)])

def visualise_trace(samples, log_pdf_trajectory, accepted, step_sizes=None, log_pdf_density=None, idx0=0, idx1=1):
    assert samples.ndim == 2
    
    D = samples.shape[1]
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(421)
    plt.plot(samples[:, idx0])
    plt.title("Trace $x_%d$" % (idx0+1))
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    
    plt.subplot(422)
    plt.plot(samples[:, idx1])
    plt.title("Trace $x_%d$" % (idx1+1))
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    
    plt.subplot(423)
    if not log_pdf_density is None and D == 2:
        Xs = np.linspace(-28, 28, 50)
        Ys = np.linspace(-6, 16, len(Xs))
        D, _ = pdf_grid(Xs, Ys, log_pdf_density)
        visualise_array(Xs, Ys, D)
        
    plt.plot(samples[:, idx0], samples[:, idx1])
    plt.title("Trace $(x_%d, x_%d)$" % (idx0+1, idx1+1))
    plt.grid(True)
    plt.xlabel("$x_%d$" % (idx0+1))
    plt.ylabel("$x_%d$" % (idx1+1))
    
    plt.subplot(424)
    plt.plot(log_pdf_trajectory)
    plt.title("log pdf along trajectory")
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    
    plt.subplot(425)
    plt.plot(autocorr(samples[:, idx0]))
    plt.title("Autocorrelation $x_%d$" % (idx0+1))
    plt.xlabel("Lag")
    plt.grid(True)
    
    plt.subplot(426)
    plt.plot(autocorr(samples[:, idx1]))
    plt.title("Autocorrelation $x_%d$" % (idx1+1))
    plt.xlabel("Lag")
    plt.grid(True)
    
    plt.subplot(427)
    plt.plot(np.cumsum(accepted) / np.arange(1, len(accepted)+1))
    plt.title("Average acceptance rate")
    plt.xlabel("MCMC iterations")
    plt.grid(True)
    
    if step_sizes is not None:
        plt.subplot(428)
        if step_sizes.ndim>1:
            for i in range(step_sizes.shape[1]):
                plt.plot(step_sizes[:,i])
            plt.title("Step sizes")
        else:
            plt.plot(step_sizes)
            plt.title("Step size")
            
        plt.xlabel("MCMC iterations")
        plt.grid(True)
