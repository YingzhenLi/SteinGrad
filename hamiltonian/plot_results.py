import numpy as np
import pickle, sys
from plotting import visualise_trajectory, pdf_grid, compare_grad, density, compute_ksd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from banana import Banana, sample_banana
from scipy import stats

if __name__ == '__main__':
    D = 2
    N = 200
    M = 200
    seed = int(sys.argv[1])
    num_steps = 2000
    b = 0.03
    V = 100
    scale =16.0
    
    # visualisation
    def visualise_array(ax, Xs, Ys, A, samples=None):
#        im = ax.imshow(A, origin='lower')
#        im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
#        im.set_interpolation('nearest')
#        im.set_cmap('Greens')
        cs = ax.contour(Xs, Ys, A, 5, linewidth=2)
        #ax.clabel(cs, inline=1, fontsize=15, linewidth=2)
        if samples is not None:
            ax.plot(samples[:, 0], samples[:, 1], 'bx')
        ax.set_ylim([Ys.min(), Ys.max()])
        ax.set_xlim([Xs.min(), Xs.max()])
    
    target = Banana(bananicity=b, V=V)
    
    names = [r'$Score$', r'$Stein\ param$', r'$Stein\ nonparam$', r'$HMC$']
    fig, ax = plt.subplots(2, 4, figsize=(10, 3.5),)
    color = ['g', 'm', 'b', 'r']
    Xs = np.linspace(-30, 30, 200)
    Ys = np.linspace(-10, 30, len(Xs))
    pdf = density(Xs, Ys, target)
    
    # first load data
    path = 'results/'
    plot_results = []
    
    filename = path + 'banana_b%.2f_V%d_M%d_num_step%d_N%d_seed%d.pkl' % \
        (b, V, M, num_steps, N, seed)
    print filename
    results = pickle.load(open(filename, 'r'))
    X = results[-1]	# datapoints to fit estimators
    
    for i in xrange(4):
        Qs_list, acc_probs, accor, ksd, ess, mean_x1 = results[i]
        print np.mean(ess), names[i]
        #ksd = np.mean(ksd, 0)
        Qs = Qs_list.reshape(M*(num_steps+1), D)
        
#        # fit with kde
#        Qs_batch = Qs_list[:, -50:, :].reshape(-1, D)
#        kernel = stats.gaussian_kde(Qs_batch.T)
#        # not efficient!
#        pdf = np.zeros((len(Xs), len(Xs)))
#        for j, x in enumerate(Xs):
#            loc = np.vstack((np.ones(len(Xs)) * x, Ys))
#            pdf[:, j] = kernel(loc) 
        visualise_array(ax[0, i], Xs, Ys, pdf)    
        ax[0, i].scatter(Qs[:, 0], Qs[:, 1], s=3, facecolors='none', \
                         edgecolors = color[i], alpha = 0.05, linewidth=0.5)
        # visualise data
        if i < 3:
            ax[0, i].plot(X[:, 0], X[:, 1], 'c+', linewidth=1, alpha = 0.5)
        K = -2
        ax[0, i].plot(Qs_list[K, :200, 0], Qs_list[K, :200, 1], 'y', linewidth=3)
        ax[0, i].plot(Qs_list[K, 0, 0], Qs_list[K, 0, 1], 'y*', markersize=15)
        ax[0, i].set_title(names[i])
               
        #ax[0, i].set_aspect(0.7)
        
        # show auto correlation
        acc_mean = np.mean(accor, 0)
        xaxis = np.arange(0, acc_mean.shape[0])
        acc_ste = np.sqrt(np.var(accor) / accor.shape[0])
        ax[1, 0].plot(xaxis, acc_mean, '%s-'%color[i], linewidth=3)
        ax[1, 0].fill_between(xaxis, acc_mean - acc_ste, acc_mean + acc_ste, \
                                  color=color[i], alpha=0.2, linewidth=0)
        ax[1, 0].set_xlabel(r'$Lag$')
        ax[1, 0].set_title(r'$autocorrelation$')
                                  
        # show acceptance
        acc_mean = np.mean(acc_probs, 0)
        acc_ste = np.sqrt(np.var(acc_probs) / acc_probs.shape[0])
        ax[1, 1].plot(xaxis, acc_mean, '%s-'%color[i], linewidth=3)
        ax[1, 1].fill_between(xaxis, acc_mean - acc_ste, acc_mean + acc_ste, \
                              color=color[i], alpha=0.2, linewidth=0)
        ax[1, 1].set_xlabel(r'$iteration$')
        ax[1, 1].set_title(r'$acceptance \ rate$') 
        ax[1, 1].set_yticks([0.65, 0.75, 0.85, 0.95, 1.05])
        
        # show ksd
        T = 100
        xaxis = np.arange(0, num_steps+1, T) 
        ax[1, 2].plot(xaxis, ksd, '%s-'%color[i], linewidth=3, label=names[i]) 
        ax[1, 2].set_xlabel(r'$iteration$')
        ax[1, 2].set_title(r'$KSD$')    
        
        # show mean of x1
        mean_x1_mean = np.mean(mean_x1, 0)
        xaxis = np.arange(1, len(mean_x1_mean)+1, 1)
        mean_x1_ste = np.sqrt(np.var(mean_x1, 0) / mean_x1.shape[0]) 
        ax[1, 3].plot(xaxis, mean_x1_mean, '%s-'%color[i], linewidth=3, label=names[i]) 
        ax[1, 3].fill_between(xaxis, mean_x1_mean - mean_x1_ste, mean_x1_mean + mean_x1_ste, \
                              color=color[i], alpha=0.2, linewidth=0)
        ax[1, 3].set_xlabel(r'$iteration$')
        ax[1, 3].set_title(r'$E[x_1]$')        
    
    #ax[1, 2].legend(frameon=False, labelspacing=0.15, fontsize = 'medium')
    plt.tight_layout()
    plt.savefig('banana_seed%d.png' % seed, format='png', bbox_inches='tight')      
    print 'save at banana_seed%d.png' % seed 
    #plt.show()
    
